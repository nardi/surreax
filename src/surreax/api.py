"""The fit entry point that wires the pipeline stages together.

`fit` samples the domain, evaluates the target, fits the chosen form by one
linear solve in a normalized reduced space, and measures held-out error. It
returns a `FittedSurrogate` with the target's signature.
"""

import functools
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from surreax import normalize, reduction, sampling
from surreax.domain import Domain
from surreax.fitted import FittedSurrogate
from surreax.fitting import fit_linear, fit_linear_sobolev
from surreax.flatten import Unraveler, ravel, ravel_example
from surreax.forms.protocol import LinearForm
from surreax.normalize import Normalizer
from surreax.reduction import LinearReduction
from surreax.report import ErrorReport

Target = Callable[[PyTree[Array]], PyTree[Array]]
"""A pure JAX function from one input pytree to a pytree of outputs."""


def _flat_target(
    point: Float[Array, " d"], *, target: Target, domain: Domain
) -> Float[Array, " k"]:
    """The target evaluated at a flat input, flattened to a vector."""
    return ravel(target(domain.unravel(point)))


def _normalized_target(
    point: Float[Array, " d"],
    *,
    target: Target,
    domain: Domain,
    input_reduction: LinearReduction,
    input_normalizer: Normalizer,
    output_reduction: LinearReduction,
    output_normalizer: Normalizer,
) -> Float[Array, " k"]:
    """The target in the model's normalized reduced coordinates.

    Inverts the input transform to reach a real input, evaluates the target, and
    applies the output transform, so its Jacobian is the Jacobian the model
    fits against.
    """
    flat_input = input_reduction.inverse(input_normalizer.inverse(point))
    flat_output = ravel(target(domain.unravel(flat_input)))
    return output_normalizer.forward(output_reduction.forward(flat_output))


def _flatten_batch(trees: PyTree[Array]) -> Float[Array, "n k"]:
    """Flatten each example of a batched pytree into a row."""
    return jax.vmap(ravel)(trees)


def _error_report(
    surrogate: FittedSurrogate,
    target: Target,
    inputs: Float[Array, "n d"],
    unravel: Unraveler,
) -> ErrorReport:
    """Measure value and derivative error against the target on held-out points.

    Both functions take a pytree input, so they are composed with `unravel` to
    run in the flat input space the held-out points live in.
    """

    def flat_surrogate(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(surrogate(unravel(point)))

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(unravel(point)))

    predicted = jax.vmap(flat_surrogate)(inputs)
    actual = jax.vmap(flat_target)(inputs)
    value_residual = jnp.linalg.norm(predicted - actual, axis=1)

    surrogate_jacobian = jax.vmap(jax.jacobian(flat_surrogate))(inputs)
    target_jacobian = jax.vmap(jax.jacobian(flat_target))(inputs)
    sample_count = inputs.shape[0]
    jacobian_residual = jnp.linalg.norm(
        (surrogate_jacobian - target_jacobian).reshape(sample_count, -1), axis=1
    )

    def reconstruct(output: Float[Array, " k"]) -> Float[Array, " k"]:
        reduction = surrogate.output_reduction
        return reduction.inverse(reduction.forward(output))

    projection_residual = jnp.linalg.norm(actual - jax.vmap(reconstruct)(actual), axis=1)
    return ErrorReport(
        value_mean=jnp.mean(value_residual),
        value_max=jnp.max(value_residual),
        derivative_mean=jnp.mean(jacobian_residual),
        derivative_max=jnp.max(jacobian_residual),
        projection_mean=jnp.mean(projection_residual),
        projection_max=jnp.max(projection_residual),
    )


def fit(
    target: Target,
    domain: Domain,
    form: LinearForm,
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    derivative_weight: float = 0.0,
    reduce_inputs: reduction.InputReducer | None = None,
    reduce_outputs: reduction.OutputReducer | None = None,
    seed: int = 0,
) -> FittedSurrogate:
    """Fit a surrogate of `form` to `target` over `domain`.

    Samples the domain with a low-discrepancy sequence, holds out a fraction for
    the error report, and fits the form by one regularized linear solve. A
    positive `derivative_weight` turns on Sobolev training, which fits against
    the target Jacobian alongside its values. A `reduce_inputs` or
    `reduce_outputs` builder fits the model in a smaller space.
    """
    inputs = sampling.sample(domain, samples, seed=seed)
    holdout_count = max(1, round(samples * holdout))
    train_inputs = inputs[:-holdout_count]
    holdout_inputs = inputs[-holdout_count:]

    def evaluate(point: Float[Array, " d"]) -> PyTree[Array]:
        return target(domain.unravel(point))

    train_outputs = jax.vmap(evaluate)(train_inputs)
    _, output_unravel = ravel_example(jax.tree.map(lambda leaf: leaf[0], train_outputs))
    flat_outputs = _flatten_batch(train_outputs)

    input_dimension = inputs.shape[1]
    output_dimension = flat_outputs.shape[1]

    if reduce_inputs is None:
        input_reduction = reduction.identity(input_dimension)
    else:
        flat_target = functools.partial(_flat_target, target=target, domain=domain)
        train_jacobians = jax.vmap(jax.jacobian(flat_target))(train_inputs)
        input_reduction = reduce_inputs(train_inputs, train_jacobians)
    output_reduction = (
        reduction.identity(output_dimension)
        if reduce_outputs is None
        else reduce_outputs(flat_outputs)
    )

    encoded_inputs = jax.vmap(input_reduction.forward)(train_inputs)
    if reduce_inputs is None:
        # An identity reduction keeps the domain box, so map it to [-1, 1].
        lower, upper = domain.bounds()
        input_normalizer = normalize.from_bounds(lower, upper)
    else:
        # A reduced space has no box, so center and whiten from the samples.
        input_normalizer = normalize.from_moments(encoded_inputs)
    normalized_inputs = jax.vmap(input_normalizer.forward)(encoded_inputs)

    encoded_outputs = jax.vmap(output_reduction.forward)(flat_outputs)
    output_normalizer = normalize.from_moments(encoded_outputs)
    normalized_outputs = jax.vmap(output_normalizer.forward)(encoded_outputs)

    if derivative_weight > 0.0:
        transform = functools.partial(
            _normalized_target,
            target=target,
            domain=domain,
            input_reduction=input_reduction,
            input_normalizer=input_normalizer,
            output_reduction=output_reduction,
            output_normalizer=output_normalizer,
        )
        target_jacobians = jax.vmap(jax.jacobian(transform))(normalized_inputs)
        model = fit_linear_sobolev(
            form,
            normalized_inputs,
            normalized_outputs,
            target_jacobians,
            ridge=ridge,
            derivative_weight=derivative_weight,
        )
    else:
        model = fit_linear(form, normalized_inputs, normalized_outputs, ridge=ridge)

    surrogate = FittedSurrogate(
        input_reduction=input_reduction,
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=output_normalizer,
        output_reduction=output_reduction,
        output_unravel=output_unravel,
        domain=domain,
        report=ErrorReport(
            value_mean=jnp.asarray(0.0),
            value_max=jnp.asarray(0.0),
            derivative_mean=jnp.asarray(0.0),
            derivative_max=jnp.asarray(0.0),
            projection_mean=jnp.asarray(0.0),
            projection_max=jnp.asarray(0.0),
        ),
    )
    report = _error_report(surrogate, target, holdout_inputs, domain.unravel)
    return eqx.tree_at(lambda s: s.report, surrogate, report)
