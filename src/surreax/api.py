"""The fit entry point that wires the pipeline stages together.

`fit` samples the domain, evaluates the target, fits the chosen form by one
linear solve in a normalized reduced space, and measures held-out error. It
returns a `FittedSurrogate` with the target's signature.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from surreax import normalize, reduction, sampling
from surreax.domain import Domain
from surreax.fitted import FittedSurrogate
from surreax.fitting import fit_linear
from surreax.flatten import ravel, ravel_example
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport

Target = Callable[[Float[Array, " d"]], PyTree[Array]]
"""A pure JAX function from one flat input to a pytree of outputs."""


def _flatten_batch(trees: PyTree[Array]) -> Float[Array, "n k"]:
    """Flatten each example of a batched pytree into a row."""
    return jax.vmap(ravel)(trees)


def _error_report(
    surrogate: FittedSurrogate, target: Target, inputs: Float[Array, "n d"]
) -> ErrorReport:
    """Measure value and derivative error against the target on held-out points."""
    predicted = _flatten_batch(jax.vmap(surrogate)(inputs))
    actual = _flatten_batch(jax.vmap(target)(inputs))
    value_residual = jnp.linalg.norm(predicted - actual, axis=1)

    def flat_surrogate(x: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(surrogate(x))

    def flat_target(x: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(x))

    surrogate_jacobian = jax.vmap(jax.jacobian(flat_surrogate))(inputs)
    target_jacobian = jax.vmap(jax.jacobian(flat_target))(inputs)
    sample_count = inputs.shape[0]
    jacobian_residual = jnp.linalg.norm(
        (surrogate_jacobian - target_jacobian).reshape(sample_count, -1), axis=1
    )
    return ErrorReport(
        value_mean=jnp.mean(value_residual),
        value_max=jnp.max(value_residual),
        derivative_mean=jnp.mean(jacobian_residual),
        derivative_max=jnp.max(jacobian_residual),
    )


def fit(
    target: Target,
    domain: Domain,
    form: LinearForm,
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    seed: int = 0,
) -> FittedSurrogate:
    """Fit a surrogate of `form` to `target` over `domain`.

    Samples the domain with a low-discrepancy sequence, holds out a fraction for
    the error report, and fits the form by one regularized linear solve.
    """
    inputs = sampling.sample(domain, samples, seed=seed)
    holdout_count = max(1, round(samples * holdout))
    train_inputs = inputs[:-holdout_count]
    holdout_inputs = inputs[-holdout_count:]

    train_outputs = jax.vmap(target)(train_inputs)
    _, output_unravel = ravel_example(jax.tree.map(lambda leaf: leaf[0], train_outputs))
    flat_outputs = _flatten_batch(train_outputs)

    input_dimension = inputs.shape[1]
    output_dimension = flat_outputs.shape[1]
    input_reduction = reduction.identity(input_dimension)
    output_reduction = reduction.identity(output_dimension)

    lower, upper = domain.bounds()
    input_normalizer = normalize.from_bounds(lower, upper)
    encoded_inputs = jax.vmap(input_reduction.forward)(train_inputs)
    normalized_inputs = jax.vmap(input_normalizer.forward)(encoded_inputs)

    encoded_outputs = jax.vmap(output_reduction.forward)(flat_outputs)
    output_normalizer = normalize.from_moments(encoded_outputs)
    normalized_outputs = jax.vmap(output_normalizer.forward)(encoded_outputs)

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
        ),
    )
    report = _error_report(surrogate, target, holdout_inputs)
    return eqx.tree_at(lambda s: s.report, surrogate, report)
