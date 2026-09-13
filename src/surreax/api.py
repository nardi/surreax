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
from surreax.error_search import high_error_points, worst_case
from surreax.fitted import FittedSurrogate
from surreax.fitting import fit_linear, fit_linear_sobolev, fit_linear_sparse
from surreax.flatten import Unraveler, ravel, ravel_example
from surreax.forms.difference_max_affine import (
    DifferenceOfMaxAffineForm,
    fit_difference_max_affine,
)
from surreax.forms.max_affine import MaxAffineForm, fit_max_affine
from surreax.forms.network import NetworkForm, fit_network, parameter_count
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


@eqx.filter_jit
def _error_report(
    surrogate: FittedSurrogate,
    target: Target,
    inputs: Float[Array, "n d"],
    unravel: Unraveler,
    structure_cost: int,
    worst_case_value: Float[Array, ""],
    worst_case_point: Float[Array, " d"],
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
        worst_case=worst_case_value,
        worst_case_point=worst_case_point,
        structure_cost=structure_cost,
    )


def fit(
    target: Target,
    domain: Domain,
    form: LinearForm | MaxAffineForm | DifferenceOfMaxAffineForm | NetworkForm,
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    derivative_weight: float = 0.0,
    sparse_terms: int | None = None,
    reduce_inputs: reduction.InputReducer | None = None,
    reduce_outputs: reduction.OutputReducer | None = None,
    worst_case_starts: int = 8,
    worst_case_steps: int = 200,
    worst_case_rate: float = 0.05,
    inputs: Float[Array, "n d"] | None = None,
    seed: int = 0,
) -> FittedSurrogate:
    """Fit a surrogate of `form` to `target` over `domain`.

    Samples the domain with a low-discrepancy sequence, holds out a fraction for
    the error report, and fits the form by one regularized linear solve. A
    positive `derivative_weight` turns on Sobolev training, which fits against
    the target Jacobian alongside its values. A `sparse_terms` budget fits a
    linear form by matching pursuit, keeping only that many basis terms. A
    `reduce_inputs` or `reduce_outputs` builder fits the model in a smaller
    space. The report's worst-case error is searched from `worst_case_starts`
    random points, and setting that to zero skips the search. Passing `inputs`
    fits on those flat points instead of sampling the domain, which is how
    adaptive sampling refits on a growing set.
    """
    if inputs is None:
        inputs = sampling.sample(domain, samples, seed=seed)
    holdout_count = max(1, round(inputs.shape[0] * holdout))
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

    # The target Jacobian in the model's coordinates supervises the derivative,
    # shared by the linear Sobolev fit and the network fit.
    target_jacobians = None
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

    match form:
        case LinearForm():
            if sparse_terms is not None:
                model = fit_linear_sparse(
                    form, normalized_inputs, normalized_outputs, terms=sparse_terms, ridge=ridge
                )
            elif target_jacobians is not None:
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
            # Count the basis terms the fit actually uses, which the sparse fit trims.
            structure_cost = int(jnp.sum(jnp.any(model.weights != 0, axis=0)))
        case MaxAffineForm():
            model = fit_max_affine(
                form, normalized_inputs, normalized_outputs, key=jax.random.key(seed)
            )
            structure_cost = model.slopes.shape[1]
        case DifferenceOfMaxAffineForm():
            model = fit_difference_max_affine(
                form, normalized_inputs, normalized_outputs, key=jax.random.key(seed)
            )
            structure_cost = model.positive_slopes.shape[1] + model.negative_slopes.shape[1]
        case NetworkForm():
            model = fit_network(
                form,
                normalized_inputs,
                normalized_outputs,
                key=jax.random.key(seed),
                jacobians=target_jacobians,
                derivative_weight=derivative_weight,
            )
            structure_cost = parameter_count(model)
        case _:
            raise TypeError(f"unsupported form: {type(form).__name__}")

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
            worst_case=jnp.asarray(0.0),
            worst_case_point=jnp.zeros(input_dimension),
            structure_cost=structure_cost,
        ),
    )
    if worst_case_starts > 0:
        worst_case_value, worst_case_point = worst_case(
            surrogate,
            target,
            domain,
            starts=worst_case_starts,
            steps=worst_case_steps,
            learning_rate=worst_case_rate,
            key=jax.random.key(seed + 1),
            seed_points=holdout_inputs,
        )
    else:
        worst_case_value = jnp.asarray(0.0)
        worst_case_point = jnp.zeros(input_dimension)
    report = _error_report(
        surrogate,
        target,
        holdout_inputs,
        domain.unravel,
        structure_cost,
        worst_case_value,
        worst_case_point,
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)


def fit_adaptive(
    target: Target,
    domain: Domain,
    form: LinearForm | MaxAffineForm | DifferenceOfMaxAffineForm | NetworkForm,
    *,
    initial_samples: int = 256,
    points_per_round: int = 32,
    rounds: int = 5,
    tolerance: float = 1e-3,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    derivative_weight: float = 0.0,
    worst_case_starts: int = 8,
    worst_case_steps: int = 200,
    worst_case_rate: float = 0.05,
    seed: int = 0,
) -> FittedSurrogate:
    """Fit while adding samples where the surrogate is worst.

    Fits on an initial low-discrepancy sample, then repeats: locate the
    highest-error points with the domain search, add them to the sample, and
    refit. Stops when the worst-case error falls below `tolerance` or after
    `rounds`. Because the target is callable, the sample is not a fixed resource,
    so this spends points where the fit needs them.
    """
    inputs = sampling.sample(domain, initial_samples, seed=seed)
    surrogate = fit(
        target,
        domain,
        form,
        holdout=holdout,
        ridge=ridge,
        derivative_weight=derivative_weight,
        worst_case_starts=worst_case_starts,
        worst_case_steps=worst_case_steps,
        worst_case_rate=worst_case_rate,
        inputs=inputs,
        seed=seed,
    )
    for extra in range(rounds):
        if float(surrogate.report.worst_case) < tolerance:
            break
        new_points = high_error_points(
            surrogate,
            target,
            domain,
            count=points_per_round,
            steps=worst_case_steps,
            learning_rate=worst_case_rate,
            key=jax.random.key(seed + extra + 2),
            seed_points=inputs,
        )
        inputs = jnp.concatenate([inputs, new_points])
        surrogate = fit(
            target,
            domain,
            form,
            holdout=holdout,
            ridge=ridge,
            derivative_weight=derivative_weight,
            worst_case_starts=worst_case_starts,
            worst_case_steps=worst_case_steps,
            worst_case_rate=worst_case_rate,
            inputs=inputs,
            seed=seed,
        )
    return surrogate
