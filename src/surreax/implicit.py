"""Implicit fitting: fitting a target defined by a residual `g(x, y) = 0`.

The target `f` has `g(x, f(x))` near zero. Fitting it reuses the explicit engine
through small conversions. `fit_data` fits a form to given input-output pairs,
`root_find` and `refine` turn a residual, or a residual and a surrogate, into a
solve-at-call approximator, and `fit_implicit` orchestrates these through a
strategy `method`.

The default `InverseMap` method inverts the residual: writing `r = g(x, y)`, the
map `(x, y) -> (x, r)` inverts to `y = h(x, r)`, and `f(x) = h(x, 0)`. The pairs
`((x, r), y)` come from sampling `(x, y)` and evaluating `g` forward, so `h` is
an ordinary regression and no solve is involved. The `Objective` method fits `f`
directly by minimizing the residual it leaves.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float

from surreax import normalize, reduction, sampling
from surreax.domain import Domain, box_from_data
from surreax.fitted import FittedSurrogate
from surreax.fitting import LinearModel, fit_linear
from surreax.flatten import ravel_example
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport

Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""A residual `g(x, y)`, from an input and an unknown to the residual vector."""

Surrogate = Callable[[Float[Array, " n"]], Float[Array, " m"]]
"""Any approximator of the target, from a flat input to the unknown."""


def _zero_report(dimension: int) -> ErrorReport:
    zero = jnp.asarray(0.0)
    return _value_report(zero, zero, dimension, 0)


def _value_report(
    value_mean: Float[Array, ""],
    value_max: Float[Array, ""],
    dimension: int,
    structure_cost: int,
) -> ErrorReport:
    zero = jnp.asarray(0.0)
    return ErrorReport(
        value_mean=value_mean,
        value_max=value_max,
        derivative_mean=zero,
        derivative_max=zero,
        projection_mean=zero,
        projection_max=zero,
        worst_case=zero,
        worst_case_point=jnp.zeros(dimension),
        structure_cost=structure_cost,
    )


def fit_data(
    inputs: Float[Array, "s d"],
    outputs: Float[Array, "s k"],
    form: LinearForm,
    *,
    holdout: float = 0.2,
    ridge: float = 1e-8,
) -> FittedSurrogate:
    """Fit a linear form to given input-output pairs, with no target callable.

    Normalizes from sample moments, since the pairs carry no domain box, and
    reports the held-out prediction error. This is the array counterpart of
    `fit`, used where the data is pairs rather than a callable.
    """
    holdout_count = max(1, round(inputs.shape[0] * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]
    train_outputs, holdout_outputs = outputs[:-holdout_count], outputs[-holdout_count:]

    input_reduction = reduction.identity(inputs.shape[1])
    output_reduction = reduction.identity(outputs.shape[1])
    input_normalizer = normalize.from_moments(train_inputs)
    output_normalizer = normalize.from_moments(train_outputs)
    normalized_inputs = jax.vmap(input_normalizer.forward)(train_inputs)
    normalized_outputs = jax.vmap(output_normalizer.forward)(train_outputs)
    model = fit_linear(form, normalized_inputs, normalized_outputs, ridge=ridge)
    _, output_unravel = ravel_example(train_outputs[0])

    surrogate = FittedSurrogate(
        input_reduction=input_reduction,
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=output_normalizer,
        output_reduction=output_reduction,
        output_unravel=output_unravel,
        domain=box_from_data(inputs),
        report=_zero_report(inputs.shape[1]),
    )
    predicted = jax.vmap(surrogate)(holdout_inputs)
    residual = jnp.linalg.norm(predicted - holdout_outputs, axis=1)
    report = _value_report(
        jnp.mean(residual), jnp.max(residual), inputs.shape[1], model.weights.shape[1]
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)


def _newton_step(
    residual: Residual, x: Float[Array, " n"], y: Float[Array, " m"]
) -> Float[Array, " m"]:
    """One Newton step in the unknown, `y - (dg/dy)^{-1} g(x, y)`."""
    jacobian = jax.jacobian(lambda unknown: residual(x, unknown))(y)
    return y - jnp.linalg.solve(jacobian, residual(x, y))


class ImplicitSurrogate(eqx.Module):
    """The zero-residual slice of a fitted inverse map, `f(x) = h(x, 0)`.

    Feed-forward, so it is an explicit approximator of the target built from an
    explicit surrogate of the inverse map. `root_find` and `refine` turn it into
    a solve-at-call approximator.
    """

    inverse_map: FittedSurrogate
    residual_dimension: int = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        augmented = jnp.concatenate([x, jnp.zeros(self.residual_dimension)])
        return self.inverse_map(augmented)


class RefinedSurrogate(eqx.Module):
    """A surrogate polished by Newton steps on the residual at call time.

    It approximates the target indirectly: the base surrogate gives the starting
    guess, and the Newton steps drive `g(x, f(x))` toward zero. Both an explicit
    surrogate and a reduced-residual solver use this to correct their output.
    """

    base: Surrogate
    steps: int = eqx.field(static=True)
    residual: Residual = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        prediction = self.base(x)
        for _ in range(self.steps):
            prediction = _newton_step(self.residual, x, prediction)
        return prediction


def refine(surrogate: Surrogate, residual: Residual, *, steps: int) -> RefinedSurrogate:
    """Polish a surrogate's output with Newton steps on the residual at call time."""
    report = getattr(surrogate, "report", _zero_report(1))
    return RefinedSurrogate(base=surrogate, steps=steps, residual=residual, report=report)


def root_find(residual: Residual, initial: Surrogate, *, steps: int) -> RefinedSurrogate:
    """Solve `residual(x, .) = 0` at call, warm-started from `initial`.

    An explicit surrogate of the residual, wrapped this way, becomes an implicit
    surrogate of the target.
    """
    return refine(initial, residual, steps=steps)


class InverseMap(eqx.Module):
    """Fit the target by inverting the residual and reading the zero slice."""

    form: LinearForm
    unknown_domain: Domain
    newton_steps: int = eqx.field(static=True, default=1)


class Objective(eqx.Module):
    """Fit the target directly by minimizing the residual it leaves.

    Experimental: the objective is indifferent between solution branches, so it
    can converge to a biased mixture where the residual has several solutions.
    """

    form: LinearForm
    unknown_dimension: int = eqx.field(static=True)
    steps: int = eqx.field(static=True, default=500)
    learning_rate: float = eqx.field(static=True, default=0.01)


def _fit_inverse_map(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: InverseMap,
    holdout: float,
    ridge: float,
) -> ImplicitSurrogate:
    """Sample unknowns, pull them to the manifold, and fit the inverse map."""
    unknowns = sampling.sample(method.unknown_domain, inputs.shape[0], seed=0)
    for _ in range(method.newton_steps):
        unknowns = jax.vmap(lambda x, y: _newton_step(residual, x, y))(inputs, unknowns)
    residuals = jax.vmap(residual)(inputs, unknowns)
    augmented = jnp.concatenate([inputs, residuals], axis=1)
    inverse_map = fit_data(augmented, unknowns, method.form, holdout=holdout, ridge=ridge)
    return ImplicitSurrogate(
        inverse_map=inverse_map,
        residual_dimension=residuals.shape[1],
        report=_value_report(
            jnp.asarray(0.0), jnp.asarray(0.0), inputs.shape[1], inverse_map.report.structure_cost
        ),
    )


def _fit_objective(
    residual: Residual, domain: Domain, inputs: Float[Array, "n d"], method: Objective
) -> FittedSurrogate:
    """Fit a linear form directly by minimizing the residual it leaves."""
    lower, upper = domain.bounds()
    input_normalizer = normalize.from_bounds(lower, upper)
    normalized = jax.vmap(input_normalizer.forward)(inputs)
    feature_count = method.form.features(normalized[0]).shape[0]

    def loss(weights, _):
        residuals = jax.vmap(
            lambda x, z: residual(x, weights @ method.form.features(z))
        )(inputs, normalized)
        return jnp.mean(residuals**2)

    solver = optx.OptaxMinimiser(optax.adam(method.learning_rate), rtol=1e-8, atol=1e-8)
    start = jnp.zeros((method.unknown_dimension, feature_count))
    weights = optx.minimise(loss, solver, start, max_steps=method.steps, throw=False).value

    identity_output = normalize.Normalizer(
        center=jnp.zeros(method.unknown_dimension), scale=jnp.ones(method.unknown_dimension)
    )
    _, output_unravel = ravel_example(jnp.zeros(method.unknown_dimension))
    return FittedSurrogate(
        input_reduction=reduction.identity(inputs.shape[1]),
        input_normalizer=input_normalizer,
        model=LinearModel(weights=weights, form=method.form),
        output_normalizer=identity_output,
        output_reduction=reduction.identity(method.unknown_dimension),
        output_unravel=output_unravel,
        domain=domain,
        report=_value_report(jnp.asarray(0.0), jnp.asarray(0.0), inputs.shape[1], feature_count),
    )


def fit_implicit(
    residual: Residual,
    domain: Domain,
    method: InverseMap | Objective,
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    refine_steps: int = 0,
    seed: int = 0,
) -> ImplicitSurrogate | FittedSurrogate | RefinedSurrogate:
    """Fit a surrogate `f` with `residual(x, f(x))` near zero, through a method.

    Samples the domain, builds the surrogate through the strategy `method`, and
    reports the residual the surrogate leaves on held-out inputs. A positive
    `refine_steps` polishes each prediction with Newton steps on the residual at
    call time, which both the inverse-map and objective results can use.
    """
    inputs = sampling.sample(domain, samples, seed=seed)
    holdout_count = max(1, round(samples * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]

    match method:
        case InverseMap():
            base: ImplicitSurrogate | FittedSurrogate = _fit_inverse_map(
                residual, train_inputs, method, holdout, ridge
            )
        case Objective():
            base = _fit_objective(residual, domain, train_inputs, method)
        case _:
            raise TypeError(f"unsupported implicit method: {type(method).__name__}")

    surrogate: ImplicitSurrogate | FittedSurrogate | RefinedSurrogate = base
    if refine_steps > 0:
        surrogate = refine(base, residual, steps=refine_steps)

    predicted = jax.vmap(surrogate)(holdout_inputs)
    left = jnp.linalg.norm(jax.vmap(residual)(holdout_inputs, predicted), axis=1)
    report = _value_report(
        jnp.mean(left), jnp.max(left), inputs.shape[1], base.report.structure_cost
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)
