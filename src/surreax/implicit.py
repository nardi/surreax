"""Implicit fitting through the inverse of the residual.

Given a residual `g(x, y)`, the target is `f` with `g(x, f(x))` near zero. Rather
than solving `g = 0` per sample, this inverts the residual in its second
argument. Writing `r = g(x, y)`, the map `(x, y) -> (x, r)` is inverted to
`y = h(x, r)`, and then `f(x) = h(x, 0)`.

The inverse is cheap to sample: draw `(x, y)`, evaluate `r = g(x, y)` forward,
and the pair `((x, r), y)` is a labeled sample of `h`. So `h` is fitted as an
ordinary explicit regression over the augmented input `(x, r)`, and `f` reads it
at the zero-residual slice. No root solve is involved.

When `g` is affine in `y`, `h` is linear in `r`, so a linear form recovers the
zero-residual slice exactly regardless of how the residual is sampled.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from surreax import normalize, reduction, sampling
from surreax.domain import Domain, box_from_data
from surreax.fitted import FittedSurrogate
from surreax.fitting import fit_linear
from surreax.flatten import ravel_example
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport

Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""A residual `g(x, y)`, from an input and an unknown to the residual vector."""


def _fit_from_data(
    inputs: Float[Array, "s d"],
    outputs: Float[Array, "s k"],
    form: LinearForm,
    *,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Fit a linear form to given input-output pairs, with no target callable.

    Normalizes from sample moments, since the augmented input has no box, and
    reports the held-out prediction error.
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


class ImplicitSurrogate(eqx.Module):
    """A surrogate of an implicit target, the zero-residual slice of the inverse.

    Applies the fitted inverse map `h` at zero residual, so `f(x) = h(x, 0)`.
    """

    inverse_map: FittedSurrogate
    residual_dimension: int = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        augmented = jnp.concatenate([x, jnp.zeros(self.residual_dimension)])
        return self.inverse_map(augmented)


def _newton_step(
    residual: Residual, x: Float[Array, " n"], y: Float[Array, " m"]
) -> Float[Array, " m"]:
    """One Newton step in the unknown, `y - (dg/dy)^{-1} g(x, y)`.

    This is one linear solve, not an iterative solve, and it lands near the
    solution manifold, so the sampled residuals cluster near zero where the
    zero-residual slice is read.
    """
    jacobian = jax.jacobian(lambda unknown: residual(x, unknown))(y)
    return y - jnp.linalg.solve(jacobian, residual(x, y))


def fit_implicit(
    residual: Residual,
    domain: Domain,
    unknown_domain: Domain,
    form: LinearForm,
    *,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    newton_steps: int = 1,
    seed: int = 0,
) -> ImplicitSurrogate:
    """Fit a surrogate `f` with `residual(x, f(x))` near zero, without solving.

    Samples inputs over `domain` and unknowns over `unknown_domain`, takes
    `newton_steps` Newton steps to pull the unknowns near the solution manifold,
    evaluates the residual forward, fits the inverse map over the augmented
    input, and returns its zero-residual slice. The report is the residual the
    surrogate leaves on held-out inputs.
    """
    inputs = sampling.sample(domain, samples, seed=seed)
    unknowns = sampling.sample(unknown_domain, samples, seed=seed + 1)
    for _ in range(newton_steps):
        unknowns = jax.vmap(lambda x, y: _newton_step(residual, x, y))(inputs, unknowns)
    residuals = jax.vmap(residual)(inputs, unknowns)
    augmented = jnp.concatenate([inputs, residuals], axis=1)

    inverse_map = _fit_from_data(augmented, unknowns, form, holdout=holdout, ridge=ridge)
    surrogate = ImplicitSurrogate(
        inverse_map=inverse_map,
        residual_dimension=residuals.shape[1],
        report=_zero_report(inputs.shape[1]),
    )

    holdout_count = max(1, round(samples * holdout))
    holdout_inputs = inputs[-holdout_count:]
    predicted = jax.vmap(surrogate)(holdout_inputs)
    left = jnp.linalg.norm(jax.vmap(residual)(holdout_inputs, predicted), axis=1)
    report = _value_report(
        jnp.mean(left), jnp.max(left), inputs.shape[1], inverse_map.report.structure_cost
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)
