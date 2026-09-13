"""Tests for the difference-of-max-affine form."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.forms.difference_max_affine import DifferenceOfMaxAffine


def _nonconvex(x: jax.Array) -> jax.Array:
    positive = jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])
    negative = jnp.maximum(-x[0], x[1])
    return (positive - negative)[None]


def test_model_is_the_difference_of_two_maxima() -> None:
    model = DifferenceOfMaxAffine(
        positive_slopes=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
        positive_intercepts=jnp.array([[0.0, 0.0]]),
        negative_slopes=jnp.array([[[1.0, 1.0]]]),
        negative_intercepts=jnp.array([[0.0]]),
    )
    value = model(jnp.array([0.3, 0.7]))
    assert jnp.allclose(value, jnp.array([max(0.3, 0.7) - (0.3 + 0.7)]))


def test_difference_recovers_a_nonconvex_piecewise_linear_target() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.DifferenceOfMaxAffineForm(pieces=2, restarts=6, steps=400)
    surrogate = sx.fit(_nonconvex, domain, form, samples=400, worst_case_starts=0, seed=0)
    assert float(surrogate.report.value_max) < 1e-2
    assert surrogate.report.structure_cost == 4


def test_difference_beats_a_single_max_affine_on_a_nonconvex_target() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    difference = sx.fit(
        _nonconvex,
        domain,
        sx.DifferenceOfMaxAffineForm(pieces=2, restarts=6, steps=400),
        samples=400,
        worst_case_starts=0,
        seed=0,
    )
    # A single convex max-affine of the same total pieces cannot fit a nonconvex target.
    convex = sx.fit(
        _nonconvex,
        domain,
        sx.MaxAffineForm(pieces=4, restarts=6),
        samples=400,
        worst_case_starts=0,
        seed=0,
    )
    assert float(difference.report.value_mean) < float(convex.report.value_mean)


def test_difference_surrogate_survives_jit_and_grad() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.DifferenceOfMaxAffineForm(pieces=2, restarts=4, steps=300)
    surrogate = sx.fit(_nonconvex, domain, form, samples=256, worst_case_starts=0, seed=0)
    point = jnp.array([0.3, -0.2])
    plain = surrogate(point)
    assert jnp.allclose(jax.jit(lambda m, x: m(x))(surrogate, point), plain)
    gradient = jax.grad(lambda x: surrogate(x)[0])(point)
    assert gradient.shape == (2,)
