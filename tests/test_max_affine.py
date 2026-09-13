"""Tests for the max-affine form and its fit."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.forms.max_affine import MaxAffine, fit_max_affine


def _convex_target(x: jax.Array) -> jax.Array:
    piece_a = 2 * x[0] - x[1] + 1.0
    piece_b = -x[0] + 3 * x[1] - 0.5
    piece_c = x[0] + x[1]
    return jnp.maximum(jnp.maximum(piece_a, piece_b), piece_c)[None]


def test_max_affine_model_takes_the_maximum_piece() -> None:
    model = MaxAffine(
        slopes=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
        intercepts=jnp.array([[0.0, 0.5]]),
    )
    value = model(jnp.array([0.2, 0.9]))
    assert jnp.allclose(value, jnp.array([max(0.2, 0.9 + 0.5)]))


def test_alternating_fit_recovers_a_max_of_affine_functions() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.MaxAffineForm(pieces=3, restarts=6, iterations=25)
    surrogate = sx.fit(_convex_target, domain, form, samples=400, seed=0)
    assert float(surrogate.report.value_max) < 1e-3
    assert surrogate.report.structure_cost == 3


def test_gradient_fit_also_recovers_the_target() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.MaxAffineForm(pieces=3, restarts=3, method="gradient", steps=300)
    surrogate = sx.fit(_convex_target, domain, form, samples=400, seed=0)
    assert float(surrogate.report.value_mean) < 1e-2


def test_fitted_max_affine_is_convex() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.MaxAffineForm(pieces=4, restarts=4, iterations=20)
    surrogate = sx.fit(_convex_target, domain, form, samples=400, seed=0)

    left = jax.random.uniform(jax.random.key(1), (200, 2), minval=-1.0, maxval=1.0)
    right = jax.random.uniform(jax.random.key(2), (200, 2), minval=-1.0, maxval=1.0)
    midpoint = (left + right) / 2
    value_left = jax.vmap(surrogate)(left)[:, 0]
    value_right = jax.vmap(surrogate)(right)[:, 0]
    value_mid = jax.vmap(surrogate)(midpoint)[:, 0]
    # A convex function sits below the chord between any two points.
    assert jnp.all(value_mid <= 0.5 * (value_left + value_right) + 1e-4)


def test_max_affine_fit_is_deterministic_for_a_key() -> None:
    inputs = jax.random.uniform(jax.random.key(0), (128, 2), minval=-1.0, maxval=1.0)
    targets = jax.vmap(_convex_target)(inputs)
    form = sx.MaxAffineForm(pieces=3, restarts=4, iterations=15)
    first = fit_max_affine(form, inputs, targets, key=jax.random.key(7))
    second = fit_max_affine(form, inputs, targets, key=jax.random.key(7))
    assert jnp.allclose(first.slopes, second.slopes)
