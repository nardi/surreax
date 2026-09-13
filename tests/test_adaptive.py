"""Tests for adaptive, error-driven sampling."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.error_search import high_error_points


def _bump(x: jax.Array) -> jax.Array:
    return (jnp.exp(-8 * (x[0] ** 2 + x[1] ** 2)))[None]


def test_adaptive_beats_a_fixed_sample_of_the_same_size() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.ChebyshevPolynomial(degree=6)
    adaptive = sx.fit_adaptive(
        _bump,
        domain,
        form,
        initial_samples=128,
        points_per_round=32,
        rounds=4,
        tolerance=1e-4,
        worst_case_steps=120,
        seed=0,
    )
    # A fixed sample of the same total budget: 128 initial plus 4 rounds of 32.
    fixed = sx.fit(_bump, domain, form, samples=256, worst_case_steps=120, seed=0)
    assert float(adaptive.report.worst_case) < float(fixed.report.worst_case)


def test_adaptive_stops_early_on_an_easy_target() -> None:
    def affine(x: jax.Array) -> jax.Array:
        return (2 * x[0] - x[1] + 1.0)[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    # The affine form is exact, so the first fit already meets the tolerance.
    surrogate = sx.fit_adaptive(
        affine, domain, sx.Affine(), initial_samples=64, rounds=5, tolerance=1e-3, seed=0
    )
    assert float(surrogate.report.worst_case) < 1e-3


def test_high_error_points_returns_the_requested_count_inside_the_domain() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(_bump, domain, sx.ChebyshevPolynomial(degree=4), samples=128, seed=0)
    seeds = jax.random.uniform(jax.random.key(1), (16, 2), minval=-1.0, maxval=1.0)
    points = high_error_points(
        surrogate,
        _bump,
        domain,
        count=10,
        steps=80,
        learning_rate=0.05,
        key=jax.random.key(0),
        seed_points=seeds,
    )
    assert points.shape == (10, 2)
    lower, upper = domain.bounds()
    assert jnp.all(points >= lower)
    assert jnp.all(points <= upper)
