"""Tests for adaptive, error-driven sampling."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax.error_search import high_error_points


def _bump(x: jax.Array) -> jax.Array:
    return (jnp.exp(-8 * (x[0] ** 2 + x[1] ** 2)))[None]


@pytest.mark.slow
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


def _convex(x: jax.Array) -> jax.Array:
    return jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])[None]


def test_warm_start_never_worsens_a_max_affine_refit() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.MaxAffineForm(pieces=3, restarts=4, iterations=15)
    previous = sx.fit(_convex, domain, form, samples=256, worst_case_starts=0, seed=0)
    cold = sx.fit(_convex, domain, form, samples=256, worst_case_starts=0, seed=0)
    # Same data and seed, so the warm start adds the previous fit as a candidate
    # and keeps the best, which cannot be worse than the cold fit.
    warm = sx.fit(
        _convex, domain, form, samples=256, worst_case_starts=0, warm_start=previous, seed=0
    )
    assert float(warm.report.value_mean) <= float(cold.report.value_mean) + 1e-6


def test_warm_start_reuses_a_network_across_a_refit() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.NetworkForm(width=16, depth=2, steps=300)
    previous = sx.fit(_bump, domain, form, samples=256, worst_case_starts=0, seed=0)
    warm = sx.fit(
        _bump, domain, form, samples=256, worst_case_starts=0, warm_start=previous, seed=0
    )
    assert isinstance(warm.model, sx.Network)
    assert jnp.isfinite(warm.report.value_mean)


@pytest.mark.slow
def test_adaptive_reuses_the_previous_fit_for_an_iterative_form() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.DifferenceOfMaxAffineForm(pieces=2, restarts=3, steps=200)
    # A tolerance it will not meet, so the loop refits and passes the warm start.
    surrogate = sx.fit_adaptive(
        _convex,
        domain,
        form,
        initial_samples=128,
        points_per_round=16,
        rounds=2,
        tolerance=0.0,
        worst_case_steps=80,
        seed=0,
    )
    assert jnp.isfinite(surrogate.report.worst_case)


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
