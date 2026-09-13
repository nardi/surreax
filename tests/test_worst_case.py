"""Tests for the worst-case error search over the domain."""

import jax
import jax.numpy as jnp

import surreax as sx


def test_worst_case_is_near_zero_for_an_exact_fit() -> None:
    coefficients = jnp.array([2.0, -1.0])

    def affine(x: jax.Array) -> jax.Array:
        return (coefficients @ x + 3.0)[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(affine, domain, sx.Affine(), samples=128, seed=0)
    assert float(surrogate.report.worst_case) < 1e-4


def test_worst_case_is_at_least_the_held_out_maximum() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (jnp.sin(3 * x[0]) * jnp.cos(3 * x[1]))[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=2), samples=128, seed=0)
    # A search over the domain cannot do worse than the held-out sample it contains.
    assert float(surrogate.report.worst_case) >= float(surrogate.report.value_max) - 1e-5


def test_worst_case_search_can_be_disabled() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (x[0] + x[1])[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.Affine(), samples=64, worst_case_starts=0, seed=0)
    assert float(surrogate.report.worst_case) == 0.0


def test_worst_case_point_lies_in_the_domain() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (jnp.sin(3 * x[0]) * jnp.cos(3 * x[1]))[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=2), samples=128, seed=0)
    lower, upper = domain.bounds()
    point = surrogate.report.worst_case_point
    assert jnp.all(point >= lower)
    assert jnp.all(point <= upper)
