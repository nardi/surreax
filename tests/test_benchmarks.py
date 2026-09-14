"""Benchmarks for the hot paths.

Each benchmark warms the JIT once outside the timed region, so the measurement
is of the compiled path. These track performance over time and do not gate CI.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax.error_search import worst_case
from tests import systems


@pytest.fixture(scope="module")
def surrogate() -> sx.FittedSurrogate:
    return sx.fit(
        systems.projectile_range,
        systems.projectile_domain(),
        sx.ChebyshevPolynomial(degree=4),
        samples=256,
        seed=0,
    )


def test_benchmark_fit(benchmark) -> None:
    domain = systems.projectile_domain()
    form = sx.ChebyshevPolynomial(degree=4)
    benchmark(lambda: sx.fit(systems.projectile_range, domain, form, samples=256, seed=0))


def test_benchmark_single_call(benchmark, surrogate: sx.FittedSurrogate) -> None:
    call = eqx.filter_jit(surrogate)
    point = jnp.array([12.0, 0.7])
    jax.block_until_ready(call(point))
    benchmark(lambda: jax.block_until_ready(call(point)))


def test_benchmark_vmap_call(benchmark, surrogate: sx.FittedSurrogate) -> None:
    call = eqx.filter_jit(jax.vmap(surrogate))
    batch = jax.random.uniform(jax.random.key(0), (1024, 2))
    jax.block_until_ready(call(batch))
    benchmark(lambda: jax.block_until_ready(call(batch)))


def test_benchmark_jacobian_call(benchmark, surrogate: sx.FittedSurrogate) -> None:
    call = eqx.filter_jit(jax.jacobian(surrogate))
    point = jnp.array([12.0, 0.7])
    jax.block_until_ready(call(point))
    benchmark(lambda: jax.block_until_ready(call(point)))


def test_benchmark_worst_case_search(benchmark, surrogate: sx.FittedSurrogate) -> None:
    domain = systems.projectile_domain()
    lower, upper = domain.bounds()
    seeds = lower + (upper - lower) * jax.random.uniform(jax.random.key(0), (32, 2))

    def search() -> jax.Array:
        value, _ = worst_case(
            surrogate,
            systems.projectile_range,
            domain,
            starts=8,
            steps=100,
            learning_rate=0.05,
            key=jax.random.key(0),
            seed_points=seeds,
        )
        return jax.block_until_ready(value)

    search()
    benchmark(search)


def _reduced_solve_method() -> sx.ReducedSolve:
    return sx.ReducedSolve(
        form=sx.Affine(),
        unknown_domain=systems.large_sparse_unknown_domain(),
        rank=systems.LARGE_INPUTS,
        snapshots=16,
        newton_steps=4,
        reduced_steps=4,
    )


def test_benchmark_large_reduced_solve_fit(benchmark) -> None:
    # A small input, a large field, and a sparse Jacobian, fitted through a few
    # snapshot solves and a reduced Newton, so the cost is set by the snapshots.
    domain = systems.large_sparse_domain()
    method = _reduced_solve_method()

    def build() -> sx.FittedSurrogate:
        surrogate = sx.fit_implicit(
            systems.large_sparse_residual, domain, method, samples=64
        )
        return jax.block_until_ready(surrogate(jnp.zeros(systems.LARGE_INPUTS)))

    build()
    benchmark(build)


def test_benchmark_large_field_call(benchmark) -> None:
    surrogate = sx.fit_implicit(
        systems.large_sparse_residual,
        systems.large_sparse_domain(),
        _reduced_solve_method(),
        samples=64,
    )
    call = eqx.filter_jit(surrogate)
    point = jnp.array([0.4, -0.2, 0.6, 0.0])
    jax.block_until_ready(call(point))
    benchmark(lambda: jax.block_until_ready(call(point)))
