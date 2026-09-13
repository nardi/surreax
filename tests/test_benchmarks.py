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
