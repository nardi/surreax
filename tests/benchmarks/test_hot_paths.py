"""Benchmarks for the hot paths.

Each benchmark warms the JIT once outside the timed region, so the measurement
is of the compiled path. These track performance over time and do not gate CI.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import splineax

import surreax as sx
from surreax.error_search import worst_case
from surreax.implicit import _solve_independently_with, newton_setup
from tests import implicit_systems, systems


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
            systems.large_sparse_residual, domain, method, sx.Affine(), samples=64
        )
        return jax.block_until_ready(surrogate(jnp.zeros(systems.LARGE_INPUTS)))

    build()
    benchmark(build)


def test_benchmark_large_field_call(benchmark) -> None:
    surrogate = sx.fit_implicit(
        systems.large_sparse_residual,
        systems.large_sparse_domain(),
        _reduced_solve_method(),
        sx.Affine(),
        samples=64,
    )
    call = eqx.filter_jit(surrogate)
    point = jnp.array([0.4, -0.2, 0.6, 0.0])
    jax.block_until_ready(call(point))
    benchmark(lambda: jax.block_until_ready(call(point)))


def _hot_spot_solvers(points: int) -> tuple[sx.ReducedSolver, sx.HyperReducedSolver]:
    """The linear reduced solver of the rod with hot spots, and its hyper-reduction."""
    rod = implicit_systems.rod_with_hot_spots(points)
    method = sx.ReducedSolve(
        unknown_domain=rod.unknown_domain,
        rank=6,
        snapshots=32,
        newton_steps=8,
        reduced_steps=8,
    )
    solver = sx.reduced_solver(rod.residual, rod.domain, method, samples=64)
    return solver, sx.hyper_reduce(solver, rod.residual, rod.domain)


@pytest.mark.parametrize("points", [1024, 16384, 131072])
@pytest.mark.parametrize("linear_solver", ["matrix_free", "sparse"])
def test_benchmark_snapshot_solve_against_state_size(
    benchmark, points: int, linear_solver: sx.LinearSolver
) -> None:
    # Both Newton steps should grow about linearly with the state. The sparse
    # step runs in double precision, which selects KLU on CPU.
    with jax.enable_x64(True):
        rod = implicit_systems.heated_rod(points)
        x = jnp.full(4, 0.5)
        field = jnp.zeros(points)
        setup = newton_setup(rod.residual, linear_solver, None, x, field)
        step = jax.jit(lambda x, y: setup.step(rod.residual, x, y))
        jax.block_until_ready(step(x, field))
        benchmark(lambda: jax.block_until_ready(step(x, field)))


_NEWTON_SYSTEMS = {
    "rod": lambda: implicit_systems.radiating_rod(4096),
    "network": lambda: implicit_systems.diode_network(64, 64),
    "plate": lambda: implicit_systems.radiating_plate(64),
    "block": lambda: implicit_systems.radiating_block(16),
}
"""Four nonlinear systems of 4,096 unknowns: a chain, a general graph, a 2D grid and a 3D
grid."""


_HYBRID = splineax.HybridSettings(iterative=splineax.GMRESOptions(), tol=1e-6)
"""GMRES preconditioned by a stale factorization, at a tolerance the conditioning allows."""


@pytest.mark.parametrize(
    ("linear_solver", "reuse"),
    [("matrix_free", None), ("sparse", None), ("sparse", _HYBRID)],
    ids=["matrix_free", "sparse", "hybrid"],
)
@pytest.mark.parametrize("system", sorted(_NEWTON_SYSTEMS))
def test_benchmark_newton_solve_against_linear_solver(
    benchmark,
    system: str,
    linear_solver: sx.LinearSolver,
    reuse: splineax.HybridSettings | None,
) -> None:
    # Six Newton steps for eight inputs. The hybrid only pays off where the
    # factorization fills in heavily, as on the 3D block.
    with jax.enable_x64(True):
        built = _NEWTON_SYSTEMS[system]()
        unit = jax.random.uniform(jax.random.key(0), (8, built.domain.dimension))
        inputs = built.domain.sample_unit(unit)
        starts = jnp.zeros((8, built.unknown_domain.dimension))
        setup = newton_setup(built.residual, linear_solver, reuse, inputs[0], starts[0])
        solve = jax.jit(lambda a, b: _solve_independently_with(setup, built.residual, a, b, 6))
        jax.block_until_ready(solve(inputs, starts))
        benchmark(lambda: jax.block_until_ready(solve(inputs, starts)))


@pytest.mark.parametrize("points", [512, 8192])
@pytest.mark.parametrize("variant", ["full", "hyper"])
def test_benchmark_latent_solve_against_state_size(benchmark, points: int, variant: str) -> None:
    # The full reduced solve reads the whole residual, the hyper-reduced one a
    # fixed number of entries, so only the first grows with the state.
    full, hyper = _hot_spot_solvers(points)
    solver = full if variant == "full" else hyper
    call = eqx.filter_jit(lambda s, x: s.latent(x))
    x = jnp.array([0.3, 0.6, 0.2, 0.8])
    jax.block_until_ready(call(solver, x))
    benchmark(lambda: jax.block_until_ready(call(solver, x)))


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_benchmark_reduced_solve_against_latent_dimension(benchmark, rank: int) -> None:
    # Each reduced Newton step is an r-by-r solve, so the cost follows the rank.
    rod = implicit_systems.radiating_rod(4096)
    method = sx.ReducedSolve(
        unknown_domain=rod.unknown_domain,
        rank=rank,
        snapshots=32,
        newton_steps=8,
        reduced_steps=8,
    )
    solver = sx.reduced_solver(rod.residual, rod.domain, method, samples=64)
    call = eqx.filter_jit(lambda s, x: s.latent(x))
    x = jnp.full(4, 0.5)
    jax.block_until_ready(call(solver, x))
    benchmark(lambda: jax.block_until_ready(call(solver, x)))
