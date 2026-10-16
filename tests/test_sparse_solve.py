"""Sparse direct Newton solves: patterns, steps, state threading, and adjoint derivatives."""

from collections.abc import Callable

import asdex
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import pytest
import splineax
from jax.extend.core import Primitive

import surreax as sx
from surreax import sparse_solve
from surreax.implicit import (
    NewtonSetup,
    _newton_step,
    _newton_step_matrix_free,
    _solve_independently_with,
    newton_setup,
)
from tests import implicit_systems

SYSTEMS: dict[str, Callable[[], implicit_systems.ImplicitSystem]] = {
    "heated_rod": lambda: implicit_systems.heated_rod(32),
    "radiating_rod": lambda: implicit_systems.radiating_rod(32),
    "rod_with_hot_spots": lambda: implicit_systems.rod_with_hot_spots(32),
    "hot_fluid_pipe": lambda: implicit_systems.hot_fluid_pipe(32),
    "buckling_beam": lambda: implicit_systems.buckling_beam(24),
    "resistor_network": lambda: implicit_systems.resistor_network(4, 4),
    "diode_network": lambda: implicit_systems.diode_network(4, 4),
    "latch_ring": lambda: implicit_systems.latch_ring(4),
    "radiating_plate": lambda: implicit_systems.radiating_plate(6),
}


def _points(system: implicit_systems.ImplicitSystem, count: int, seed: int) -> jax.Array:
    """Inputs drawn uniformly from the system's box."""
    unit = jax.random.uniform(jax.random.key(seed), (count, system.domain.dimension))
    return system.domain.sample_unit(unit)


def _unknowns(system: implicit_systems.ImplicitSystem, count: int, seed: int) -> jax.Array:
    """Unknowns drawn uniformly from the system's start box."""
    unit = jax.random.uniform(jax.random.key(seed), (count, system.unknown_domain.dimension))
    return system.unknown_domain.sample_unit(unit)


def _entries(coloring: asdex.ColoredPattern) -> set[tuple[int, int]]:
    rows, cols = np.asarray(coloring.sparsity.rows), np.asarray(coloring.sparsity.cols)
    return set(zip(rows.tolist(), cols.tolist(), strict=True))


def _dense_coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    """A residual whose every entry reads every unknown."""
    return y + 0.1 * jnp.sum(y) - x[0]


_unreadable = Primitive("surreax_unreadable")
"""A primitive asdex has no rule for, so it cannot read a pattern through it."""
_unreadable.def_impl(lambda y: 2.0 * y)
_unreadable.def_abstract_eval(lambda y: y)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_stencil_pattern_covers_the_traced_pattern(name: str) -> None:
    system = SYSTEMS[name]()
    x, y = _points(system, 1, 0)[0], _unknowns(system, 1, 1)[0]
    stated = sx.residual_pattern(system.residual, x, y)
    traced = sx.residual_pattern(lambda a, b: system.residual(a, b), x, y)
    assert stated is not None
    assert traced is not None
    # Padded stencil slots add explicit zeros, so the stated pattern may be wider.
    assert _entries(traced) <= _entries(stated)


@pytest.mark.parametrize("name", ["radiating_rod", "diode_network", "buckling_beam"])
def test_colored_jacobian_matches_the_dense_jacobian(name: str) -> None:
    system = SYSTEMS[name]()
    x, y = _points(system, 1, 0)[0], _unknowns(system, 1, 1)[0]
    pattern = sx.residual_pattern(system.residual, x, y)
    assert pattern is not None
    operator = lineax.JacobianLinearOperator(
        lambda unknown, _: system.residual(x, unknown),
        y,
        tags=splineax.sparsity_coloring_tag(pattern),
    )
    sparse = splineax.materialise_as_bcoo(operator).matrix.todense()
    dense = jax.jacobian(lambda unknown: system.residual(x, unknown))(y)
    assert jnp.allclose(sparse, dense, atol=1e-5)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_sparse_step_matches_the_dense_step(name: str) -> None:
    # Double precision keeps both solves exact on the ill-conditioned systems, and
    # turning it on is the caller's choice that selects KLU on CPU.
    with jax.enable_x64(True):
        system = SYSTEMS[name]()
        x, y = _points(system, 1, 0)[0], _unknowns(system, 1, 1)[0]
        assert y.dtype == jnp.float64
        setup = newton_setup(system.residual, "sparse", None, x, y)
        assert setup.pattern is not None
        sparse = setup.step(system.residual, x, y)
        dense = _newton_step(system.residual, x, y)
        assert float(jnp.linalg.norm(sparse - dense) / jnp.linalg.norm(dense)) < 1e-8


def test_sparse_step_solves_in_single_precision() -> None:
    # Without x64 the direct solve runs in single precision through `Spsolve`.
    system = implicit_systems.heated_rod(32)
    x, y = _points(system, 1, 0)[0], _unknowns(system, 1, 1)[0]
    assert y.dtype == jnp.float32
    setup = newton_setup(system.residual, "sparse", None, x, y)
    solved = setup.step(system.residual, x, y)
    assert float(jnp.linalg.norm(system.residual(x, solved))) < 1e-3


def test_unusable_patterns_fall_back_to_the_matrix_free_step() -> None:
    x, y = jnp.array([0.3]), jnp.zeros(64)
    assert sx.residual_pattern(_dense_coupling, x, y) is None
    assert sx.residual_pattern(lambda a, b: _unreadable.bind(b) - a[0], x, y) is None
    setup = newton_setup(_dense_coupling, "sparse", None, x, y)
    assert setup.step is _newton_step_matrix_free
    assert setup.pattern is None
    assert newton_setup(_dense_coupling, "dense", None, x, y).step is _newton_step
    assert newton_setup(_dense_coupling, "matrix_free", None, x, y).step is (
        _newton_step_matrix_free
    )


def test_threaded_solve_matches_the_unthreaded_solve() -> None:
    system = implicit_systems.radiating_rod(48)
    inputs, starts = _points(system, 6, 0), _unknowns(system, 6, 1)
    setup = newton_setup(system.residual, "sparse", None, inputs[0], starts[0])
    threaded = _solve_independently_with(setup, system.residual, inputs, starts, 12)
    unthreaded = _solve_independently_with(
        NewtonSetup(step=setup.step, pattern=None), system.residual, inputs, starts, 12
    )
    assert jnp.allclose(threaded, unthreaded, atol=1e-4)
    left = jnp.linalg.norm(jax.vmap(system.residual)(inputs, threaded), axis=1)
    start = jnp.linalg.norm(jax.vmap(system.residual)(inputs, starts), axis=1)
    assert float(jnp.max(left / start)) < 1e-4


def test_hybrid_reuse_reaches_the_same_solution() -> None:
    system = implicit_systems.heated_rod(48)
    inputs, starts = _points(system, 4, 0), _unknowns(system, 4, 1)
    reuse = splineax.HybridSettings(iterative=splineax.RichardsonOptions(), tol=1e-4)
    setup = newton_setup(system.residual, "sparse", reuse, inputs[0], starts[0])
    hybrid = _solve_independently_with(setup, system.residual, inputs, starts, 3)
    dense = _solve_independently_with(
        NewtonSetup(step=_newton_step, pattern=None), system.residual, inputs, starts, 3
    )
    assert jnp.allclose(hybrid, dense, atol=1e-3)


@pytest.mark.parametrize("linear_solver", ["sparse", "dense", "matrix_free"])
def test_reduced_state_jacobian_matches_the_implicit_function_theorem(
    linear_solver: sx.LinearSolver,
) -> None:
    system = implicit_systems.radiating_rod(32)
    inputs, starts = _points(system, 2, 0), _unknowns(system, 2, 1)
    setup = newton_setup(system.residual, "dense", None, inputs[0], starts[0])
    unknowns = _solve_independently_with(setup, system.residual, inputs, starts, 10)
    basis = jnp.linalg.qr(jax.random.normal(jax.random.key(2), (32, 3)))[0].T
    pattern = (
        sx.residual_pattern(system.residual, inputs[0], unknowns[0])
        if linear_solver == "sparse"
        else None
    )
    reduced = sparse_solve.reduced_state_jacobians(
        system.residual, inputs, unknowns, basis, linear_solver, pattern
    )

    def expected(x: jax.Array, y: jax.Array) -> jax.Array:
        in_unknown = jax.jacobian(lambda u: system.residual(x, u))(y)
        in_input = jax.jacobian(lambda point: system.residual(point, y))(x)
        return basis @ -jnp.linalg.solve(in_unknown, in_input)

    reference = jax.vmap(expected)(inputs, unknowns)
    assert reduced.shape == (2, 3, system.domain.dimension)
    assert float(jnp.linalg.norm(reduced - reference) / jnp.linalg.norm(reference)) < 1e-3


def test_exact_solve_derivative_supervision_lowers_the_jacobian_error() -> None:
    system = implicit_systems.radiating_rod(32)

    def fitted(weight: float) -> sx.FittedSurrogate:
        method = sx.ExactSolve(
            form=sx.ChebyshevPolynomial(degree=2),
            unknown_domain=system.unknown_domain,
            rank=4,
            newton_steps=10,
            derivative_weight=weight,
        )
        surrogate = sx.fit_implicit(system.residual, system.domain, method, samples=48)
        assert isinstance(surrogate, sx.FittedSurrogate)
        return surrogate

    points = _points(system, 8, 5)
    starts = jnp.zeros((8, system.unknown_domain.dimension))
    setup = newton_setup(system.residual, "dense", None, points[0], starts[0])
    solved = _solve_independently_with(setup, system.residual, points, starts, 12)

    def true_jacobian(x: jax.Array, y: jax.Array) -> jax.Array:
        in_unknown = jax.jacobian(lambda u: system.residual(x, u))(y)
        in_input = jax.jacobian(lambda point: system.residual(point, y))(x)
        return -jnp.linalg.solve(in_unknown, in_input)

    reference = jax.vmap(true_jacobian)(points, solved)

    def jacobian_error(surrogate: sx.FittedSurrogate) -> float:
        predicted = jax.vmap(jax.jacobian(surrogate))(points)
        return float(jnp.linalg.norm(predicted - reference) / jnp.linalg.norm(reference))

    assert jacobian_error(fitted(1.0)) < jacobian_error(fitted(0.0))


def test_linear_solver_recommendation_follows_the_structure() -> None:
    latch = implicit_systems.latch_ring(4)
    small = sx.recommend_linear_solver(latch.residual, latch.domain, latch.unknown_domain)
    assert small == "dense"
    rod = implicit_systems.heated_rod(64)
    assert sx.recommend_linear_solver(rod.residual, rod.domain, rod.unknown_domain) == "sparse"
    domain, unknown_domain = sx.box([-1.0], [1.0]), sx.box([-1.0] * 64, [1.0] * 64)
    assert sx.recommend_linear_solver(_dense_coupling, domain, unknown_domain) == "matrix_free"
    method = sx.recommend_method(rod.residual, rod.domain, rod.unknown_domain)
    assert isinstance(method, sx.ExactSolve)
    assert method.linear_solver == "sparse"


def test_threaded_solve_reuses_the_analysis_across_newton_steps() -> None:
    with jax.enable_x64(True):
        system = implicit_systems.radiating_rod(32)
        inputs, starts = _points(system, 2, 0), _unknowns(system, 2, 1)
        setup = newton_setup(system.residual, "sparse", None, inputs[0], starts[0])
        profile = splineax.create_solve_profile()
        with profile:
            jax.block_until_ready(
                _solve_independently_with(setup, system.residual, inputs, starts, 4)
            )
    operations = [(record.solver, record.operation) for record in profile.records]
    # Each sample analyzes once, then refactors with its pivots for later steps.
    assert operations.count(("KLU", "analyze")) == 2
    assert operations.count(("KLU", "refactor")) == 6
