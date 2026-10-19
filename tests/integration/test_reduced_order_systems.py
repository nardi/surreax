"""Reduced-order methods on the physical systems: trust reports and linear exactness."""

from typing import Any

import jax
import jax.numpy as jnp

import surreax as sx
from tests import implicit_systems


def _linear_solver(
    system: implicit_systems.ImplicitSystem, rank: int, **settings: object
) -> sx.ReducedSolver:
    # Each value is checked by the constructor that receives the keyword arguments,
    # so the values are not narrowed here.
    options: dict[str, Any] = {
        "snapshots": 32,
        "newton_steps": 12,
        "reduced_steps": 10,
        "linear_solver": "sparse",
    }
    options.update(settings)
    method = sx.ReducedSolve(
        unknown_domain=system.unknown_domain,
        rank=rank,
        **options,
    )
    return sx.reduced_solver(system.residual, system.domain, method, samples=64)


def _relative(system: implicit_systems.ImplicitSystem, solver: sx.ReducedSolver) -> float:
    """Mean residual a solver leaves, over the mean residual at zero."""
    unit = jax.random.uniform(jax.random.key(3), (32, system.domain.dimension))
    points = system.domain.sample_unit(unit)
    unknown = system.unknown_domain.dimension
    left = jax.vmap(lambda x: jnp.linalg.norm(system.residual(x, solver(x))))(points)
    base = jax.vmap(lambda x: jnp.linalg.norm(system.residual(x, jnp.zeros(unknown))))(points)
    return float(jnp.mean(left) / jnp.mean(base))


def test_beam_report_flags_two_shapes_near_the_critical_load() -> None:
    # Newton converges only linearly near the fold, so the snapshot check is off.
    beam = implicit_systems.buckling_beam()
    solver = _linear_solver(beam, rank=2, convergence_tolerance=None)
    assert solver.report.multiple_roots
    # The worst conditioning sits near the critical load, where the beam folds.
    assert float(solver.report.min_conditioning) < 0.05
    assert abs(float(solver.report.boundary_point[0]) - 1.0) < 0.3


def test_latch_ring_report_flags_two_stored_states() -> None:
    solver = _linear_solver(implicit_systems.latch_ring(4), rank=3)
    assert solver.report.multiple_roots


def test_resistor_network_is_exact_on_a_general_graph() -> None:
    network = implicit_systems.resistor_network(8, 8, sources=4)
    solver = _linear_solver(network, rank=4)
    assert not solver.report.multiple_roots
    assert _relative(network, solver) < 1e-5
    # Four sources span the voltages, so a greedy choice of six snapshots suffices.
    greedy = _linear_solver(network, rank=4, snapshots=6, greedy=True)
    assert _relative(network, greedy) < 1e-5


def test_heated_rod_residual_drops_at_the_number_of_heaters() -> None:
    rod = implicit_systems.heated_rod(256, heaters=4)

    def at_rank(rank: int) -> float:
        solver = _linear_solver(rod, rank, newton_steps=3, reduced_steps=3)
        return _relative(rod, solver)

    assert at_rank(3) > 0.1
    assert at_rank(4) < 1e-5
