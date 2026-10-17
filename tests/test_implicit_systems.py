"""Checks that the physical test systems have the properties the tests rely on."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surreax.implicit import _solve_independently
from tests import implicit_systems as systems


def _solutions(system: systems.ImplicitSystem, count: int = 64) -> tuple[jax.Array, jax.Array]:
    """Solve the system from zero at inputs drawn across its domain."""
    unit = jax.random.uniform(jax.random.key(0), (count, system.domain.dimension))
    inputs = system.domain.sample_unit(unit)
    starts = jnp.zeros((count, system.unknown_domain.dimension))
    return inputs, _solve_independently(system.residual, inputs, starts, 12)


def _energy(states: jax.Array, rank: int) -> float:
    """Fraction of the state variation the leading `rank` directions capture."""
    singular = jnp.linalg.svd(states - states.mean(axis=0), compute_uv=False)
    return float(jnp.sum(singular[:rank] ** 2) / jnp.sum(singular**2))


_SOLVABLE: dict[str, Callable[[], systems.ImplicitSystem]] = {
    "heated_rod": lambda: systems.heated_rod(64),
    "radiating_rod": lambda: systems.radiating_rod(64),
    "rod_with_hot_spots": lambda: systems.rod_with_hot_spots(64),
    "hot_fluid_pipe": lambda: systems.hot_fluid_pipe(64),
    "resistor_network": lambda: systems.resistor_network(8, 8),
    "diode_network": lambda: systems.diode_network(8, 8),
    "radiating_plate": lambda: systems.radiating_plate(8),
    "radiating_block": lambda: systems.radiating_block(5),
}


@pytest.mark.parametrize("name", sorted(_SOLVABLE))
def test_system_solves_and_stays_within_its_stencil(name: str) -> None:
    system = _SOLVABLE[name]()
    inputs, states = _solutions(system)
    left = jax.vmap(lambda x, y: jnp.linalg.norm(system.residual(x, y)))(inputs, states)
    assert float(jnp.max(left)) < 1e-4

    # The Jacobian reads the unknown only where the stencil says it does.
    jacobian = np.asarray(jax.jacobian(system.residual, argnums=1)(inputs[0], states[0]))
    stencil = np.asarray(system.residual.stencil)
    allowed = np.zeros_like(jacobian, dtype=bool)
    np.put_along_axis(allowed, stencil, True, axis=1)
    assert not np.any((jacobian != 0) & ~allowed)


def test_stencil_entries_match_the_full_residual() -> None:
    system = systems.diode_network(6, 6)
    inputs, states = _solutions(system, count=2)
    rows = jnp.array([0, 7, 20])
    neighbors = states[0][system.residual.stencil[rows]]
    partial = system.residual.entries(inputs[0], rows, neighbors)
    assert jnp.allclose(partial, system.residual(inputs[0], states[0])[rows])


def test_linear_systems_have_exact_low_rank_manifolds() -> None:
    _, rod = _solutions(systems.heated_rod(64, heaters=4))
    _, mesh = _solutions(systems.resistor_network(8, 8, sources=4))
    assert _energy(rod, 4) > 1 - 1e-6
    assert _energy(mesh, 4) > 1 - 1e-6


def test_curved_and_moving_manifolds_need_more_modes() -> None:
    _, radiating = _solutions(systems.radiating_rod(64, heaters=4))
    _, pipe = _solutions(systems.hot_fluid_pipe(64))
    # Radiation bends the four-heater manifold out of a four-dimensional subspace.
    assert _energy(radiating, 4) < 0.999
    # The moving front has two knobs but spreads over many linear directions.
    assert _energy(pipe, 4) < 0.95


def test_hot_spot_nonlinearity_stays_on_the_patches() -> None:
    def support(points: int) -> int:
        hot = systems.rod_with_hot_spots(points)
        plain = systems.heated_rod(points)
        state = jnp.linspace(0.5, 1.5, points)
        x = jnp.full(4, 0.5)
        return int(jnp.sum(jnp.abs(hot.residual(x, state) - plain.residual(x, state)) > 0))

    # The patches cover a fixed number of points however fine the rod is.
    assert support(64) == support(256)
    assert support(256) < 40


def test_beam_buckles_past_the_critical_load() -> None:
    beam = systems.buckling_beam()
    mode = jnp.sin(jnp.pi * (jnp.arange(24) + 1) / 25)

    def settle(load: float, side: float) -> jax.Array:
        x = jnp.array([load, 0.0])
        return _solve_independently(beam.residual, x[None], side * mode[None], 30)[0]

    # Below the critical load both pushes settle straight.
    assert float(jnp.max(jnp.abs(settle(0.5, 1.0)))) < 1e-3
    # Above it the beam bows one way or the other.
    assert float(jnp.max(settle(1.5, 1.0))) > 0.7
    assert float(jnp.min(settle(1.5, -1.0))) < -0.7

    def stiffness(load: float) -> float:
        jacobian = jax.jacobian(beam.residual, argnums=1)(jnp.array([load, 0.0]), jnp.zeros(24))
        return float(jnp.min(jnp.abs(jnp.linalg.eigvalsh(jacobian))))

    # The straight beam loses its stiffness at the critical load.
    assert stiffness(1.0) < 0.05
    assert stiffness(0.5) > 0.3


def test_latch_ring_holds_two_states() -> None:
    ring = systems.latch_ring(4)
    x = jnp.array([4.0, 0.0])
    zero = jnp.tile(jnp.array([0.0, 1.0]), 4)
    one = jnp.tile(jnp.array([1.0, 0.0]), 4)
    stored_zero = _solve_independently(ring.residual, x[None], zero[None], 30)[0]
    stored_one = _solve_independently(ring.residual, x[None], one[None], 30)[0]
    assert float(jnp.linalg.norm(ring.residual(x, stored_zero))) < 1e-5
    assert float(jnp.linalg.norm(ring.residual(x, stored_one))) < 1e-5
    assert float(jnp.linalg.norm(stored_zero - stored_one)) > 1.0
