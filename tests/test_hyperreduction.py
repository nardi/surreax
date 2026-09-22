"""Tests for hyper-reduction, the reduced solve that reads only a few entries."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import surreax as sx
from tests import implicit_systems


def _relative(
    system: implicit_systems.ImplicitSystem, solver: eqx.Module, points: jax.Array
) -> float:
    """Mean residual a solver leaves, over the mean residual at zero."""
    unknown = system.unknown_domain.dimension
    left = jax.vmap(lambda x: jnp.linalg.norm(system.residual(x, solver(x))))(points)
    base = jax.vmap(lambda x: jnp.linalg.norm(system.residual(x, jnp.zeros(unknown))))(points)
    return float(jnp.mean(left) / jnp.mean(base))


def _hot_spot_solvers(points: int) -> tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver]:
    system = implicit_systems.rod_with_hot_spots(points)
    solver = sx.reduced_solver(
        system.residual,
        system.domain,
        sx.ReducedSolve(
            form=sx.Affine(),
            unknown_domain=system.unknown_domain,
            rank=6,
            snapshots=32,
            newton_steps=8,
            reduced_steps=8,
        ),
        samples=64,
    )
    return system, solver


@pytest.fixture(scope="module")
def hot_spots() -> tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver, sx.HyperReducedSolver]:
    system, solver = _hot_spot_solvers(256)
    return system, solver, sx.hyper_reduce(solver, system.residual, system.domain)


def test_hyper_reduced_solve_reads_a_fixed_number_of_rows() -> None:
    small_system, small_solver = _hot_spot_solvers(256)
    large_system, large_solver = _hot_spot_solvers(2048)
    small = sx.hyper_reduce(small_solver, small_system.residual, small_system.domain)
    large = sx.hyper_reduce(large_solver, large_system.residual, large_system.domain)
    # The rows read depend on the samples and the stencil, not on the rod.
    assert abs(small.rows.shape[0] - large.rows.shape[0]) <= 3
    assert large.rows.shape[0] < 0.03 * 2048


def test_hyper_reduced_solve_matches_the_full_reduced_solve(
    hot_spots: tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver, sx.HyperReducedSolver],
) -> None:
    system, solver, hyper = hot_spots
    points = system.domain.sample_unit(jax.random.uniform(jax.random.key(9), (32, 4)))
    assert _relative(system, hyper, points) < 1.2 * _relative(system, solver, points)
    compiled = eqx.filter_jit(lambda s, x: s(x))
    assert jnp.allclose(compiled(hyper, points[0]), hyper(points[0]), atol=1e-5)


def test_samples_land_on_the_radiating_patches(
    hot_spots: tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver, sx.HyperReducedSolver],
) -> None:
    # The nonlinear term lives on three bare patches, so most samples go there.
    _, _, hyper = hot_spots
    position = np.asarray(hyper.sampled) / 255
    near = np.min(np.abs(position[:, None] - np.asarray(implicit_systems.PATCHES)), axis=1)
    assert np.mean(near < 4 / 256) > 0.5


def test_hyper_reduced_derivative_matches_finite_differences(
    hot_spots: tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver, sx.HyperReducedSolver],
) -> None:
    _, _, hyper = hot_spots
    point = jnp.array([0.3, 0.6, 0.2, 0.8])
    implicit_jacobian = jax.jacobian(hyper.latent)(point)
    eps = 1e-2
    finite = jnp.stack(
        [
            (hyper.latent(point.at[i].add(eps)) - hyper.latent(point.at[i].add(-eps))) / (2 * eps)
            for i in range(4)
        ],
        axis=1,
    )
    scale = float(jnp.max(jnp.abs(implicit_jacobian)))
    assert float(jnp.max(jnp.abs(implicit_jacobian - finite))) < 1e-3 * scale


def test_learned_manifold_hyper_reduces_on_a_graph() -> None:
    circuit = implicit_systems.diode_network(8, 8)
    learned = sx.autoencoder_solver(
        circuit.residual,
        circuit.domain,
        sx.AutoencoderSolve(form=sx.Affine(), unknown_dimension=64, rank=4, steps=600),
    )
    hyper = sx.hyper_reduce(learned, circuit.residual, circuit.domain, basis_size=12, samples=20)
    points = circuit.domain.sample_unit(jax.random.uniform(jax.random.key(9), (32, 4)))
    assert hyper.rows.shape[0] < 64
    assert _relative(circuit, hyper, points) < 1.1 * _relative(circuit, learned, points)


def test_hyper_reduce_needs_enough_samples(
    hot_spots: tuple[implicit_systems.ImplicitSystem, sx.ReducedSolver, sx.HyperReducedSolver],
) -> None:
    system, solver, _ = hot_spots
    with pytest.raises(ValueError, match="basis_size"):
        sx.hyper_reduce(solver, system.residual, system.domain, basis_size=16, samples=8)
