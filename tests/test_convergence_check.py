"""The convergence check on the Newton solves for the full states."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax.implicit import NewtonSetup, _newton_step, _solve_to_convergence
from tests import implicit_systems


def _diode_network() -> implicit_systems.ImplicitSystem:
    return implicit_systems.diode_network(12, 12, sources=4)


def _far_starts() -> sx.Box:
    return sx.box([-5.0] * 144, [5.0] * 144)


def _near_starts() -> sx.Box:
    return sx.box([-0.5] * 144, [0.5] * 144)


def test_exact_solve_raises_when_the_newton_solves_diverge() -> None:
    system = _diode_network()
    far = sx.ExactSolve(unknown_domain=_far_starts(), rank=4, newton_steps=14)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        sx.fit_implicit(system.residual, system.domain, far, sx.Affine(), samples=32)
    # The same solves from a closer start converge, and the fit goes through.
    near = sx.ExactSolve(unknown_domain=_near_starts(), rank=4, newton_steps=14)
    fitted = sx.fit_implicit(system.residual, system.domain, near, sx.Affine(), samples=32)
    assert fitted is not None


def _reduced_solve(tolerance: float | None) -> sx.ReducedSolve:
    return sx.ReducedSolve(
        unknown_domain=_far_starts(),
        rank=4,
        snapshots=8,
        newton_steps=14,
        convergence_tolerance=tolerance,
    )


def test_reduced_solve_raises_on_unconverged_snapshots_unless_told_not_to() -> None:
    system = _diode_network()
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        sx.reduced_solver(system.residual, system.domain, _reduced_solve(1e-3))
    unchecked = _reduced_solve(None)
    assert sx.reduced_solver(system.residual, system.domain, unchecked) is not None


def test_autoencoder_snapshots_are_checked_too() -> None:
    # Two Newton steps on an arctangent leave a residual of 16% of the zero state.
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.arctan(y) - x[0]

    domain = sx.box([1.3], [1.5])
    method = sx.AutoencoderSolve(
        unknown_dimension=2,
        rank=1,
        snapshots=4,
        steps=2,
        pretrain_steps=2,
        newton_steps=2,
        linear_solver="dense",
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        sx.autoencoder_solver(residual, domain, method)


def test_the_check_runs_under_jit() -> None:
    system = _diode_network()
    inputs = system.domain.sample_unit(jax.random.uniform(jax.random.key(0), (4, 4)))
    far = jax.random.uniform(jax.random.key(1), (4, 144), minval=-5.0, maxval=5.0)
    near = jax.random.uniform(jax.random.key(2), (4, 144), minval=-0.5, maxval=0.5)
    setup = NewtonSetup(step=_newton_step, pattern=None)

    @eqx.filter_jit
    def solve(starts: jax.Array) -> jax.Array:
        return _solve_to_convergence(setup, system.residual, inputs, starts, 14, 1e-3)

    assert solve(near).shape == (4, 144)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        solve(far)
