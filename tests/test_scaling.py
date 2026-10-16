"""Scaling tests on a large sparse implicit system.

The system has a small input, a large unknown, and a nearest-neighbor Jacobian,
so the sparse solves, direct or matrix-free, and the output reduction should
reach a scale the dense path cannot. The residual is reported relative to the
residual at zero, so the check does not depend on the state size.
"""

import jax.numpy as jnp
import pytest

import surreax as sx
from surreax.implicit import newton_setup
from tests import implicit_systems, systems


def _baseline_residual() -> float:
    """The residual norm at a zero field, the scale to measure against."""
    point = jnp.array([0.5, -0.3, 0.7, 0.1])
    zero = jnp.zeros(systems.LARGE_STATE)
    return float(jnp.linalg.norm(systems.large_sparse_residual(point, zero)))


def test_reduced_solve_scales_to_a_large_field() -> None:
    method = sx.ReducedSolve(
        form=sx.Affine(),
        unknown_domain=systems.large_sparse_unknown_domain(),
        rank=systems.LARGE_INPUTS,
        snapshots=16,
        newton_steps=4,
        reduced_steps=4,
    )
    surrogate = sx.fit_implicit(
        systems.large_sparse_residual, systems.large_sparse_domain(), method, samples=64
    )
    # The large field is predicted through four latent coefficients.
    assert surrogate.output_reduction.decode_matrix.shape == (systems.LARGE_STATE, 4)
    assert float(surrogate.report.value_mean) / _baseline_residual() < 1e-3


def test_exact_solve_scales_to_a_large_field() -> None:
    method = sx.ExactSolve(
        form=sx.Affine(),
        unknown_domain=systems.large_sparse_unknown_domain(),
        rank=systems.LARGE_INPUTS,
        newton_steps=4,
    )
    surrogate = sx.fit_implicit(
        systems.large_sparse_residual, systems.large_sparse_domain(), method, samples=64
    )
    assert surrogate.output_reduction.encode_matrix.shape == (4, systems.LARGE_STATE)
    assert float(surrogate.report.value_mean) / _baseline_residual() < 1e-3


def test_reduced_solver_call_is_cheap_at_scale() -> None:
    # The distilled surrogate maps the small input straight to the large field.
    method = sx.ReducedSolve(
        form=sx.Affine(),
        unknown_domain=systems.large_sparse_unknown_domain(),
        rank=systems.LARGE_INPUTS,
        snapshots=16,
        newton_steps=4,
        reduced_steps=4,
    )
    surrogate = sx.fit_implicit(
        systems.large_sparse_residual, systems.large_sparse_domain(), method, samples=64
    )
    field = surrogate(jnp.array([0.4, -0.2, 0.6, 0.0]))
    assert field.shape == (systems.LARGE_STATE,)
    left = jnp.linalg.norm(
        systems.large_sparse_residual(jnp.array([0.4, -0.2, 0.6, 0.0]), field)
    )
    assert float(left) / _baseline_residual() < 1e-2


@pytest.mark.parametrize("linear_solver", ["sparse", "matrix_free"])
def test_newton_solve_reaches_a_field_a_dense_jacobian_cannot_hold(
    linear_solver: sx.LinearSolver,
) -> None:
    # A dense Jacobian at this size would take 160 GB. The sparse step factors
    # the tridiagonal Jacobian, and the matrix-free step only needs products with
    # it, which remains the route when a factorization does not fit in memory.
    points = 200_000
    rod = implicit_systems.heated_rod(points)
    x = jnp.full(4, 0.5)
    field = jnp.zeros(points)
    setup = newton_setup(rod.residual, linear_solver, None, x, field)
    assert (setup.pattern is not None) == (linear_solver == "sparse")
    for _ in range(3):
        field = setup.step(rod.residual, x, field)
    start = jnp.linalg.norm(rod.residual(x, jnp.zeros(points)))
    assert float(jnp.linalg.norm(rod.residual(x, field)) / start) < 1e-5


def test_learned_manifold_training_lowers_the_residual_with_more_steps() -> None:
    rod = implicit_systems.radiating_rod(32)

    def after(steps: int) -> float:
        method = sx.AutoencoderSolve(
            form=sx.Affine(), unknown_dimension=32, rank=4, steps=steps
        )
        return float(sx.autoencoder_solver(rod.residual, rod.domain, method).training_residual)

    assert after(600) < after(100) / 2
