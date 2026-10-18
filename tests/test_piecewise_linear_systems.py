"""Piecewise linear approximations of an implicit test system.

The radiating rod is smooth, so its solution is not piecewise linear and each fit
here is an approximation. Fitting the latent coefficients of its reduced solver
keeps the structure, since the decode back to the field is affine, so a piecewise
linear latent is a piecewise linear field.
"""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from tests import implicit_systems


@pytest.fixture(scope="module")
def rod() -> tuple[sx.ReducedSolver, sx.Box]:
    """A radiating rod of 32 points with two heaters, and a rank-two reduced solver."""
    system = implicit_systems.radiating_rod(32, heaters=2)
    method = sx.ReducedSolve(
        form=sx.Affine(),
        unknown_domain=system.unknown_domain,
        rank=2,
        snapshots=32,
        newton_steps=12,
        reduced_steps=12,
        linear_solver="dense",
    )
    return sx.reduced_solver(system.residual, system.domain, method, samples=64), system.domain


def _relative_error(
    surrogate: sx.FittedSurrogate,
    solver: sx.ReducedSolver,
    points: jax.Array,
) -> float:
    """Mean latent error of a surrogate over the mean latent norm."""
    true = jax.vmap(solver.latent)(points)
    predicted = jax.vmap(surrogate)(points)
    return float(
        jnp.mean(jnp.linalg.norm(predicted - true, axis=1))
        / jnp.mean(jnp.linalg.norm(true, axis=1))
    )


def test_a_difference_of_max_affine_approximates_the_rod_better_than_an_affine_fit(
    rod: tuple[sx.ReducedSolver, sx.Box],
) -> None:
    solver, domain = rod
    points = domain.sample_unit(jax.random.uniform(jax.random.key(7), (100, 2)))

    def fitted(form: sx.Affine | sx.MaxAffineForm | sx.DifferenceOfMaxAffineForm):
        return sx.fit(solver.latent, domain, form, samples=256, worst_case_starts=0)

    affine = _relative_error(fitted(sx.Affine()), solver, points)
    piecewise = fitted(sx.DifferenceOfMaxAffineForm(pieces=4, restarts=6, steps=400))
    error = _relative_error(piecewise, solver, points)
    # Four positive and four negative pieces cut the affine error by several times.
    assert error < affine / 3
    assert piecewise.report.structure_cost == 8

    # The exported pieces reproduce the surrogate, and the affine decode of the
    # exported latent gives the field, so the field is piecewise linear in the input.
    export = sx.export(piecewise)
    assert isinstance(export, sx.DifferenceExport)

    def by_hand(x: jax.Array) -> jax.Array:
        positive = jnp.max(
            jnp.einsum("kmd,d->km", export.positive_slopes, x) + export.positive_intercepts, axis=1
        )
        negative = jnp.max(
            jnp.einsum("kmd,d->km", export.negative_slopes, x) + export.negative_intercepts, axis=1
        )
        return positive - negative

    latent = jax.vmap(by_hand)(points)
    assert jnp.allclose(latent, jax.vmap(piecewise)(points), atol=1e-4)
    field = solver.center + latent @ solver.basis
    true_field = jax.vmap(solver)(points)
    relative = jnp.linalg.norm(field - true_field) / jnp.linalg.norm(true_field)
    assert float(relative) < 0.1


def test_a_max_affine_fit_is_never_worse_than_an_affine_fit(
    rod: tuple[sx.ReducedSolver, sx.Box],
) -> None:
    # The rod's latents are concave, which a convex form fits badly, and the
    # alternating fit used to land far from the affine fit it could have kept.
    solver, domain = rod
    points = domain.sample_unit(jax.random.uniform(jax.random.key(7), (100, 2)))
    affine = sx.fit(solver.latent, domain, sx.Affine(), samples=256, worst_case_starts=0)
    for pieces in (2, 4, 8):
        form = sx.MaxAffineForm(pieces=pieces, restarts=6, iterations=25)
        convex = sx.fit(solver.latent, domain, form, samples=256, worst_case_starts=0)
        assert _relative_error(convex, solver, points) < 1.1 * _relative_error(
            affine, solver, points
        )
