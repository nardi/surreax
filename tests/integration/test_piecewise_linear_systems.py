"""Piecewise linear approximations of implicit test systems.

The radiating rod, the diode network and the thermostat rod are smooth, so their
solutions are not piecewise linear and each fit here is an approximation, see
`piecewise_cases` for how the latent coefficients of a reduced solver are fitted.
"""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from tests.piecewise_cases import Case, build


@pytest.fixture(scope="module", params=["rod", "network", "thermostat"])
def case(request: pytest.FixtureRequest) -> Case:
    return build(request.param)


def test_a_difference_of_max_affine_approximates_the_system_better_than_an_affine_fit(
    case: Case,
) -> None:
    affine = case.latent_error(case.fit(sx.Affine()))
    piecewise = case.fit(sx.DifferenceOfMaxAffineForm(pieces=4, restarts=6, steps=400))
    # Four positive and four negative pieces cut the affine error by several times.
    assert case.latent_error(piecewise) < affine / 3
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

    latent = jax.vmap(by_hand)(case.points)
    assert jnp.allclose(latent, jax.vmap(piecewise)(case.points), atol=1e-4)
    field = case.solver.center + latent @ case.solver.basis
    true_field = jax.vmap(case.solver)(case.points)
    relative = jnp.linalg.norm(field - true_field) / jnp.linalg.norm(true_field)
    assert float(relative) < 0.1

    # The piecewise linear field leaves nearly the residual the reduced solve does.
    def mean_residual(fields: jax.Array) -> float:
        return float(
            jnp.mean(jnp.linalg.norm(jax.vmap(case.residual)(case.points, fields), axis=1))
        )

    assert mean_residual(field) < 1.5 * mean_residual(true_field) + 1e-3


def test_a_max_affine_fit_is_never_worse_than_an_affine_fit(case: Case) -> None:
    # The latents are not convex, which a max-affine form fits badly, and the
    # alternating fit used to land far from the affine fit it could have kept.
    affine = case.latent_error(case.fit(sx.Affine()))
    for pieces in (2, 4, 8):
        convex = case.fit(sx.MaxAffineForm(pieces=pieces, restarts=6, iterations=25))
        assert case.latent_error(convex) < 1.1 * affine
