"""Piecewise linear approximations of implicit test systems.

The radiating rod and the diode network are smooth, so their solutions are not
piecewise linear and each fit here is an approximation. Fitting the latent
coefficients of a reduced solver keeps the structure, since the decode back to the
field is affine, so a piecewise linear latent is a piecewise linear field. The rod
has two inputs and 32 points. The network is a 12 by 12 mesh of 144 nodes with four
sources, wired with random long wires and a diode on every third wire.
"""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from tests import implicit_systems


class Case:
    """A reduced solver of an implicit system, with the box its inputs are drawn from."""

    def __init__(self, system: implicit_systems.ImplicitSystem, rank: int, start: float) -> None:
        unknowns = system.unknown_domain.dimension
        method = sx.ReducedSolve(
            form=sx.Affine(),
            # The diode exponentials make Newton diverge from far starts, so a small
            # start box keeps the snapshot solves converged.
            unknown_domain=sx.box([-start] * unknowns, [start] * unknowns),
            rank=rank,
            snapshots=48,
            newton_steps=14,
            reduced_steps=12,
            linear_solver="dense",
        )
        self.domain = system.domain
        self.residual = system.residual
        self.solver = sx.reduced_solver(system.residual, system.domain, method, samples=96)
        unit = jax.random.uniform(jax.random.key(7), (100, system.domain.dimension))
        self.points = system.domain.sample_unit(unit)
        self.latent = jax.vmap(self.solver.latent)(self.points)

    def latent_error(self, surrogate: sx.FittedSurrogate) -> float:
        """Mean latent error of a surrogate over the mean latent norm."""
        predicted = jax.vmap(surrogate)(self.points)
        return float(
            jnp.mean(jnp.linalg.norm(predicted - self.latent, axis=1))
            / jnp.mean(jnp.linalg.norm(self.latent, axis=1))
        )

    def fit(
        self, form: sx.Affine | sx.MaxAffineForm | sx.DifferenceOfMaxAffineForm
    ) -> sx.FittedSurrogate:
        return sx.fit(
            self.solver.latent,
            self.domain,
            form,
            samples=512,
            worst_case_starts=0,
        )


@pytest.fixture(scope="module", params=["rod", "network"])
def case(request: pytest.FixtureRequest) -> Case:
    if request.param == "rod":
        return Case(implicit_systems.radiating_rod(32, heaters=2), rank=2, start=2.0)
    return Case(implicit_systems.diode_network(12, 12, sources=4), rank=4, start=0.5)


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
