"""Reduced solvers of implicit test systems, as targets for piecewise linear fits.

Fitting the latent coefficients of a reduced solver keeps the structure, since the
decode back to the field is affine, so a piecewise linear latent is a piecewise
linear field. The rod has two inputs and 32 points, the network is a 12 by 12 mesh
of 144 nodes with four sources, wired with random long wires and a diode on every
third wire, and the thermostat rod has two inputs and 32 points.
"""

import jax
import jax.numpy as jnp

import surreax as sx
from tests import implicit_systems


class Case:
    """A reduced solver of an implicit system, with the inputs its fits are scored on."""

    def __init__(self, system: implicit_systems.ImplicitSystem, rank: int, start: float) -> None:
        unknowns = system.unknown_domain.dimension
        method = sx.ReducedSolve(
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

    def fit(self, form: sx.Form) -> sx.FittedSurrogate:
        """Fit a form to the latent coefficients of the solver."""
        return sx.fit(
            self.solver.latent, self.domain, form, samples=512, worst_case_starts=0
        )


def build(name: str) -> Case:
    """The case called `name`, one of `rod`, `network` and `thermostat`."""
    if name == "rod":
        return Case(implicit_systems.radiating_rod(32, heaters=2), rank=2, start=2.0)
    if name == "network":
        return Case(implicit_systems.diode_network(12, 12, sources=4), rank=4, start=0.5)
    return Case(implicit_systems.thermostat_rod(32), rank=3, start=2.0)
