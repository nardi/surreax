"""Drawing inputs from a domain.

Uniform random points leave gaps. Low-discrepancy sequences cover a region
more evenly, so the surrogate sees the whole domain rather than clusters. The
sampler runs once at fit time on the host, so it uses SciPy's quasi-Monte Carlo
engines rather than a JAX equivalent. Sobol points are the default, and Halton
points and a Latin hypercube are the alternatives.
"""

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.stats import qmc

from surreax.domain import Domain

Sampler = Literal["sobol", "halton", "latin_hypercube"]
"""The sequence that places points in the unit hypercube."""


def sobol_unit(count: int, dimension: int, *, seed: int) -> Float[Array, "n d"]:
    """Draw `count` scrambled Sobol points from the unit hypercube."""
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = engine.random(count)
    return jnp.asarray(points, float)


def unit_points(
    count: int, dimension: int, *, seed: int, sampler: Sampler = "sobol"
) -> Float[Array, "n d"]:
    """Draw `count` points from the unit hypercube with the chosen sequence."""
    if sampler == "sobol":
        return sobol_unit(count, dimension, seed=seed)
    if sampler == "halton":
        engine: qmc.QMCEngine = qmc.Halton(d=dimension, scramble=True, seed=seed)
    else:
        engine = qmc.LatinHypercube(d=dimension, seed=seed)
    return jnp.asarray(engine.random(count), float)


def sample(
    domain: Domain, count: int, *, seed: int, sampler: Sampler = "sobol"
) -> Float[Array, "n d"]:
    """Draw `count` low-discrepancy points from a domain."""
    unit = unit_points(count, domain.dimension, seed=seed, sampler=sampler)
    return domain.sample_unit(unit)
