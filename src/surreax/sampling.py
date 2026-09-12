"""Drawing inputs from a domain.

Uniform random points leave gaps. Low-discrepancy sequences cover a region
more evenly, so the surrogate sees the whole domain rather than clusters. The
sampler runs once at fit time on the host, so it uses SciPy's quasi-Monte Carlo
engines rather than a JAX equivalent.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.stats import qmc

from surreax.domain import Domain


def sobol_unit(count: int, dimension: int, *, seed: int) -> Float[Array, "n d"]:
    """Draw `count` scrambled Sobol points from the unit hypercube."""
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = engine.random(count)
    return jnp.asarray(points, float)


def sample(domain: Domain, count: int, *, seed: int) -> Float[Array, "n d"]:
    """Draw `count` low-discrepancy points from a domain."""
    unit = sobol_unit(count, domain.dimension, seed=seed)
    return domain.sample_unit(unit)
