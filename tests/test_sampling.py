"""Tests for low-discrepancy sampling."""

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from surreax.domain import box
from surreax.sampling import sample, sobol_unit


def test_sobol_points_lie_in_the_unit_cube() -> None:
    points = sobol_unit(64, 3, seed=0)
    assert points.shape == (64, 3)
    assert jnp.all(points >= 0.0)
    assert jnp.all(points < 1.0)


def test_sample_stays_inside_the_box() -> None:
    region = box([-2.0, 1.0], [3.0, 4.0])
    points = sample(region, 64, seed=1)
    lower, upper = region.bounds()
    assert jnp.all(points >= lower)
    assert jnp.all(points <= upper)


def test_sobol_covers_more_evenly_than_uniform_random() -> None:
    count, dimension = 64, 2
    sobol = np.asarray(sobol_unit(count, dimension, seed=0))
    random = np.random.default_rng(0).random((count, dimension))
    assert qmc.discrepancy(sobol) < qmc.discrepancy(random)
