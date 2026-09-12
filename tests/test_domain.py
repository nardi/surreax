"""Tests for the domain specification."""

import jax.numpy as jnp

from surreax.domain import Box, Domain, box, box_from_data


def test_box_reports_dimension_and_bounds() -> None:
    region = box([0.0, -1.0], [2.0, 1.0])
    assert region.dimension == 2
    lower, upper = region.bounds()
    assert jnp.array_equal(lower, jnp.array([0.0, -1.0]))
    assert jnp.array_equal(upper, jnp.array([2.0, 1.0]))


def test_box_maps_unit_corners_to_its_own_corners() -> None:
    region = box([0.0, -1.0], [2.0, 1.0])
    corners = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    mapped = region.sample_unit(corners)
    assert jnp.array_equal(mapped[0], jnp.array([0.0, -1.0]))
    assert jnp.array_equal(mapped[1], jnp.array([2.0, 1.0]))


def test_box_from_data_takes_the_tightest_bounds() -> None:
    points = jnp.array([[1.0, 5.0], [-2.0, 3.0], [0.0, 4.0]])
    region = box_from_data(points)
    lower, upper = region.bounds()
    assert jnp.array_equal(lower, jnp.array([-2.0, 3.0]))
    assert jnp.array_equal(upper, jnp.array([1.0, 5.0]))


def test_box_satisfies_the_domain_protocol() -> None:
    assert isinstance(box([0.0], [1.0]), Domain)
    assert isinstance(Box(lower=jnp.zeros(1), upper=jnp.ones(1)), Domain)
