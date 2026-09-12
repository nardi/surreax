"""Tests for the affine normalizer."""

import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from surreax.normalize import from_bounds, from_moments


def test_from_bounds_sends_the_box_to_minus_one_and_one() -> None:
    normalizer = from_bounds(jnp.array([0.0, -4.0]), jnp.array([2.0, 4.0]))
    assert jnp.allclose(normalizer.forward(jnp.array([0.0, -4.0])), jnp.array([-1.0, -1.0]))
    assert jnp.allclose(normalizer.forward(jnp.array([2.0, 4.0])), jnp.array([1.0, 1.0]))


def test_from_moments_centers_and_whitens() -> None:
    values = jnp.array([[1.0], [3.0], [5.0]])
    normalizer = from_moments(values)
    transformed = jax.vmap(normalizer.forward)(values)
    assert jnp.allclose(jnp.mean(transformed), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.std(transformed), 1.0, atol=1e-6)


def test_constant_coordinate_maps_to_zero_without_dividing_by_zero() -> None:
    normalizer = from_bounds(jnp.array([2.0]), jnp.array([2.0]))
    assert jnp.allclose(normalizer.forward(jnp.array([2.0])), jnp.array([0.0]))


@given(
    st.lists(st.floats(-10, 10, allow_nan=False), min_size=1, max_size=4),
    st.lists(st.floats(0.1, 5.0, allow_nan=False), min_size=1, max_size=4),
)
@settings(deadline=None)
def test_forward_then_inverse_is_identity(centers: list[float], widths: list[float]) -> None:
    size = min(len(centers), len(widths))
    lower = jnp.array(centers[:size])
    upper = lower + jnp.array(widths[:size])
    normalizer = from_bounds(lower, upper)
    point = (lower + upper) / 3
    assert jnp.allclose(normalizer.inverse(normalizer.forward(point)), point, atol=1e-5)
