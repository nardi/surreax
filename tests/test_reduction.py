"""Tests for the linear reduction stage."""

import jax.numpy as jnp

from surreax.reduction import LinearReduction, identity


def test_identity_reduction_passes_values_through() -> None:
    reduction = identity(3)
    values = jnp.array([1.0, -2.0, 0.5])
    assert jnp.array_equal(reduction.forward(values), values)
    assert jnp.array_equal(reduction.inverse(values), values)


def test_affine_reduction_encodes_and_decodes() -> None:
    reduction = LinearReduction(
        encode_matrix=jnp.array([[1.0, 0.0]]),
        encode_bias=jnp.array([1.0]),
        decode_matrix=jnp.array([[2.0], [0.0]]),
        decode_bias=jnp.array([0.0, 3.0]),
    )
    encoded = reduction.forward(jnp.array([4.0, 9.0]))
    assert jnp.array_equal(encoded, jnp.array([5.0]))
    decoded = reduction.inverse(encoded)
    assert jnp.array_equal(decoded, jnp.array([10.0, 3.0]))
