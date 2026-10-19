"""Tests for the linear reduction stage."""

import jax
import jax.numpy as jnp

from surreax.reduction import (
    LinearReduction,
    active_subspace_inputs,
    identity,
    pca_inputs,
    pca_outputs,
)


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


def test_pca_outputs_reconstructs_low_rank_data_at_fixed_rank() -> None:
    coefficients = jax.random.normal(jax.random.key(0), (40, 2))
    basis = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    outputs = coefficients @ basis
    reduction = pca_outputs(rank=2)(outputs)
    assert reduction.encode_matrix.shape == (2, 5)
    reconstructed = jax.vmap(lambda y: reduction.inverse(reduction.forward(y)))(outputs)
    assert jnp.allclose(reconstructed, outputs, atol=1e-4)


def test_pca_inputs_keeps_enough_directions_for_the_variance() -> None:
    # Data varies in two of three coordinates, so 99% of variance is two directions.
    points = jax.random.normal(jax.random.key(1), (64, 2)) @ jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    reduction = pca_inputs(variance=0.99)(points, jnp.zeros((64, 1, 3)))
    assert reduction.encode_matrix.shape == (2, 3)


def test_active_subspace_recovers_a_ridge_direction() -> None:
    direction = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    # Every Jacobian points along one direction, so the active subspace is it.
    jacobians = (
        jax.random.normal(jax.random.key(2), (32, 1))[:, :, None] * direction[None, None, :]
    )
    inputs = jax.random.normal(jax.random.key(3), (32, 3))
    reduction = active_subspace_inputs(rank=1)(inputs, jacobians)
    recovered = reduction.encode_matrix[0]
    assert jnp.allclose(jnp.abs(jnp.dot(recovered, direction)), 1.0, atol=1e-5)
