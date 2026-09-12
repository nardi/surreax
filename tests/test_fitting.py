"""Tests for the closed-form linear fit."""

import jax
import jax.numpy as jnp

from surreax.fitting import LinearModel, fit_linear
from surreax.forms import Affine


def test_fit_linear_recovers_an_affine_map() -> None:
    weights = jnp.array([[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]])
    inputs = jax.random.normal(jax.random.key(0), (128, 2))
    features = jax.vmap(Affine().features)(inputs)
    targets = features @ weights.T
    model = fit_linear(Affine(), inputs, targets, ridge=0.0)
    assert jnp.allclose(model.weights, weights, atol=1e-4)


def test_linear_model_applies_weights_to_features() -> None:
    model = LinearModel(weights=jnp.array([[1.0, 2.0, 3.0]]), form=Affine())
    prediction = model(jnp.array([1.0, 1.0]))
    assert jnp.array_equal(prediction, jnp.array([1.0 + 2.0 + 3.0]))
