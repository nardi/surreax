"""Tests for the input-convex and ReLU network forms."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax.normalize import Normalizer

_DOMAIN = sx.box([0.0, -2.0], [3.0, 5.0])


def _convex(x: jax.Array) -> jax.Array:
    return (jnp.abs(x[0] - 1.5) + 0.5 * jnp.abs(x[1] - 1.0) + 0.3 * x[0])[None]


def _nonconvex(x: jax.Array) -> jax.Array:
    return (jnp.abs(x[0] - 1.0) - jnp.abs(x[1] - 2.0) + 0.5 * x[0])[None]


def _fit(
    form: sx.ConvexNetworkForm | sx.ReluNetworkForm,
    target: Callable[[jax.Array], jax.Array],
    *,
    samples: int = 512,
    derivative_weight: float = 0.0,
    warm_start: sx.FittedSurrogate | None = None,
) -> sx.FittedSurrogate:
    return sx.fit(
        target,
        _DOMAIN,
        form,
        samples=samples,
        derivative_weight=derivative_weight,
        warm_start=warm_start,
        worst_case_starts=0,
    )


@pytest.mark.slow
def test_a_convex_network_is_convex_with_nonnegative_hidden_weights() -> None:
    surrogate = _fit(sx.ConvexNetworkForm(width=8, depth=3, steps=300), _convex)
    model = surrogate.model
    assert isinstance(model, sx.ConvexNetwork)
    assert all(bool(jnp.all(weights >= 0)) for weights in model.hidden_weights())
    keys = jax.random.split(jax.random.key(0), 2)
    left = _DOMAIN.sample_unit(jax.random.uniform(keys[0], (200, 2)))
    right = _DOMAIN.sample_unit(jax.random.uniform(keys[1], (200, 2)))
    middle = jax.vmap(surrogate)((left + right) / 2)
    chord = (jax.vmap(surrogate)(left) + jax.vmap(surrogate)(right)) / 2
    assert float(jnp.max(middle - chord)) < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_convex_network_of_any_depth_fits_a_convex_target(depth: int) -> None:
    surrogate = _fit(sx.ConvexNetworkForm(width=12, depth=depth, steps=1500), _convex)
    # The deeper networks train a little slower, so the bound has some room.
    assert float(surrogate.report.value_mean) < 0.08


@pytest.mark.slow
def test_a_relu_network_fits_a_nonconvex_piecewise_linear_target() -> None:
    surrogate = _fit(sx.ReluNetworkForm(width=16, depth=2, steps=2000), _nonconvex)
    assert float(surrogate.report.value_mean) < 0.05
    # Piecewise linear, so the Hessian vanishes away from the kinks.
    point = jnp.array([2.2, 3.7])
    assert float(jnp.max(jnp.abs(jax.hessian(surrogate)(point)))) < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize(
    "form", [sx.ConvexNetworkForm(width=8, steps=200), sx.ReluNetworkForm(width=8, steps=200)]
)
def test_the_networks_survive_jit_vmap_and_grad(
    form: sx.ConvexNetworkForm | sx.ReluNetworkForm,
) -> None:
    surrogate = _fit(form, _convex, samples=128)
    batch = _DOMAIN.sample_unit(jax.random.uniform(jax.random.key(1), (8, 2)))
    assert eqx.filter_jit(jax.vmap(surrogate))(batch).shape == (8, 1)
    assert jax.grad(lambda x: surrogate(x)[0])(batch[0]).shape == (2,)


@pytest.mark.slow
@pytest.mark.parametrize(
    "form",
    [sx.ConvexNetworkForm(width=8, steps=600), sx.ReluNetworkForm(width=8, steps=600)],
)
def test_derivative_supervision_and_warm_starts_reach_the_networks(
    form: sx.ConvexNetworkForm | sx.ReluNetworkForm,
) -> None:
    first = _fit(form, _convex, samples=256, derivative_weight=0.5)
    again = _fit(form, _convex, samples=256, derivative_weight=0.5, warm_start=first)
    assert jnp.isfinite(first.report.derivative_mean)
    # The warm start joins the restarts, so a refit is no worse after training.
    assert float(again.report.value_mean) <= 1.5 * float(first.report.value_mean) + 1e-3


@pytest.mark.slow
def test_the_convex_export_reproduces_the_surrogate_in_original_coordinates() -> None:
    surrogate = _fit(sx.ConvexNetworkForm(width=8, depth=2, steps=500), _convex)
    export = sx.export(surrogate)
    assert isinstance(export, sx.ConvexNetworkExport)
    assert all(bool(jnp.all(weights >= 0)) for weights in export.hidden_weights)

    def by_hand(x: jax.Array) -> jax.Array:
        z = jax.nn.relu(export.input_weights[0] @ x + export.biases[0])
        for layer in range(1, len(export.input_weights) - 1):
            z = jax.nn.relu(
                export.hidden_weights[layer - 1] @ z
                + export.input_weights[layer] @ x
                + export.biases[layer]
            )
        return export.hidden_weights[-1] @ z + export.input_weights[-1] @ x + export.biases[-1]

    points = _DOMAIN.sample_unit(jax.random.uniform(jax.random.key(2), (50, 2)))
    assert jnp.allclose(jax.vmap(by_hand)(points), jax.vmap(surrogate)(points), atol=1e-3)

    # A nonpositive output scale would flip convexity, so the export refuses it.
    flipped = eqx.tree_at(
        lambda s: s.output_normalizer, surrogate, Normalizer(
            center=surrogate.output_normalizer.center, scale=-surrogate.output_normalizer.scale
        ),
    )
    with pytest.raises(ValueError, match="positive"):
        sx.export(flipped)


@pytest.mark.slow
def test_the_relu_export_reproduces_the_surrogate_in_original_coordinates() -> None:
    surrogate = _fit(sx.ReluNetworkForm(width=8, depth=2, steps=500), _nonconvex)
    export = sx.export(surrogate)
    assert isinstance(export, sx.ReluNetworkExport)

    def by_hand(x: jax.Array) -> jax.Array:
        activation = x
        for weights, biases in zip(export.weights[:-1], export.biases[:-1], strict=True):
            activation = jax.nn.relu(weights @ activation + biases)
        return export.weights[-1] @ activation + export.biases[-1]

    points = _DOMAIN.sample_unit(jax.random.uniform(jax.random.key(3), (50, 2)))
    assert jnp.allclose(jax.vmap(by_hand)(points), jax.vmap(surrogate)(points), atol=1e-3)
    reduced = sx.fit(
        lambda x: jnp.stack([x[0], 2 * x[0], x[1], 2 * x[1]]),
        _DOMAIN,
        sx.ReluNetworkForm(width=4, steps=50),
        samples=64,
        reduce_outputs=sx.pca_outputs(rank=2),
        worst_case_starts=0,
    )
    with pytest.raises(ValueError, match="identity output reduction"):
        sx.export(reduced)
