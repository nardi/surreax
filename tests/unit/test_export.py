"""Tests for exporting a piecewise linear surrogate as arrays."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx


def _max_affine_value(export: sx.MaxAffineExport, x: jax.Array) -> jax.Array:
    return jnp.max(jnp.einsum("kmd,d->km", export.slopes, x) + export.intercepts, axis=1)


def test_max_affine_export_matches_the_surrogate_in_original_coordinates() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(
        target, domain, sx.MaxAffineForm(pieces=2, restarts=6), samples=400, worst_case_starts=0
    )
    export = sx.export(surrogate)
    assert isinstance(export, sx.MaxAffineExport)
    points = jax.random.uniform(jax.random.key(1), (50, 2), minval=-1.0, maxval=1.0)
    by_hand = jax.vmap(lambda x: _max_affine_value(export, x))(points)
    assert jnp.allclose(by_hand, jax.vmap(surrogate)(points), atol=1e-4)


@pytest.mark.slow
def test_difference_export_matches_the_surrogate() -> None:
    def wave(x: jax.Array) -> jax.Array:
        return (jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1]) - jnp.maximum(-x[0], x[1]))[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(
        wave,
        domain,
        sx.DifferenceOfMaxAffineForm(pieces=2, restarts=4, steps=300),
        samples=400,
        worst_case_starts=0,
    )
    export = sx.export(surrogate)
    assert isinstance(export, sx.DifferenceExport)

    def by_hand(x: jax.Array) -> jax.Array:
        positive = jnp.max(
            jnp.einsum("kmd,d->km", export.positive_slopes, x) + export.positive_intercepts, axis=1
        )
        negative = jnp.max(
            jnp.einsum("kmd,d->km", export.negative_slopes, x) + export.negative_intercepts, axis=1
        )
        return positive - negative

    points = jax.random.uniform(jax.random.key(1), (50, 2), minval=-1.0, maxval=1.0)
    assert jnp.allclose(jax.vmap(by_hand)(points), jax.vmap(surrogate)(points), atol=1e-4)


def test_export_refuses_a_reduced_output_space() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.maximum(x[0], x[1]) * jnp.array([1.0, 2.0, 3.0])

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(
        target,
        domain,
        sx.MaxAffineForm(pieces=2, restarts=4),
        samples=400,
        worst_case_starts=0,
        reduce_outputs=sx.pca_outputs(rank=1),
    )
    with pytest.raises(ValueError, match="identity output reduction"):
        sx.export(surrogate)


def test_export_refuses_a_form_without_pieces() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (x[0] + x[1])[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.Affine(), samples=128, worst_case_starts=0)
    with pytest.raises(TypeError, match="no array export"):
        sx.export(surrogate)
