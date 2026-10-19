"""Tests for per-block composite fitting."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx


def _separable(x: jax.Array) -> jax.Array:
    # Three independent blocks: {x0, x1} -> out0, {x2} -> out1, {x3, x4} -> out2.
    return jnp.array([jnp.sin(x[0]) * x[1], 2 * x[2], x[3] ** 2 - x[4]])


@pytest.mark.slow
def test_composite_matches_the_target_on_a_separable_function() -> None:
    domain = sx.box([-1.0] * 5, [1.0] * 5)
    surrogate = sx.fit_separable(_separable, domain, sx.ChebyshevPolynomial(degree=3), samples=256)
    assert surrogate.input_indices == ((0, 1), (2,), (3, 4))
    assert surrogate.output_indices == ((0,), (1,), (2,))
    point = jnp.array([0.3, -0.4, 0.5, 0.2, -0.1])
    assert jnp.allclose(surrogate(point), _separable(point), atol=1e-2)
    # A truly separable target drops no coupling.
    assert float(surrogate.report.projection_max) < 1e-5


@pytest.mark.slow
def test_per_block_forms_fit_a_linear_block_exactly() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.array([2 * x[0] - 1.0, jnp.sin(3 * x[1])])

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    forms = [sx.Affine(), sx.ChebyshevPolynomial(degree=5)]
    surrogate = sx.fit_separable(target, domain, forms, samples=256)
    # The linear block, fitted with the affine form, is exact.
    assert float(surrogate.blocks[0].report.value_max) < 1e-4
    point = jnp.array([0.4, 0.5])
    assert jnp.allclose(surrogate(point), target(point), atol=1e-2)


@pytest.mark.slow
def test_composite_survives_jit_vmap_and_grad() -> None:
    domain = sx.box([-1.0] * 5, [1.0] * 5)
    surrogate = sx.fit_separable(_separable, domain, sx.ChebyshevPolynomial(degree=3), samples=256)
    point = jnp.array([0.3, -0.4, 0.5, 0.2, -0.1])

    plain = surrogate(point)
    assert jnp.allclose(jax.jit(lambda m, z: m(z))(surrogate, point), plain)
    batch = jnp.stack([point, point])
    assert jax.vmap(surrogate)(batch).shape == (2, 3)
    jacobian = jax.jacobian(surrogate)(point)
    assert jacobian.shape == (3, 5)


def test_weak_coupling_is_dropped_and_bounded() -> None:
    def target(x: jax.Array) -> jax.Array:
        # A small cross term couples out0 to x1, below the tolerance.
        return jnp.array([x[0] + 1e-3 * x[1], x[1]])

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit_separable(target, domain, sx.Affine(), samples=128, tolerance=1e-2)
    # The weak cross term is dropped, so the two outputs split into two blocks.
    assert surrogate.input_indices == ((0,), (1,))
    assert float(surrogate.report.projection_max) > 0.0
