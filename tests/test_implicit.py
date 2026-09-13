"""Tests for implicit fitting through the inverse map."""

import jax
import jax.numpy as jnp

import surreax as sx


def test_affine_in_the_unknown_is_recovered_exactly() -> None:
    # g(x, y) = y - 2x, so f(x) = 2x, and the inverse map is exactly affine.
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return y - 2.0 * x

    domain = sx.box([-1.0], [1.0])
    unknown_domain = sx.box([-3.0], [3.0])
    surrogate = sx.fit_implicit(residual, domain, unknown_domain, sx.Affine(), samples=256)
    assert float(surrogate.report.value_max) < 1e-4
    prediction = surrogate(jnp.array([0.5]))
    assert jnp.allclose(prediction, jnp.array([1.0]), atol=1e-3)


def test_coupled_affine_system_is_recovered_exactly() -> None:
    matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])

    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return matrix @ y - x

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    unknown_domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit_implicit(residual, domain, unknown_domain, sx.Affine(), samples=256)
    assert float(surrogate.report.value_max) < 1e-4
    solved = jnp.linalg.solve(matrix, jnp.array([0.4, -0.2]))
    assert jnp.allclose(surrogate(jnp.array([0.4, -0.2])), solved, atol=1e-3)


def _cubic(x: jax.Array, y: jax.Array) -> jax.Array:
    return y**3 + y - x


def test_newton_steps_lower_the_residual() -> None:
    domain = sx.box([-2.0], [2.0])
    unknown_domain = sx.box([-1.2], [1.2])
    form = sx.ChebyshevPolynomial(degree=4)
    without = sx.fit_implicit(_cubic, domain, unknown_domain, form, samples=512, newton_steps=0)
    with_steps = sx.fit_implicit(_cubic, domain, unknown_domain, form, samples=512, newton_steps=2)
    assert float(with_steps.report.value_mean) < float(without.report.value_mean)


def test_nonlinear_monotone_reaches_a_low_residual() -> None:
    domain = sx.box([-2.0], [2.0])
    unknown_domain = sx.box([-1.2], [1.2])
    form = sx.ChebyshevPolynomial(degree=4)
    surrogate = sx.fit_implicit(_cubic, domain, unknown_domain, form, samples=512, newton_steps=2)
    assert float(surrogate.report.value_mean) < 0.5


def test_implicit_surrogate_survives_jit_vmap_and_grad() -> None:
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return y - 2.0 * x

    domain = sx.box([-1.0], [1.0])
    unknown_domain = sx.box([-3.0], [3.0])
    surrogate = sx.fit_implicit(residual, domain, unknown_domain, sx.Affine(), samples=256)
    point = jnp.array([0.3])

    plain = surrogate(point)
    assert jnp.allclose(jax.jit(lambda m, z: m(z))(surrogate, point), plain)
    batched = jax.vmap(surrogate)(jnp.array([[0.1], [0.3]]))
    assert batched.shape == (2, 1)
    gradient = jax.grad(lambda z: surrogate(z)[0])(point)
    assert gradient.shape == (1,)
