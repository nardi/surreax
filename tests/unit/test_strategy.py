"""Tests for structure-driven form and method selection."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx

_DOMAIN = sx.box([-1.0, -1.0], [1.0, 1.0])


def _affine(x: jax.Array) -> jax.Array:
    return jnp.array([2 * x[0] - x[1] + 1.0])


def _polynomial(x: jax.Array) -> jax.Array:
    return jnp.array([x[0] ** 2 - 2 * x[0] * x[1]])


def _piecewise_linear(x: jax.Array) -> jax.Array:
    return jnp.array([jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])])


def _discontinuous(x: jax.Array) -> jax.Array:
    return jnp.array([jnp.where(x[0] > 0.0, 1.0, -1.0) + 0.1 * x[1]])


def test_function_kinds_reads_each_kind() -> None:
    assert sx.function_kinds(_affine, _DOMAIN) == ("affine",)
    assert sx.function_kinds(_polynomial, _DOMAIN) == ("polynomial",)
    assert sx.function_kinds(_piecewise_linear, _DOMAIN) == ("piecewise_linear",)
    assert sx.function_kinds(_discontinuous, _DOMAIN) == ("discontinuous",)


def test_recommend_form_maps_kind_to_form() -> None:
    assert isinstance(sx.recommend_form(_affine, _DOMAIN), sx.Affine)
    assert isinstance(sx.recommend_form(_polynomial, _DOMAIN), sx.ChebyshevPolynomial)
    assert isinstance(sx.recommend_form(_piecewise_linear, _DOMAIN), sx.MaxAffineForm)


def test_recommend_form_warns_on_a_discontinuity() -> None:
    with pytest.warns(UserWarning, match="discontinuous"):
        form = sx.recommend_form(_discontinuous, _DOMAIN)
    assert isinstance(form, sx.NetworkForm)


def test_recommend_form_falls_back_to_a_network_for_mixed_kinds() -> None:
    def smooth_and_kinked(x: jax.Array) -> jax.Array:
        # One output is smooth, the other piecewise linear, both on the same
        # inputs, so the block covers both kinds and needs a general form.
        return jnp.array([x[0] ** 2 + x[1] ** 2, jnp.maximum(x[0], x[1])])

    assert isinstance(sx.recommend_form(smooth_and_kinked, _DOMAIN), sx.NetworkForm)


@pytest.mark.slow
def test_auto_fit_recovers_a_polynomial_without_a_form() -> None:
    surrogate = sx.fit(_polynomial, _DOMAIN, samples=256)
    assert float(surrogate.report.value_max) < 1e-3


def test_recommend_block_forms_picks_a_form_per_block() -> None:
    def separable(x: jax.Array) -> jax.Array:
        return jnp.array([3 * x[0] + 1.0, x[1] ** 2])

    forms = sx.recommend_block_forms(separable, _DOMAIN)
    assert len(forms) == 2
    assert isinstance(forms[0], sx.Affine)
    assert isinstance(forms[1], sx.ChebyshevPolynomial)


@pytest.mark.slow
def test_auto_separable_fits_a_linear_and_a_nonlinear_block() -> None:
    def separable(x: jax.Array) -> jax.Array:
        return jnp.array([3 * x[0] + 1.0, x[1] ** 2])

    composite = sx.fit_separable(separable, _DOMAIN, samples=256)
    assert float(composite.report.value_max) < 1e-3


def _affine_residual(x: jax.Array, y: jax.Array) -> jax.Array:
    return y - 2.0 * x


def _cubic_residual(x: jax.Array, y: jax.Array) -> jax.Array:
    return y**3 + y - x


def test_residual_affine_detection() -> None:
    from surreax.strategy import residual_is_affine_in_unknown

    unknown = sx.box([-1.0], [1.0])
    assert residual_is_affine_in_unknown(_affine_residual, sx.box([-1.0], [1.0]), unknown)
    assert not residual_is_affine_in_unknown(_cubic_residual, sx.box([-2.0], [2.0]), unknown)


def test_recommend_method_picks_the_route() -> None:
    small = sx.box([-1.0], [1.0])
    affine_method = sx.recommend_method(_affine_residual, small, sx.box([-3.0], [3.0]))
    assert isinstance(affine_method, sx.InverseMap)
    assert affine_method.newton_steps == 1

    cubic_method = sx.recommend_method(
        _cubic_residual, sx.box([-2.0], [2.0]), sx.box([-1.3], [1.3])
    )
    assert isinstance(cubic_method, sx.InverseMap)
    assert cubic_method.newton_steps > 0

    # A large unknown routes to the reducing exact solve, affine or not.
    large = sx.box([-3.0] * 40, [3.0] * 40)

    def large_affine(x: jax.Array, y: jax.Array) -> jax.Array:
        return y - jnp.concatenate([x, jnp.zeros(39)])

    def large_nonlinear(x: jax.Array, y: jax.Array) -> jax.Array:
        return y**3 + y - jnp.concatenate([x, jnp.zeros(39)])

    assert isinstance(sx.recommend_method(large_affine, small, large), sx.ExactSolve)
    assert isinstance(sx.recommend_method(large_nonlinear, small, large), sx.ExactSolve)


def test_auto_fit_implicit_solves_an_affine_residual() -> None:
    surrogate = sx.fit_implicit(
        _affine_residual, sx.box([-1.0], [1.0]), unknown_domain=sx.box([-3.0], [3.0]), samples=256
    )
    assert float(surrogate.report.value_max) < 1e-3


def test_auto_fit_implicit_needs_an_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown_domain"):
        sx.fit_implicit(_affine_residual, sx.box([-1.0], [1.0]), samples=64)
