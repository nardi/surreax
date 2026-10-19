"""Polynomial degrees read from the jaxpr, and the fit-settings recommendation."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx

_SQUARE = sx.box([-1.0, -1.0], [1.0, 1.0])


@jax.custom_jvp
def _custom_square(x: jax.Array) -> jax.Array:
    return x**2


@_custom_square.defjvp
def _custom_square_jvp(primals: tuple, tangents: tuple) -> tuple:
    (x,), (dx,) = primals, tangents
    return x**2, 2 * x * dx


def test_degrees_follow_the_polynomial_rules() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.stack(
            [
                jnp.asarray(3.0),
                2.0 * x[0] - x[1] / 4.0,
                x[0] * x[1],
                x[0] ** 3 + x[1],
                jnp.sum(jnp.square(x)),
                (x @ jnp.ones(2)) ** 2,
                jnp.power(x[0], 2.0),
                jax.jit(lambda v: v[0] * v[1] ** 2)(x),
                _custom_square(x[1]),
                x.at[0].set(x[1] ** 2)[0],
                jnp.concatenate([x, x**2])[3],
            ]
        )

    assert sx.polynomial_degrees(target, _SQUARE) == (0, 1, 2, 3, 2, 2, 2, 3, 2, 2, 2)


def test_non_polynomial_operations_read_as_none() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.stack(
            [
                jnp.sin(x[0]),
                x[0] / x[1],
                x[0] ** -1,
                jnp.power(x[0], 0.5),
                jnp.max(x),
                jnp.where(x[0] > 0, x[0], 0.0),
                x[jnp.argmax(x)],
                jnp.sin(jnp.asarray(2.0)) * x[0],
            ]
        )

    degrees = sx.polynomial_degrees(target, _SQUARE)
    assert degrees[:7] == (None,) * 7
    # A transcendental of a constant is a constant, so the last output is affine.
    assert degrees[7] == 1


def test_degrees_read_the_flat_coordinates_of_the_domain() -> None:
    def target(inputs: dict[str, jax.Array]) -> jax.Array:
        return jnp.stack([inputs["x"][0] ** 2 * inputs["count"][0], inputs["rate"][0] ** 2])

    domain = sx.product(
        {
            "x": sx.box([-1.0], [1.0]),
            "count": sx.integer([1], [3]),
            "rate": sx.log_range([0.1], [1.0]),
        }
    )
    # The integer is held constant, and the square of a rate is not a polynomial
    # in its logarithm.
    assert sx.polynomial_degrees(target, domain) == (2, None)


def test_a_cubic_target_is_fitted_exactly_without_a_degree() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (x[0] ** 3 - x[0] * x[1] + 0.5)[None]

    settings = sx.recommend_fit(target, _SQUARE, samples=128)
    assert settings.form == sx.ChebyshevPolynomial(degree=3)
    assert settings.reduce_inputs is None
    assert settings.sparse_terms is None
    surrogate = sx.fit(target, _SQUARE, samples=128, worst_case_starts=0)
    assert float(surrogate.report.value_max) < 1e-4


@pytest.mark.slow
def test_a_smooth_non_polynomial_target_gets_a_searched_degree() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.exp(x[0]) * jnp.cos(x[1])[None]

    settings = sx.recommend_fit(target, _SQUARE, samples=128)
    assert isinstance(settings.form, sx.ChebyshevPolynomial)
    assert settings.form.degree >= 4
    # Asking for a large gain per degree stops the search at the lowest degree.
    strict = sx.recommend_fit(target, _SQUARE, samples=128, improvement=1e6)
    assert strict.form == sx.ChebyshevPolynomial(degree=2)


@pytest.mark.slow
def test_few_directions_and_correlated_outputs_get_reductions() -> None:
    direction = jnp.linspace(0.1, 1.0, 10)
    other = jnp.cos(jnp.arange(10.0))

    def target(x: jax.Array) -> jax.Array:
        a, b = x @ direction / 5.0, x @ other / 5.0
        return jnp.stack([a**2 + b, a * b, 2 * a**2 + 2 * b, a * b - a**2 - b, 3 * a * b, b])

    domain = sx.box([-1.0] * 10, [1.0] * 10)
    settings = sx.recommend_fit(target, domain, samples=256)
    assert settings.reduce_inputs is not None
    assert settings.reduce_outputs is not None
    surrogate = sx.fit(target, domain, samples=256, worst_case_starts=0)
    assert surrogate.input_reduction.encode_matrix.shape == (2, 10)
    assert surrogate.output_reduction.encode_matrix.shape[0] <= 3


@pytest.mark.slow
def test_a_large_basis_gets_a_sparse_budget() -> None:
    # Every input matters about equally, so no input reduction applies, and the
    # degree-three basis over twelve inputs outgrows the samples.
    def target(x: jax.Array) -> jax.Array:
        return (jnp.sum(x**3) + x[0] * x[1])[None]

    domain = sx.box([-1.0] * 12, [1.0] * 12)
    settings = sx.recommend_fit(target, domain, samples=128)
    assert settings.form == sx.ChebyshevPolynomial(degree=3)
    assert settings.sparse_terms is not None
    # An explicit budget overrides the recommended one.
    surrogate = sx.fit(target, domain, samples=128, sparse_terms=20, worst_case_starts=0)
    assert surrogate.report.structure_cost <= 20


def test_a_kinked_target_keeps_its_piecewise_form() -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.abs(x[0] + x[1])[None]

    settings = sx.recommend_fit(target, _SQUARE, samples=128)
    assert isinstance(settings.form, sx.MaxAffineForm)
    assert settings.reduce_inputs is None


@pytest.mark.parametrize("degree", [2, 4])
def test_block_forms_take_the_block_degree(degree: int) -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.stack([x[0] ** degree, 2.0 * x[1]])

    forms = sx.recommend_block_forms(target, _SQUARE, samples=64)
    assert forms == [sx.ChebyshevPolynomial(degree=degree), sx.Affine()]
