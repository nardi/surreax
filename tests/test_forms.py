"""Tests for the model forms and their feature maps."""

import jax.numpy as jnp
import pytest

from surreax.forms import Affine, ChebyshevPolynomial, LinearForm
from surreax.forms.polynomial import _chebyshev_table, _exponents


def test_affine_features_prepend_a_constant() -> None:
    features = Affine().features(jnp.array([2.0, -3.0]))
    assert jnp.array_equal(features, jnp.array([1.0, 2.0, -3.0]))


def test_affine_satisfies_the_linear_form_protocol() -> None:
    assert isinstance(Affine(), LinearForm)
    assert isinstance(ChebyshevPolynomial(degree=2), LinearForm)


@pytest.mark.parametrize(
    ("dimension", "degree", "expected"),
    [(1, 0, 1), (1, 3, 4), (2, 2, 6), (3, 2, 10)],
)
def test_exponent_count_matches_the_total_degree_basis(
    dimension: int, degree: int, expected: int
) -> None:
    assert len(_exponents(dimension, degree)) == expected


@pytest.mark.parametrize("degree", [0, 1, 4])
def test_chebyshev_table_follows_the_recurrence(degree: int) -> None:
    x = jnp.array([0.5, -0.25])
    table = _chebyshev_table(x, degree)
    assert table.shape == (2, degree + 1)
    assert jnp.allclose(table[:, 0], 1.0)
    if degree >= 1:
        assert jnp.allclose(table[:, 1], x)
    if degree >= 2:
        assert jnp.allclose(table[:, 2], 2 * x**2 - 1)


def test_chebyshev_features_are_one_for_a_zero_degree_form() -> None:
    features = ChebyshevPolynomial(degree=0).features(jnp.array([0.3, 0.7]))
    assert jnp.array_equal(features, jnp.array([1.0]))


def test_chebyshev_feature_count_matches_the_basis_size() -> None:
    form = ChebyshevPolynomial(degree=3)
    features = form.features(jnp.array([0.1, -0.2]))
    assert features.shape == (len(_exponents(2, 3)),)
