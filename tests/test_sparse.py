"""Tests for sparse basis selection by matching pursuit."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.fitting import fit_linear_sparse
from tests import systems


def test_matching_pursuit_recovers_the_active_support() -> None:
    inputs = jax.random.uniform(jax.random.key(0), (128, 10), minval=-1.0, maxval=1.0)
    targets = (2.0 * inputs[:, 0] - 3.0 * inputs[:, 5])[:, None]
    model = fit_linear_sparse(sx.Affine(), inputs, targets, terms=3, ridge=1e-8)
    active = jnp.where(jnp.any(model.weights != 0, axis=0))[0]
    # Affine features are [1, x0, ..., x9], so x0 is column 1 and x5 is column 6.
    assert set(int(index) for index in active) <= {0, 1, 6}
    predicted = jax.vmap(model)(inputs)
    assert jnp.allclose(predicted, targets, atol=1e-4)


def test_sparse_fit_is_accurate_with_a_small_structure_cost() -> None:
    surrogate = sx.fit(
        systems.few_active_inputs,
        systems.high_dimensional_domain(),
        sx.ChebyshevPolynomial(degree=2),
        samples=200,
        sparse_terms=6,
        worst_case_starts=0,
        seed=0,
    )
    assert float(surrogate.report.value_mean) < 1e-3
    assert surrogate.report.structure_cost <= 6


def test_sparse_beats_dense_when_samples_are_fewer_than_the_basis() -> None:
    domain = systems.high_dimensional_domain()
    form = sx.ChebyshevPolynomial(degree=3)
    # The degree-three basis over twelve inputs has far more terms than samples.
    dense = sx.fit(
        systems.few_active_inputs,
        domain,
        form,
        samples=120,
        ridge=1e-6,
        worst_case_starts=0,
        seed=0,
    )
    sparse = sx.fit(
        systems.few_active_inputs,
        domain,
        form,
        samples=120,
        sparse_terms=8,
        ridge=1e-6,
        worst_case_starts=0,
        seed=0,
    )
    assert float(sparse.report.value_mean) < float(dense.report.value_mean)
    assert surrogate_is_small(sparse)


def surrogate_is_small(surrogate: sx.FittedSurrogate) -> bool:
    return surrogate.report.structure_cost <= 8
