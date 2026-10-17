"""Domain leaves beyond the box: log-scaled, normal, integer and categorical inputs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import surreax as sx
from surreax.sampling import sample


def test_log_range_samples_uniformly_in_the_logarithm() -> None:
    region = sx.log_range([1e-3], [1.0])
    flat = sample(region, 512, seed=0)
    values = jax.vmap(region.unravel)(flat)[:, 0]
    assert float(jnp.min(values)) >= 1e-3
    assert float(jnp.max(values)) <= 1.0
    fraction = (np.log(np.asarray(values)) - np.log(1e-3)) / (np.log(1.0) - np.log(1e-3))
    assert stats.kstest(fraction, "uniform").pvalue > 0.05


def test_log_range_rejects_a_nonpositive_bound() -> None:
    with pytest.raises(ValueError, match="positive"):
        sx.log_range([0.0], [1.0])


def test_normal_samples_follow_the_truncated_law() -> None:
    region = sx.normal([2.0], [0.5], truncate=3.0)
    flat = np.asarray(sample(region, 1024, seed=0))[:, 0]
    law = stats.truncnorm(-3.0, 3.0, loc=2.0, scale=0.5)
    assert stats.kstest(flat, law.cdf).pvalue > 0.05
    lower, upper = region.bounds()
    assert float(lower[0]) == pytest.approx(0.5)
    assert float(upper[0]) == pytest.approx(3.5)


def test_integer_samples_every_value_evenly() -> None:
    region = sx.integer([1], [4])
    flat = np.asarray(sample(region, 400, seed=0))[:, 0]
    values, counts = np.unique(flat, return_counts=True)
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert counts.min() >= 80
    assert jnp.array_equal(region.project(jnp.array([2.6])), jnp.array([3.0]))
    assert jnp.array_equal(region.project(jnp.array([9.0])), jnp.array([4.0]))


def test_categorical_samples_every_option_evenly() -> None:
    region = sx.categorical([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    flat = np.asarray(sample(region, 300, seed=0))
    assert np.allclose(flat.sum(axis=1), 1.0)
    counts = flat.sum(axis=0)
    assert counts.min() >= 80
    # A one-hot point unravels to its option, and an option flattens back to it.
    assert jnp.array_equal(region.unravel(jnp.array([0.0, 1.0, 0.0])), jnp.array([0.0, 2.0]))
    assert jnp.array_equal(region.flatten(jnp.array([3.0, 3.0])), jnp.array([0.0, 0.0, 1.0]))
    assert jnp.array_equal(region.project(jnp.array([0.2, 0.5, 0.3])), jnp.array([0.0, 1.0, 0.0]))


def test_product_round_trips_a_mixed_input() -> None:
    region = sx.product(
        {
            "size": sx.box([0.0, 0.0], [1.0, 2.0]),
            "rate": sx.log_range([1e-2], [10.0]),
            "spread": sx.normal([0.0], [1.0]),
            "count": sx.integer([1], [5]),
            "material": sx.categorical([1.0, 4.0]),
        }
    )
    assert region.dimension == 7
    assert region.discrete_mask.tolist() == [True, True, True, False, False, False, False]
    value = {
        "size": jnp.array([0.5, 1.5]),
        "rate": jnp.array([0.1]),
        "spread": jnp.array([0.3]),
        "count": jnp.array([3.0]),
        "material": jnp.array(4.0),
    }
    flat = region.flatten(value)
    rebuilt = region.unravel(flat)
    for name in value:
        assert jnp.allclose(rebuilt[name], value[name], rtol=1e-5)
    # Leaves are in sorted key order, so the count and the material weights lead.
    snapped = region.project(flat.at[0].set(2.7).at[1].set(0.6).at[2].set(0.4))
    assert jnp.array_equal(snapped[:3], jnp.array([3.0, 1.0, 0.0]))
    points = sample(region, 64, seed=1)
    assert jnp.array_equal(jax.vmap(region.project)(points), points)


def test_log_range_fits_a_log_scaled_target_more_accurately() -> None:
    def target(rate: jax.Array) -> jax.Array:
        return jnp.log(rate)

    def error(domain: sx.Domain) -> float:
        surrogate = sx.fit(target, domain, sx.Affine(), samples=64, worst_case_starts=0)
        return float(surrogate.report.value_max)

    assert error(sx.log_range([1e-3], [1.0])) < 1e-3 * error(sx.box([1e-3], [1.0]))


def _mixed_domain() -> sx.Product:
    return sx.product(
        {
            "x": sx.box([-1.0, -1.0], [1.0, 1.0]),
            "count": sx.integer([1], [3]),
            "material": sx.categorical([0.5, 2.0]),
        }
    )


def _mixed_target(inputs: dict[str, jax.Array]) -> jax.Array:
    x, count, material = inputs["x"], inputs["count"][0], inputs["material"]
    return jnp.stack([material * x[0] + count, x[0] * x[1] + count * material])


def test_a_mixed_domain_fits_and_reports_a_valid_worst_case() -> None:
    domain = _mixed_domain()
    surrogate = sx.fit(
        _mixed_target, domain, sx.ChebyshevPolynomial(degree=3), samples=256, worst_case_starts=4
    )
    assert float(surrogate.report.value_max) < 1e-3
    point = surrogate.report.worst_case_point
    assert jnp.array_equal(domain.project(point), point)
    value = {"x": jnp.array([0.2, -0.4]), "count": jnp.array([2.0]), "material": jnp.array(2.0)}
    assert jnp.allclose(surrogate(value), _mixed_target(value), atol=1e-3)
    assert jnp.allclose(eqx.filter_jit(surrogate)(value), surrogate(value))
    batch = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]), value)
    assert jax.vmap(surrogate)(batch).shape == (2, 2)
    jacobian = jax.jacobian(surrogate)(value)
    assert jnp.allclose(jacobian["x"], jax.jacobian(_mixed_target)(value)["x"], atol=1e-2)


def test_a_mixed_domain_supports_derivatives_and_reductions() -> None:
    domain = _mixed_domain()
    sobolev = sx.fit(
        _mixed_target,
        domain,
        sx.ChebyshevPolynomial(degree=3),
        samples=256,
        derivative_weight=1.0,
        worst_case_starts=0,
    )
    assert float(sobolev.report.derivative_max) < 1e-2
    # The reduction acts on the two continuous coordinates and keeps all five.
    reduced = sx.fit(
        _mixed_target,
        domain,
        sx.ChebyshevPolynomial(degree=3),
        samples=256,
        reduce_inputs=sx.active_subspace_inputs(rank=2),
        derivative_weight=1.0,
        worst_case_starts=0,
    )
    assert reduced.input_reduction.encode_matrix.shape == (5, 5)
    assert float(reduced.report.value_max) < 1e-2
    network = sx.fit(
        _mixed_target,
        domain,
        sx.NetworkForm(width=16, depth=1, steps=50),
        samples=64,
        derivative_weight=0.1,
        worst_case_starts=0,
    )
    assert jnp.isfinite(network.report.value_max)


def test_analysis_keeps_discrete_inputs_coupled() -> None:
    def target(inputs: dict[str, jax.Array]) -> jax.Array:
        return jnp.stack([inputs["x"][0] * inputs["count"][0], inputs["x"][1] ** 2])

    domain = sx.product({"x": sx.box([-1.0, -1.0], [1.0, 1.0]), "count": sx.integer([1], [3])})
    structure = sx.analyze(target, domain, samples=64)
    # The integer reads as coupled to both outputs, so nothing is split off.
    assert len(set(np.asarray(structure.output_block).tolist())) == 1
    assert sx.function_kinds(target, domain, samples=64) == ("affine", "polynomial")
    with pytest.raises(ValueError, match="continuous"):
        sx.function_kinds(lambda n: n, sx.integer([1], [3]), samples=16)


def test_flat_input_entry_points_need_a_box() -> None:
    domain = sx.product({"rate": sx.log_range([0.1], [1.0])})
    with pytest.raises(TypeError, match="Box"):
        sx.fit_separable(lambda inputs: inputs["rate"], domain, sx.Affine())
    method = sx.InverseMap(form=sx.Affine(), unknown_domain=sx.box([-1.0], [1.0]))
    with pytest.raises(TypeError, match="Box"):
        sx.fit_implicit(lambda x, y: y - x, domain, method)


@pytest.mark.parametrize("sampler", ["sobol", "halton", "latin_hypercube"])
def test_every_sampler_fits_a_smooth_target(sampler: sx.Sampler) -> None:
    def target(x: jax.Array) -> jax.Array:
        return jnp.sin(x[0]) * jnp.cos(x[1])

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(
        target, domain, sx.ChebyshevPolynomial(degree=6), samples=256, sampler=sampler,
        worst_case_starts=0,
    )
    assert float(surrogate.report.value_max) < 2e-3
