"""The ReLU network forms against max-affine and difference of max-affine.

All four forms represent piecewise linear functions, the convex ones and the
general ones as sets, so the comparison is about how many parameters a fit needs.
The structured targets in six inputs are sums of ridge functions, which a ReLU
network writes with a few units and a flat set of affine pieces writes with
exponentially many, so they are where the networks should win. The implicit
systems are smooth and low dimensional, where the older forms should hold up.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from tests.piecewise_cases import Case, build

_DOMAIN = sx.box([-1.0] * 6, [1.0] * 6)


def _l1_norm(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.abs(x))[None]


def _nonconvex(x: jax.Array) -> jax.Array:
    return (jnp.abs(x[0] - x[1]) - jnp.abs(x[2] + 0.3 * x[3]) + 0.5 * x[4])[None]


def _parameters(surrogate: sx.FittedSurrogate) -> int:
    leaves = jax.tree.leaves(eqx.filter(surrogate.model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


def _fit(form: sx.Form, target) -> sx.FittedSurrogate:
    return sx.fit(target, _DOMAIN, form, samples=2048, worst_case_starts=0)


def test_networks_need_far_fewer_parameters_for_a_sum_of_ridge_functions() -> None:
    # The 1-norm in six inputs has 64 affine pieces as a maximum, so 48 pieces
    # cannot represent it, while a network writes it with about twelve units.
    flat = _fit(sx.MaxAffineForm(pieces=48, restarts=4, iterations=25), _l1_norm)
    convex = _fit(sx.ConvexNetworkForm(width=12, depth=2), _l1_norm)
    general = _fit(sx.ReluNetworkForm(width=12, depth=2), _l1_norm)
    assert _parameters(convex) <= _parameters(flat)
    assert _parameters(general) < _parameters(flat)
    assert float(convex.report.value_mean) < float(flat.report.value_mean) / 2
    assert float(general.report.value_mean) < float(flat.report.value_mean) / 20


def test_a_relu_network_beats_a_difference_of_max_affine_on_a_nonconvex_ridge_sum() -> None:
    flat = _fit(sx.DifferenceOfMaxAffineForm(pieces=24, restarts=4, steps=400), _nonconvex)
    general = _fit(sx.ReluNetworkForm(width=12, depth=2), _nonconvex)
    assert _parameters(general) < _parameters(flat)
    assert float(general.report.value_mean) < float(flat.report.value_mean) / 50


@pytest.fixture(scope="module", params=["rod", "network", "thermostat"])
def case(request: pytest.FixtureRequest) -> tuple[str, Case]:
    return request.param, build(request.param)


def test_a_relu_network_approximates_each_system_far_better_than_an_affine_fit(
    case: tuple[str, Case],
) -> None:
    _, system = case
    affine = system.latent_error(system.fit(sx.Affine()))
    general = system.latent_error(system.fit(sx.ReluNetworkForm(width=16, depth=2)))
    assert general < affine / 3


def test_a_convex_network_cannot_follow_a_response_that_is_not_convex(
    case: tuple[str, Case],
) -> None:
    name, system = case
    if name == "network":
        pytest.skip("the diode network is only mildly nonconvex, so a convex fit nearly keeps up")
    convex = system.latent_error(system.fit(sx.ConvexNetworkForm(width=16, depth=2)))
    general = system.latent_error(system.fit(sx.ReluNetworkForm(width=16, depth=2)))
    assert convex > 3 * general
