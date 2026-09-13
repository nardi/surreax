"""End-to-end tests for `fit`: recovery, integration, and transforms."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import surreax as sx
from tests import systems


@given(
    weights=st.lists(st.floats(-3, 3, allow_nan=False), min_size=3, max_size=3),
    bias=st.floats(-3, 3, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_affine_form_recovers_an_affine_target(weights: list[float], bias: float) -> None:
    coefficients = jnp.array(weights)

    def target(x: jax.Array) -> jax.Array:
        return (coefficients @ x + bias)[None]

    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.Affine(), samples=128, seed=0)
    assert float(surrogate.report.value_max) < 1e-3


def test_polynomial_form_recovers_a_low_degree_polynomial() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (x[0] ** 2 - 2 * x[0] * x[1] + 0.5 * x[1] ** 3)[None]

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=3), samples=256, seed=0)
    assert float(surrogate.report.value_max) < 1e-3
    assert float(surrogate.report.derivative_max) < 1e-2


@pytest.mark.parametrize(
    ("target", "domain", "form", "tolerance"),
    [
        (
            systems.projectile_range,
            systems.projectile_domain(),
            sx.ChebyshevPolynomial(degree=5),
            0.5,
        ),
        (
            systems.oscillator_displacement,
            systems.oscillator_domain(),
            sx.ChebyshevPolynomial(degree=6),
            0.05,
        ),
        (systems.rlc_response, systems.rlc_domain(), sx.ChebyshevPolynomial(degree=5), 0.05),
    ],
)
def test_integration_on_physical_systems(target, domain, form, tolerance: float) -> None:
    surrogate = sx.fit(target, domain, form, samples=512, seed=0)
    assert float(surrogate.report.value_mean) < tolerance
    assert jnp.isfinite(surrogate.report.derivative_mean)


def _fitted_projectile() -> sx.FittedSurrogate:
    return sx.fit(
        systems.projectile_range,
        systems.projectile_domain(),
        sx.ChebyshevPolynomial(degree=4),
        samples=256,
        seed=0,
    )


def test_surrogate_agrees_across_jit_vmap_and_derivatives() -> None:
    surrogate = _fitted_projectile()
    point = jnp.array([12.0, 0.7])
    batch = jnp.array([[8.0, 0.4], [15.0, 1.0], [12.0, 0.7]])

    plain = surrogate(point)
    assert jnp.allclose(eqx.filter_jit(surrogate)(point), plain)
    assert jnp.allclose(jax.jit(lambda m, x: m(x))(surrogate, point), plain)

    batched = jax.vmap(surrogate)(batch)
    assert batched.shape == (3, 1)
    assert jnp.allclose(batched[-1], plain)

    jacobian = jax.jacobian(surrogate)(point)
    assert jacobian.shape == (1, 2)
    gradient = jax.grad(lambda x: surrogate(x)[0])(point)
    assert jnp.allclose(gradient, jacobian[0])


def test_fitting_core_runs_under_jit() -> None:
    surrogate = _fitted_projectile()
    inputs = jnp.array([[10.0, 0.5], [14.0, 0.9]])

    @jax.jit
    def predict(model: sx.FittedSurrogate, x: jax.Array) -> jax.Array:
        return jax.vmap(model)(x)

    truth = jax.vmap(systems.projectile_range)(inputs)
    assert jnp.allclose(predict(surrogate, inputs), truth, atol=0.5)


def test_dataset_domain_fits_over_inferred_bounds() -> None:
    points = jax.random.uniform(jax.random.key(1), (64, 2), minval=-1.0, maxval=1.0)
    domain = sx.box_from_data(points)

    def target(x: jax.Array) -> jax.Array:
        return (x[0] + x[1])[None]

    surrogate = sx.fit(target, domain, sx.Affine(), samples=128, seed=0)
    assert float(surrogate.report.value_max) < 1e-3


def test_box_unravels_a_pytree_bound_round_trip() -> None:
    domain = systems.oscillator_domain()
    lower, _ = domain.bounds()
    restored = domain.unravel(lower)
    assert isinstance(restored, systems.OscillatorState)
    assert jnp.allclose(restored.mass, 1.0)
    assert jnp.allclose(restored.stiffness, 8.0)
    assert jnp.allclose(restored.damping, 0.1)


def test_pytree_input_surrogate_agrees_across_jit_vmap_and_derivatives() -> None:
    domain = systems.oscillator_domain()
    surrogate = sx.fit(
        systems.oscillator_displacement,
        domain,
        sx.ChebyshevPolynomial(degree=4),
        samples=256,
        seed=0,
    )
    state = systems.OscillatorState(
        mass=jnp.asarray(1.5), stiffness=jnp.asarray(10.0), damping=jnp.asarray(0.5)
    )

    plain = surrogate(state)
    assert jnp.allclose(eqx.filter_jit(surrogate)(state), plain)

    batch = jax.vmap(domain.unravel)(jnp.array([[1.2, 9.0, 0.3], [1.8, 11.0, 0.7]]))
    batched = jax.vmap(surrogate)(batch)
    assert batched.shape == (2, 1)

    # Grad flows back into the pytree input, one cotangent per field.
    gradient = jax.grad(lambda s: surrogate(s)[0])(state)
    assert jnp.isfinite(gradient.mass)
    assert jnp.isfinite(gradient.stiffness)
    assert jnp.isfinite(gradient.damping)
