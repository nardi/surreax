"""Tests for the feedforward network form."""

import equinox as eqx
import jax
import jax.numpy as jnp

import surreax as sx
from surreax.forms.network import parameter_count


def _smooth(x: jax.Array) -> jax.Array:
    return (jnp.sin(2 * x[0]) + x[1] ** 2)[None]


def test_network_reaches_a_low_error_on_a_smooth_target() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.NetworkForm(width=32, depth=2, steps=800, learning_rate=0.01)
    surrogate = sx.fit(_smooth, domain, form, samples=256, worst_case_starts=0, seed=0)
    assert float(surrogate.report.value_mean) < 0.02
    assert isinstance(surrogate.model, sx.Network)
    assert surrogate.report.structure_cost == parameter_count(surrogate.model)
    assert surrogate.report.structure_cost > 0


def test_derivative_supervision_lowers_the_network_derivative_error() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.NetworkForm(width=32, depth=2, steps=800, learning_rate=0.01)
    value_only = sx.fit(_smooth, domain, form, samples=256, worst_case_starts=0, seed=0)
    sobolev = sx.fit(
        _smooth, domain, form, samples=256, worst_case_starts=0, derivative_weight=1.0, seed=0
    )
    assert float(sobolev.report.derivative_max) < float(value_only.report.derivative_max)


def test_network_surrogate_agrees_across_jit_vmap_and_grad() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    form = sx.NetworkForm(width=16, depth=2, steps=400)
    surrogate = sx.fit(_smooth, domain, form, samples=256, worst_case_starts=0, seed=0)
    point = jnp.array([0.3, -0.4])

    plain = surrogate(point)
    # The network holds a function leaf, so it jits through filter_jit or a closure.
    assert jnp.allclose(eqx.filter_jit(surrogate)(point), plain)
    assert jnp.allclose(jax.jit(lambda z: surrogate(z))(point), plain)

    batched = jax.vmap(surrogate)(jnp.stack([point, point]))
    assert batched.shape == (2, 1)
    gradient = jax.grad(lambda z: surrogate(z)[0])(point)
    assert gradient.shape == (2,)
