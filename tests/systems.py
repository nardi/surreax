"""Physically motivated target functions for the integration tests.

Each target is a pure JAX function of an input with a hand-derived form, so a
test can compare the surrogate against the true function and its true Jacobian
over a realistic domain. The projectile takes a flat array, while the
oscillator and the RLC circuit take an `eqx.Module` state with named fields, so
the tests exercise both a flat and a structured input.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

import surreax as sx

GRAVITY = 9.81


def projectile_range(state: Float[Array, " 2"]) -> Float[Array, " 1"]:
    """Horizontal range of a projectile from launch speed and angle."""
    speed, angle = state
    return (speed**2 * jnp.sin(2 * angle) / GRAVITY)[None]


def projectile_domain() -> sx.Box:
    """Launch speeds in m/s and angles in radians."""
    return sx.box([5.0, 0.2], [20.0, 1.3])


class OscillatorState(eqx.Module):
    """Parameters of a damped harmonic oscillator."""

    mass: Float[Array, ""]
    stiffness: Float[Array, ""]
    damping: Float[Array, ""]


def _scalar(value: float) -> Float[Array, ""]:
    return jnp.asarray(value, float)


def oscillator_displacement(state: OscillatorState) -> Float[Array, " 1"]:
    """Displacement of a damped oscillator at a fixed time."""
    decay = state.damping / (2 * state.mass)
    frequency = jnp.sqrt(state.stiffness / state.mass - decay**2)
    time = 1.0
    return (jnp.exp(-decay * time) * jnp.cos(frequency * time))[None]


def oscillator_domain() -> sx.Box:
    """Underdamped region, so the frequency stays real across the box."""
    lower = OscillatorState(mass=_scalar(1.0), stiffness=_scalar(8.0), damping=_scalar(0.1))
    upper = OscillatorState(mass=_scalar(2.0), stiffness=_scalar(12.0), damping=_scalar(1.0))
    return sx.Box(lower=lower, upper=upper)


class RlcComponents(eqx.Module):
    """Component values of an RLC circuit."""

    resistance: Float[Array, ""]
    inductance: Float[Array, ""]
    capacitance: Float[Array, ""]


_FREQUENCIES = jnp.linspace(0.5, 2.0, 6)


def rlc_response(components: RlcComponents) -> Float[Array, " 6"]:
    """Magnitude of an RLC low-pass response over a frequency grid."""
    omega = _FREQUENCIES
    real = 1 - omega**2 * components.inductance * components.capacitance
    imaginary = omega * components.resistance * components.capacitance
    return 1.0 / jnp.sqrt(real**2 + imaginary**2)


def rlc_domain() -> sx.Box:
    """Component values for the RLC circuit."""
    lower = RlcComponents(
        resistance=_scalar(0.5), inductance=_scalar(0.5), capacitance=_scalar(0.5)
    )
    upper = RlcComponents(
        resistance=_scalar(2.0), inductance=_scalar(2.0), capacitance=_scalar(2.0)
    )
    return sx.Box(lower=lower, upper=upper)


HIGH_DIMENSION = 12


def few_active_inputs(x: Float[Array, " 12"]) -> Float[Array, " 1"]:
    """A high-dimensional target that depends on only two of its inputs."""
    return (x[2] ** 2 + 1.5 * x[7] - x[2] * x[7])[None]


def high_dimensional_domain() -> sx.Box:
    """A box over many inputs, most of which the target ignores."""
    return sx.box([-1.0] * HIGH_DIMENSION, [1.0] * HIGH_DIMENSION)
