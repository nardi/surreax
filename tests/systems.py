"""Physically motivated target functions for the integration tests.

Each target is a pure JAX function of a flat input vector with a hand-derived
form, so a test can compare the surrogate against the true function and its
true Jacobian over a realistic domain.
"""

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


def oscillator_displacement(state: Float[Array, " 3"]) -> Float[Array, " 1"]:
    """Displacement of a damped oscillator at a fixed time from mass, stiffness, damping."""
    mass, stiffness, damping = state
    decay = damping / (2 * mass)
    frequency = jnp.sqrt(stiffness / mass - decay**2)
    time = 1.0
    return (jnp.exp(-decay * time) * jnp.cos(frequency * time))[None]


def oscillator_domain() -> sx.Box:
    """Underdamped region, so the frequency stays real across the box."""
    return sx.box([1.0, 8.0, 0.1], [2.0, 12.0, 1.0])


_FREQUENCIES = jnp.linspace(0.5, 2.0, 6)


def rlc_response(state: Float[Array, " 3"]) -> Float[Array, " 6"]:
    """Magnitude of an RLC low-pass response over a frequency grid."""
    resistance, inductance, capacitance = state
    omega = _FREQUENCIES
    real = 1 - omega**2 * inductance * capacitance
    imaginary = omega * resistance * capacitance
    return 1.0 / jnp.sqrt(real**2 + imaginary**2)


def rlc_domain() -> sx.Box:
    """Component values for the RLC circuit."""
    return sx.box([0.5, 0.5, 0.5], [2.0, 2.0, 2.0])
