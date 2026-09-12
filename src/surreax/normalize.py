"""Per-coordinate affine rescaling.

Fitting is better conditioned when every coordinate sits on a comparable
scale. A `Normalizer` is an invertible affine map, `(x - center) / scale`, fit
either from domain bounds, which sends a box to `[-1, 1]` and suits the
Chebyshev basis, or from sample moments, which centers and whitens outputs.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Normalizer(eqx.Module):
    """An invertible per-coordinate affine map."""

    center: Float[Array, " n"]
    scale: Float[Array, " n"]

    def forward(self, values: Float[Array, " n"]) -> Float[Array, " n"]:
        return (values - self.center) / self.scale

    def inverse(self, values: Float[Array, " n"]) -> Float[Array, " n"]:
        return values * self.scale + self.center


def _guard(scale: Float[Array, " n"]) -> Float[Array, " n"]:
    """Replace a zero scale with one, so a constant coordinate maps to zero."""
    return jnp.where(scale == 0, 1.0, scale)


def from_bounds(lower: Float[Array, " n"], upper: Float[Array, " n"]) -> Normalizer:
    """Map the box `[lower, upper]` onto `[-1, 1]` per coordinate."""
    center = (lower + upper) / 2
    scale = (upper - lower) / 2
    return Normalizer(center=center, scale=_guard(scale))


def from_moments(values: Float[Array, "n d"]) -> Normalizer:
    """Center by the mean and scale by the standard deviation per coordinate."""
    center = jnp.mean(values, axis=0)
    scale = jnp.std(values, axis=0)
    return Normalizer(center=center, scale=_guard(scale))
