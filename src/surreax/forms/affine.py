"""The affine form.

The feature map is the input with a constant one prepended, so the fitted
weights hold a bias column and a linear part. An affine target is recovered
exactly, which makes this a baseline and a sanity check.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Affine(eqx.Module):
    """A form whose features are the input augmented with a constant."""

    def features(self, x: Float[Array, " d"]) -> Float[Array, " f"]:
        return jnp.concatenate([jnp.ones((1,), x.dtype), x])
