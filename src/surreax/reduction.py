"""Linear dimension reduction on the input and output sides.

A `LinearReduction` is an affine encode paired with an affine decode. Composing
a linear map with a piecewise linear or convex model keeps that structure in
the original coordinates, which is why the reduction is linear. The first
version ships only the identity reduction, so PCA and active subspaces can slot
in later without changing the pipeline.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearReduction(eqx.Module):
    """An affine encode into a reduced space and an affine decode out of it."""

    encode_matrix: Float[Array, "r n"]
    encode_bias: Float[Array, " r"]
    decode_matrix: Float[Array, "n r"]
    decode_bias: Float[Array, " n"]

    def forward(self, values: Float[Array, " n"]) -> Float[Array, " r"]:
        return self.encode_matrix @ values + self.encode_bias

    def inverse(self, reduced: Float[Array, " r"]) -> Float[Array, " n"]:
        return self.decode_matrix @ reduced + self.decode_bias


def identity(dimension: int) -> LinearReduction:
    """A reduction that passes its input through unchanged."""
    eye = jnp.eye(dimension)
    zeros = jnp.zeros(dimension)
    return LinearReduction(
        encode_matrix=eye,
        encode_bias=zeros,
        decode_matrix=eye,
        decode_bias=zeros,
    )
