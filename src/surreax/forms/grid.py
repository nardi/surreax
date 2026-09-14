"""The piecewise linear form on a regular grid.

The form places a regular grid over the input box and interpolates multilinearly
between the nodes. It is linear in the node values, so it is fitted by one linear
solve, and it is attractive after a reduction, when the reduced input is small
enough for a grid to be practical, since the node count grows as the resolution
to the power of the dimension.

It is a `LinearForm`: the feature vector is the multilinear interpolation weight
of each node, of which only the two to the power of the dimension around the
query are nonzero, and the fitted weights are the node values.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _axis_weights(coordinate: Float[Array, ""], nodes: int) -> Float[Array, " nodes"]:
    """Linear interpolation weights along one axis, for a coordinate in [-1, 1]."""
    position = (coordinate + 1) / 2 * (nodes - 1)
    lower = jnp.clip(jnp.floor(position).astype(int), 0, nodes - 2)
    fraction = position - lower
    weights = jnp.zeros(nodes, dtype=position.dtype)
    weights = weights.at[lower].set(1 - fraction)
    return weights.at[lower + 1].set(fraction)


class Grid(eqx.Module):
    """A regular-grid multilinear form with `resolution` nodes per input axis."""

    resolution: int = eqx.field(static=True)

    def features(self, x: Float[Array, " d"]) -> Float[Array, " f"]:
        weights = _axis_weights(x[0], self.resolution)
        for axis in range(1, x.shape[0]):
            weights = jnp.outer(weights, _axis_weights(x[axis], self.resolution)).reshape(-1)
        return weights
