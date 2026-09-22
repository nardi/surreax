"""Residuals written entry by entry over a fixed stencil.

A sparse residual couples each entry to only a few others, the neighbors in a
grid or the nodes wired to it in a network. Writing it as one entry function and
a table of neighbor indices makes that sparsity explicit. The residual is still
an ordinary callable `g(x, y)`, so every fitting method accepts it, and it can
also be evaluated at a few chosen entries while reading only their neighbors,
which is what hyper-reduction needs to make a reduced solve independent of the
state size.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

Entry = Callable[[Float[Array, " n"], Int[Array, ""], Float[Array, " w"]], Float[Array, ""]]
"""One residual entry from the input, the entry index, and its neighbor values."""


@dataclass(frozen=True, eq=False)
class StencilResidual:
    """A residual whose entry `i` reads the unknown only at `stencil[i]`.

    `entry(x, i, y[stencil[i]])` returns entry `i`. A row of `stencil` shorter
    than the widest one pads with any valid index, and the entry function gives
    the padded slots no weight. It compares and hashes by identity, like a plain
    function, so a solver can hold it as static data.
    """

    stencil: Int[Array, "m w"]
    """Neighbor indices of each entry, one row per entry."""

    entry: Entry

    def __call__(self, x: Float[Array, " n"], y: Float[Array, " m"]) -> Float[Array, " m"]:
        return self.entries(x, jnp.arange(self.stencil.shape[0]), y[self.stencil])

    def entries(
        self,
        x: Float[Array, " n"],
        rows: Int[Array, " k"],
        neighbors: Float[Array, "k w"],
    ) -> Float[Array, " k"]:
        """Evaluate the entries `rows` from the unknown at their stencils only."""
        return jax.vmap(lambda row, values: self.entry(x, row, values))(rows, neighbors)
