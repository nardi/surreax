"""Conversion between a pytree of arrays and a single flat vector.

The pipeline works on flat vectors, so the fitted object flattens the caller's
input and unflattens the model output back into the target's structure. Both
directions come from `jax.flatten_util.ravel_pytree`.
"""

from collections.abc import Callable

from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree

Unraveler = Callable[[Float[Array, " k"]], PyTree[Array]]
"""Rebuild the original pytree from its flat vector."""


def ravel(tree: PyTree[Array]) -> Float[Array, " k"]:
    """Flatten a pytree of arrays into one vector."""
    flat, _ = ravel_pytree(tree)
    return flat


def ravel_example(example: PyTree[Array]) -> tuple[Float[Array, " k"], Unraveler]:
    """Return the flat vector for `example` and the closure that inverts it.

    The unraveler is captured once from a representative example and reused for
    every later flat vector of the same structure.
    """
    return ravel_pytree(example)
