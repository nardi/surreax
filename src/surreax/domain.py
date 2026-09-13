"""Where a surrogate should be accurate.

A domain describes the region of input space to fit over. Its bounds mirror the
structure of the target's input, which can be a plain array or a pytree with
named fields. The domain flattens that structure, maps points from the unit
hypercube into the flat region, and reports flat axis-aligned bounds that the
input normalizer uses to scale coordinates to `[-1, 1]`. It also unravels a
flat point back into the input structure, so a target written against a pytree
input sees that pytree.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from surreax.flatten import ravel, ravel_example


@runtime_checkable
class Domain(Protocol):
    """A region of input space with a unit-hypercube parameterization."""

    @property
    def dimension(self) -> int:
        """Number of flat input coordinates."""
        ...

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        """Map points from `[0, 1]^d` into the flat region."""
        ...

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        """Lower and upper bound per flat coordinate."""
        ...

    def unravel(self, point: Float[Array, " d"]) -> PyTree[Array]:
        """Rebuild the input structure from one flat point."""
        ...


class Box(eqx.Module):
    """An axis-aligned box, given by a lower and an upper corner.

    The corners are pytrees that mirror the input structure, with a plain array
    as the common case.
    """

    lower: PyTree[Array]
    upper: PyTree[Array]

    @property
    def dimension(self) -> int:
        return ravel(self.lower).shape[0]

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        lower, upper = self.bounds()
        return lower + unit_points * (upper - lower)

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        return ravel(self.lower), ravel(self.upper)

    def unravel(self, point: Float[Array, " d"]) -> PyTree[Array]:
        _, rebuild = ravel_example(self.lower)
        return rebuild(point)


def box(
    lower: Sequence[float] | Float[Array, " d"],
    upper: Sequence[float] | Float[Array, " d"],
) -> Box:
    """Build a `Box` from bound sequences, as float arrays."""
    return Box(lower=jnp.asarray(lower, float), upper=jnp.asarray(upper, float))


def box_from_data(points: Float[Array, "n d"]) -> Box:
    """Infer the tightest box that contains a dataset of inputs."""
    return Box(lower=jnp.min(points, axis=0), upper=jnp.max(points, axis=0))
