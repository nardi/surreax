"""Where a surrogate should be accurate.

A domain describes the region of flat input space to fit over. It maps points
from the unit hypercube into that region, which is how the low-discrepancy
samplers stay generic, and it reports axis-aligned bounds, which the input
normalizer uses to scale coordinates to `[-1, 1]`.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


@runtime_checkable
class Domain(Protocol):
    """A region of flat input space with a unit-hypercube parameterization."""

    @property
    def dimension(self) -> int:
        """Number of input coordinates."""
        ...

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        """Map points from `[0, 1]^d` into the domain."""
        ...

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        """Lower and upper bound per coordinate."""
        ...


class Box(eqx.Module):
    """An axis-aligned box, given by a lower and an upper corner."""

    lower: Float[Array, " d"]
    upper: Float[Array, " d"]

    @property
    def dimension(self) -> int:
        return self.lower.shape[0]

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        return self.lower + unit_points * (self.upper - self.lower)

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        return self.lower, self.upper


def box(
    lower: Sequence[float] | Float[Array, " d"],
    upper: Sequence[float] | Float[Array, " d"],
) -> Box:
    """Build a `Box` from bound sequences, as float arrays."""
    return Box(lower=jnp.asarray(lower, float), upper=jnp.asarray(upper, float))


def box_from_data(points: Float[Array, "n d"]) -> Box:
    """Infer the tightest box that contains a dataset of inputs."""
    return Box(lower=jnp.min(points, axis=0), upper=jnp.max(points, axis=0))
