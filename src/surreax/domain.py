"""Where a surrogate should be accurate.

A domain describes the region of input space to fit over. Its bounds mirror the
structure of the target's input, which can be a plain array or a pytree with
named fields. The domain flattens that structure, maps points from the unit
hypercube into the flat region, and reports flat axis-aligned bounds that the
input normalizer uses to scale coordinates to `[-1, 1]`. It also unravels a
flat point back into the input structure, so a target written against a pytree
input sees that pytree.

A flat coordinate need not be the input value itself. A `LogRange` keeps the
logarithm, so the model works on the scale where the target varies evenly, and a
`Categorical` keeps one weight per option. `flatten` maps an input to its flat
coordinates and `unravel` maps them back. Integer and categorical coordinates
are discrete, which `discrete_mask` marks and `project` snaps to valid values.
A `Product` arranges leaves of different kinds in a pytree that mirrors the
target's input.
"""

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, Float, PyTree

from surreax.flatten import ravel, ravel_example


@runtime_checkable
class Domain(Protocol):
    """A region of input space with a unit-hypercube parameterization."""

    @property
    def dimension(self) -> int:
        """Number of flat input coordinates."""
        ...

    @property
    def discrete_mask(self) -> np.ndarray:
        """Which flat coordinates are discrete, one boolean per coordinate."""
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

    def flatten(self, value: PyTree[Array]) -> Float[Array, " d"]:
        """Map an input in the target's structure to its flat coordinates."""
        ...

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        """Snap the discrete coordinates of a flat point to valid values."""
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

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=bool)

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        lower, upper = self.bounds()
        return lower + unit_points * (upper - lower)

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        return ravel(self.lower), ravel(self.upper)

    def unravel(self, point: Float[Array, " d"]) -> PyTree[Array]:
        _, rebuild = ravel_example(self.lower)
        return rebuild(point)

    def flatten(self, value: PyTree[Array]) -> Float[Array, " d"]:
        return ravel(value)

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        return point


class LogRange(eqx.Module):
    """A positive range sampled uniformly in the logarithm.

    Suits an input that spans orders of magnitude, such as a rate or a
    concentration. The flat coordinate is the natural logarithm of the value.
    """

    lower: Float[Array, "..."]
    upper: Float[Array, "..."]

    @property
    def dimension(self) -> int:
        return math.prod(jnp.shape(self.lower))

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=bool)

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        lower, upper = self.bounds()
        return lower + unit_points * (upper - lower)

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        return jnp.log(ravel(self.lower)), jnp.log(ravel(self.upper))

    def unravel(self, point: Float[Array, " d"]) -> Float[Array, "..."]:
        return jnp.exp(point).reshape(jnp.shape(self.lower))

    def flatten(self, value: Float[Array, "..."]) -> Float[Array, " d"]:
        return jnp.log(ravel(value))

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        return point


class Normal(eqx.Module):
    """An input drawn from a normal distribution, truncated to a finite box.

    The normalizer and the worst-case search need bounds, so the distribution is
    cut at `truncate` standard deviations either side of the mean, and the
    samples follow the truncated law through the inverse normal CDF.
    """

    mean: Float[Array, "..."]
    std: Float[Array, "..."]
    truncate: float = eqx.field(static=True, default=4.0)

    @property
    def dimension(self) -> int:
        return math.prod(jnp.shape(self.mean))

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=bool)

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        # Rescale into the CDF range the truncation keeps, then invert the CDF.
        low, high = ndtr(-self.truncate), ndtr(self.truncate)
        standard = ndtri(low + unit_points * (high - low))
        return ravel(self.mean) + ravel(self.std) * standard

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        mean, std = ravel(self.mean), ravel(self.std)
        return mean - self.truncate * std, mean + self.truncate * std

    def unravel(self, point: Float[Array, " d"]) -> Float[Array, "..."]:
        return point.reshape(jnp.shape(self.mean))

    def flatten(self, value: Float[Array, "..."]) -> Float[Array, " d"]:
        return ravel(value)

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        return point


class Integer(eqx.Module):
    """Integers from `lower` to `upper`, both included.

    The flat coordinate is a float that holds an integer at every sample, and
    `unravel` rounds it, so the target sees an integer value. The target has a
    zero derivative in it almost everywhere, so the tools that read Jacobians
    leave it out.
    """

    lower: Float[Array, "..."]
    upper: Float[Array, "..."]

    @property
    def dimension(self) -> int:
        return math.prod(jnp.shape(self.lower))

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.ones(self.dimension, dtype=bool)

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        lower, upper = self.bounds()
        # Each integer takes an equal share of the unit interval.
        return jnp.minimum(lower + jnp.floor(unit_points * (upper - lower + 1)), upper)

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        return ravel(self.lower).astype(float), ravel(self.upper).astype(float)

    def unravel(self, point: Float[Array, " d"]) -> Float[Array, "..."]:
        return jnp.round(point).reshape(jnp.shape(self.lower))

    def flatten(self, value: Float[Array, "..."]) -> Float[Array, " d"]:
        return ravel(value).astype(float)

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        lower, upper = self.bounds()
        return jnp.clip(jnp.round(point), lower, upper)


class Categorical(eqx.Module):
    """One of a fixed set of options, stacked along the first axis of `choices`.

    The flat coordinates are one weight per option, one-hot at every sample, and
    `unravel` returns `weights @ choices`. Between samples that is a convex mix
    of options, so the surrogate is defined on the whole simplex but only
    trusted at its corners.
    """

    choices: Float[Array, "c ..."]

    @property
    def dimension(self) -> int:
        return int(jnp.shape(self.choices)[0])

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.ones(self.dimension, dtype=bool)

    def sample_unit(self, unit_points: Float[Array, "n c"]) -> Float[Array, "n c"]:
        # The first unit coordinate picks the option, so a low-discrepancy
        # sequence spreads the options evenly.
        count = self.dimension
        index = jnp.minimum(jnp.floor(unit_points[:, 0] * count), count - 1).astype(int)
        return jax.nn.one_hot(index, count)

    def bounds(self) -> tuple[Float[Array, " c"], Float[Array, " c"]]:
        return jnp.zeros(self.dimension), jnp.ones(self.dimension)

    def unravel(self, point: Float[Array, " c"]) -> Float[Array, "..."]:
        return jnp.tensordot(point, jnp.asarray(self.choices, float), axes=1)

    def flatten(self, value: Float[Array, "..."]) -> Float[Array, " c"]:
        options = jnp.asarray(self.choices, float).reshape(self.dimension, -1)
        distance = jnp.sum((options - ravel(value)[None, :]) ** 2, axis=1)
        return jax.nn.one_hot(jnp.argmin(distance), self.dimension)

    def project(self, point: Float[Array, " c"]) -> Float[Array, " c"]:
        return jax.nn.one_hot(jnp.argmax(point), self.dimension)


DomainLeaf = Box | LogRange | Normal | Integer | Categorical
"""A domain over one leaf of the target's input."""


def _is_leaf(node: object) -> bool:
    return isinstance(node, DomainLeaf)


class Product(eqx.Module):
    """Independent leaf domains arranged in a pytree that mirrors the input.

    Each leaf owns a contiguous slice of the flat coordinates, in pytree leaf
    order.
    """

    leaves: tuple[DomainLeaf, ...]
    structure: jax.tree_util.PyTreeDef = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return sum(leaf.dimension for leaf in self.leaves)

    @property
    def discrete_mask(self) -> np.ndarray:
        return np.concatenate([leaf.discrete_mask for leaf in self.leaves])

    def _slices(self) -> list[slice]:
        """The flat coordinates each leaf owns."""
        edges = np.cumsum([0] + [leaf.dimension for leaf in self.leaves])
        return [slice(int(a), int(b)) for a, b in pairwise(edges)]

    def sample_unit(self, unit_points: Float[Array, "n d"]) -> Float[Array, "n d"]:
        return jnp.concatenate(
            [
                leaf.sample_unit(unit_points[:, part])
                for leaf, part in zip(self.leaves, self._slices(), strict=True)
            ],
            axis=1,
        )

    def bounds(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        lowers, uppers = zip(*(leaf.bounds() for leaf in self.leaves), strict=True)
        return jnp.concatenate(lowers), jnp.concatenate(uppers)

    def unravel(self, point: Float[Array, " d"]) -> PyTree[Array]:
        values = [
            leaf.unravel(point[part])
            for leaf, part in zip(self.leaves, self._slices(), strict=True)
        ]
        return jax.tree_util.tree_unflatten(self.structure, values)

    def flatten(self, value: PyTree[Array]) -> Float[Array, " d"]:
        parts = self.structure.flatten_up_to(value)
        return jnp.concatenate(
            [leaf.flatten(part) for leaf, part in zip(self.leaves, parts, strict=True)]
        )

    def project(self, point: Float[Array, " d"]) -> Float[Array, " d"]:
        return jnp.concatenate(
            [
                leaf.project(point[part])
                for leaf, part in zip(self.leaves, self._slices(), strict=True)
            ]
        )


def product(structure: PyTree[DomainLeaf]) -> Product:
    """Build a `Product` from a pytree whose leaves are leaf domains."""
    leaves, treedef = jax.tree_util.tree_flatten(structure, is_leaf=_is_leaf)
    return Product(leaves=tuple(leaves), structure=treedef)


def box(
    lower: Sequence[float] | Float[Array, " d"],
    upper: Sequence[float] | Float[Array, " d"],
) -> Box:
    """Build a `Box` from bound sequences, as float arrays."""
    return Box(lower=jnp.asarray(lower, float), upper=jnp.asarray(upper, float))


def log_range(
    lower: Sequence[float] | Float[Array, " d"],
    upper: Sequence[float] | Float[Array, " d"],
) -> LogRange:
    """Build a `LogRange` from bound sequences, checking that they are positive."""
    lower_array, upper_array = jnp.asarray(lower, float), jnp.asarray(upper, float)
    if not bool(jnp.all(lower_array > 0)):
        raise ValueError("a LogRange needs a positive lower bound")
    return LogRange(lower=lower_array, upper=upper_array)


def normal(
    mean: Sequence[float] | Float[Array, " d"],
    std: Sequence[float] | Float[Array, " d"],
    *,
    truncate: float = 4.0,
) -> Normal:
    """Build a `Normal` from mean and standard deviation sequences."""
    return Normal(mean=jnp.asarray(mean, float), std=jnp.asarray(std, float), truncate=truncate)


def integer(
    lower: Sequence[int] | Float[Array, " d"],
    upper: Sequence[int] | Float[Array, " d"],
) -> Integer:
    """Build an `Integer` from bound sequences, both bounds included."""
    return Integer(lower=jnp.asarray(lower, float), upper=jnp.asarray(upper, float))


def categorical(
    choices: Sequence[float] | Sequence[Sequence[float]] | Float[Array, "c ..."],
) -> Categorical:
    """Build a `Categorical` from its options, stacked along the first axis."""
    return Categorical(choices=jnp.asarray(choices, float))


def box_from_data(points: Float[Array, "n d"]) -> Box:
    """Infer the tightest box that contains a dataset of inputs."""
    return Box(lower=jnp.min(points, axis=0), upper=jnp.max(points, axis=0))
