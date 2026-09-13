"""Linear dimension reduction on the input and output sides.

A `LinearReduction` is an affine encode paired with an affine decode. Composing
a linear map with a piecewise linear or convex model keeps that structure in the
original coordinates, which is why the reduction is linear.

The identity reduction passes its input through unchanged. The other builders
fit a reduction from data. Principal component analysis finds the directions in
which the data varies, and an active subspace finds the directions in which the
function varies, read from the average outer product of the target gradient.

A builder returns a callable, so `fit` applies it to the data it has. An output
reducer takes the flat outputs. An input reducer takes the flat inputs and their
Jacobians, and each one uses whichever it needs.
"""

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class OutputReducer(Protocol):
    """Builds a reduction from the flat outputs."""

    def __call__(self, outputs: Float[Array, "n k"]) -> LinearReduction: ...


@runtime_checkable
class InputReducer(Protocol):
    """Builds a reduction from the flat inputs and their Jacobians."""

    def __call__(
        self, inputs: Float[Array, "n d"], jacobians: Float[Array, "n k d"]
    ) -> LinearReduction: ...


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


def _choose_rank(strengths: Float[Array, " m"], rank: int | None, variance: float) -> int:
    """Pick how many directions to keep.

    A given `rank` wins. Otherwise keep the fewest directions whose squared
    strengths reach the `variance` fraction of the total.
    """
    available = strengths.shape[0]
    if rank is not None:
        return min(rank, available)
    ratios = jnp.cumsum(strengths**2) / jnp.sum(strengths**2)
    return min(int(jnp.searchsorted(ratios, variance)) + 1, available)


def _projection(directions: Float[Array, "r n"], center: Float[Array, " n"]) -> LinearReduction:
    """The reduction that projects onto `directions` after subtracting `center`."""
    return LinearReduction(
        encode_matrix=directions,
        encode_bias=-directions @ center,
        decode_matrix=directions.T,
        decode_bias=center,
    )


def pca_outputs(*, rank: int | None = None, variance: float = 0.99) -> OutputReducer:
    """An output reducer that keeps the leading principal directions of the outputs."""

    def build(outputs: Float[Array, "n k"]) -> LinearReduction:
        center = jnp.mean(outputs, axis=0)
        _, singular_values, directions = jnp.linalg.svd(outputs - center, full_matrices=False)
        chosen = _choose_rank(singular_values, rank, variance)
        return _projection(directions[:chosen], center)

    return build


def pca_inputs(*, rank: int | None = None, variance: float = 0.99) -> InputReducer:
    """An input reducer that keeps the leading principal directions of the inputs."""

    def build(inputs: Float[Array, "n d"], jacobians: Float[Array, "n k d"]) -> LinearReduction:
        del jacobians
        center = jnp.mean(inputs, axis=0)
        _, singular_values, directions = jnp.linalg.svd(inputs - center, full_matrices=False)
        chosen = _choose_rank(singular_values, rank, variance)
        return _projection(directions[:chosen], center)

    return build


def active_subspace_inputs(*, rank: int | None = None, variance: float = 0.99) -> InputReducer:
    """An input reducer that keeps the directions in which the function varies most.

    Averages the outer product of the target Jacobian with itself, then takes the
    leading eigenvectors of that matrix.
    """

    def build(inputs: Float[Array, "n d"], jacobians: Float[Array, "n k d"]) -> LinearReduction:
        center = jnp.mean(inputs, axis=0)
        gram = jnp.mean(jnp.einsum("nkd,nke->nde", jacobians, jacobians), axis=0)
        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        # eigh returns ascending eigenvalues, so read the leading ones from the end.
        strengths = jnp.sqrt(jnp.clip(eigenvalues[::-1], min=0.0))
        chosen = _choose_rank(strengths, rank, variance)
        directions = eigenvectors[:, ::-1][:, :chosen].T
        return _projection(directions, center)

    return build
