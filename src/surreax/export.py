"""Exporting a fitted piecewise linear surrogate as plain arrays.

A max-affine surrogate is a maximum of affine pieces, so a solver can minimize it
as a linear program, and a difference of two is a pair of them. Exporting the
slopes and intercepts lets a user write that encoding in whatever optimization
framework they already use. The library exports arrays and nothing else, to keep
free of any solver dependency.

The exported arrays are in the domain's flat coordinates, the input itself for
a box and its logarithm for a `LogRange`. The input
normalization and reduction fold into the slopes and intercepts, and the output
normalization folds into them too, so the pieces act on the unnormalized input
and produce the unnormalized output. This needs an identity output reduction, so
the outputs are not mixed, and a reduced output space is refused.

The mixed-integer and linear-program encodings these arrays feed are the ones
OMLT builds, from Ceccon and colleagues,
[OMLT](https://arxiv.org/abs/2202.02414), which several narrower tools also do
for particular solvers. This module stops at the arrays and points there for the
encoding.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from surreax.fitted import FittedSurrogate
from surreax.fitting import LinearModel
from surreax.forms.difference_max_affine import DifferenceOfMaxAffine
from surreax.forms.grid import Grid
from surreax.forms.max_affine import MaxAffine


class MaxAffineExport(eqx.Module):
    """Affine pieces of a max-affine surrogate, in the original coordinates.

    Output `k` is `max_j (slopes[k, j] . x + intercepts[k, j])`.
    """

    slopes: Float[Array, "k m d"]
    intercepts: Float[Array, "k m"]


class DifferenceExport(eqx.Module):
    """Affine pieces of a difference-of-max-affine surrogate, original coordinates.

    Output `k` is `max_j (positive_slopes[k, j] . x + positive_intercepts[k, j])`
    minus `max_j (negative_slopes[k, j] . x + negative_intercepts[k, j])`.
    """

    positive_slopes: Float[Array, "k m d"]
    positive_intercepts: Float[Array, "k m"]
    negative_slopes: Float[Array, "k n d"]
    negative_intercepts: Float[Array, "k n"]


class GridExport(eqx.Module):
    """Node coordinates and values of a grid surrogate, in the original coordinates.

    `axes` holds the node coordinates per input axis, and `values` holds the node
    output over the grid, output first, so a solver can encode the interpolation.
    """

    axes: tuple[Float[Array, " r"], ...]
    values: Array
    """Node output over the grid, shaped output first then one axis per input."""


def _input_affine(surrogate: FittedSurrogate) -> tuple[Float[Array, "r d"], Float[Array, " r"]]:
    """The affine map from the original input to the model's normalized input."""
    reduction = surrogate.input_reduction
    normalizer = surrogate.input_normalizer
    matrix = reduction.encode_matrix / normalizer.scale[:, None]
    offset = (reduction.encode_bias - normalizer.center) / normalizer.scale
    return matrix, offset


def _output_scale(surrogate: FittedSurrogate) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
    """The per-output affine from the model output to the original output.

    Refuses a non-identity output reduction, which would mix outputs and break the
    per-output piece structure.
    """
    reduction = surrogate.output_reduction
    identity = jnp.eye(reduction.encode_matrix.shape[0])
    if not bool(np.allclose(np.asarray(reduction.encode_matrix), np.asarray(identity))):
        raise ValueError("export needs an identity output reduction, not a reduced output space")
    return surrogate.output_normalizer.scale, surrogate.output_normalizer.center


def _fold(
    slopes: Float[Array, "k m r"],
    intercepts: Float[Array, "k m"],
    matrix: Float[Array, "r d"],
    offset: Float[Array, " r"],
    output_scale: Float[Array, " k"],
    output_center: Float[Array, " k"] | None,
) -> tuple[Float[Array, "k m d"], Float[Array, "k m"]]:
    """Fold the input and output affine maps into a plane set's slopes and intercepts."""
    original_slopes = jnp.einsum("kmr,rd->kmd", slopes, matrix)
    original_intercepts = jnp.einsum("kmr,r->km", slopes, offset) + intercepts
    scaled_slopes = output_scale[:, None, None] * original_slopes
    scaled_intercepts = output_scale[:, None] * original_intercepts
    if output_center is not None:
        scaled_intercepts = scaled_intercepts + output_center[:, None]
    return scaled_slopes, scaled_intercepts


def _grid_export(surrogate: FittedSurrogate, model: LinearModel, resolution: int) -> GridExport:
    """Recover the grid node coordinates and values in the original coordinates."""
    reduction = surrogate.input_reduction
    identity = jnp.eye(reduction.encode_matrix.shape[0])
    if not bool(np.allclose(np.asarray(reduction.encode_matrix), np.asarray(identity))):
        raise ValueError("grid export needs an identity input reduction")
    output_scale, output_center = _output_scale(surrogate)
    normalizer = surrogate.input_normalizer
    normalized_axis = jnp.linspace(-1.0, 1.0, resolution)
    axes = tuple(
        center + scale * normalized_axis
        for center, scale in zip(normalizer.center, normalizer.scale, strict=True)
    )
    dimension = normalizer.center.shape[0]
    outputs = model.weights.shape[0]
    node_values = model.weights.reshape((outputs, *([resolution] * dimension)))
    reshaped_scale = output_scale.reshape((outputs, *([1] * dimension)))
    reshaped_center = output_center.reshape((outputs, *([1] * dimension)))
    return GridExport(axes=axes, values=reshaped_scale * node_values + reshaped_center)


def export(surrogate: FittedSurrogate) -> MaxAffineExport | DifferenceExport | GridExport:
    """Export a fitted piecewise linear surrogate as arrays in original coordinates."""
    model = surrogate.model
    if isinstance(model, LinearModel) and isinstance(model.form, Grid):
        return _grid_export(surrogate, model, model.form.resolution)
    matrix, offset = _input_affine(surrogate)
    scale, center = _output_scale(surrogate)
    if isinstance(model, MaxAffine):
        slopes, intercepts = _fold(model.slopes, model.intercepts, matrix, offset, scale, center)
        return MaxAffineExport(slopes=slopes, intercepts=intercepts)
    if isinstance(model, DifferenceOfMaxAffine):
        # The output center folds into the positive part only, so the difference
        # of the two maxima still recovers the shifted output.
        positive_slopes, positive_intercepts = _fold(
            model.positive_slopes, model.positive_intercepts, matrix, offset, scale, center
        )
        negative_slopes, negative_intercepts = _fold(
            model.negative_slopes, model.negative_intercepts, matrix, offset, scale, None
        )
        return DifferenceExport(
            positive_slopes=positive_slopes,
            positive_intercepts=positive_intercepts,
            negative_slopes=negative_slopes,
            negative_intercepts=negative_intercepts,
        )
    raise TypeError(f"no array export for a {type(model).__name__} surrogate")
