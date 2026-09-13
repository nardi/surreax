"""Held-out error of a fitted surrogate, split by source.

The errors are reported apart because they call for different responses. The
projection error is how much accuracy the output reduction gave up, measured by
reconstruction on held-out points before any fitting, so it is a floor the model
cannot beat. The value and derivative errors are the model on top of that, kept
apart because a surrogate can match values well and derivatives badly, and a
downstream optimizer will consume the derivatives. All are measured on held-out
points in flat output space.
"""

import equinox as eqx
from jaxtyping import Array, Float


class ErrorReport(eqx.Module):
    """Mean and worst held-out error, by source."""

    value_mean: Float[Array, ""]
    value_max: Float[Array, ""]
    derivative_mean: Float[Array, ""]
    derivative_max: Float[Array, ""]
    projection_mean: Float[Array, ""]
    projection_max: Float[Array, ""]
    structure_cost: int = eqx.field(static=True)
    """Number of parts the model is built from, its basis terms or affine pieces.

    A larger structure cost buys accuracy but makes the surrogate more expensive
    to reason about later.
    """
