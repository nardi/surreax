"""Held-out error of a fitted surrogate, split by source.

The value error and the derivative error are reported apart, because a
surrogate can match values well and derivatives badly, and a downstream
optimizer will consume the derivatives. Both are measured on held-out points
in flat output space.
"""

import equinox as eqx
from jaxtyping import Array, Float


class ErrorReport(eqx.Module):
    """Mean and worst held-out error, for values and for derivatives."""

    value_mean: Float[Array, ""]
    value_max: Float[Array, ""]
    derivative_mean: Float[Array, ""]
    derivative_max: Float[Array, ""]
