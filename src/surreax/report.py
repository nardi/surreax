"""Held-out error of a fitted surrogate, split by source.

The errors are reported apart because they call for different responses. The
projection error is how much accuracy the output reduction gave up, measured by
reconstruction on held-out points before any fitting, so it is a floor the model
cannot beat. The value and derivative errors are the model on top of that, kept
apart because a surrogate can match values well and derivatives badly, and a
downstream optimizer will consume the derivatives. These are measured on
held-out points in flat output space. The worst-case error is found by ascending
the error over the whole domain with gradients, so it catches a bad region that
the held-out sample misses. The coverage reads whether the held-out points lie
where the domain's data is.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ErrorReport(eqx.Module):
    """Mean and worst held-out error, by source."""

    value_mean: Float[Array, ""]
    value_max: Float[Array, ""]
    derivative_mean: Float[Array, ""]
    derivative_max: Float[Array, ""]
    projection_mean: Float[Array, ""]
    projection_max: Float[Array, ""]
    worst_case: Float[Array, ""]
    """Largest error found over the domain, not just over the held-out sample."""

    worst_case_point: Float[Array, " d"]
    """Flat input where the worst-case error occurs."""

    structure_cost: int = eqx.field(static=True)
    """Number of parts the model is built from, its basis terms or affine pieces.

    A larger structure cost buys accuracy but makes the surrogate more expensive
    to reason about later.
    """

    coverage_max: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.asarray(0.0))
    """Largest coverage over the held-out points, see `domain.coverage`.

    At most one means every held-out point lies where the domain says the data
    is. Above one, some held-out points lie away from it, such as a cluster the
    training split missed, or samples drawn in an empty part of the box.
    """
