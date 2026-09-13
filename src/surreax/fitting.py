"""Closed-form fitting for forms that are linear in their parameters.

The design matrix holds one feature vector per sample. The fit is one
regularized least squares solve, shared across output coordinates because the
right-hand side carries every output at once. There is no learning rate, no
step count, and the answer is deterministic.

The solve goes through `lstsq` on the design matrix rather than the normal
equations, so it keeps the condition number of the design matrix instead of
squaring it. Regularization enters by stacking `sqrt(ridge)` times the identity
below the design, the standard way to write a ridge fit as a plain least
squares.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from surreax.forms.protocol import LinearForm


class LinearModel(eqx.Module):
    """A fitted form: weights times the form's feature vector."""

    weights: Float[Array, "k f"]
    form: LinearForm = eqx.field(static=True)

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        return self.weights @ self.form.features(x)


def fit_linear(
    form: LinearForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    ridge: float,
) -> LinearModel:
    """Solve for the weights that best map `inputs` to `targets`."""
    design = jax.vmap(form.features)(inputs)
    feature_count = design.shape[1]
    # Stacking sqrt(ridge) * I below the design turns the ridge fit into a
    # plain least squares, which lstsq solves without forming the normal
    # equations.
    penalty = jnp.sqrt(ridge) * jnp.eye(feature_count, dtype=design.dtype)
    stacked_design = jnp.concatenate([design, penalty])
    stacked_targets = jnp.concatenate(
        [targets, jnp.zeros((feature_count, targets.shape[1]), dtype=targets.dtype)]
    )
    weights, *_ = jnp.linalg.lstsq(stacked_design, stacked_targets, rcond=None)
    return LinearModel(weights=weights.T, form=form)
