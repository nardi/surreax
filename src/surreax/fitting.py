"""Closed-form fitting for forms that are linear in their parameters.

The design matrix holds one feature vector per sample. The fit is one
regularized least squares solve of the normal equations, shared across output
coordinates because the right-hand side carries every output at once. There is
no learning rate, no step count, and the answer is deterministic.
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
    gram = design.T @ design + ridge * jnp.eye(feature_count, dtype=design.dtype)
    rhs = design.T @ targets
    weights = jnp.linalg.solve(gram, rhs).T
    return LinearModel(weights=weights, form=form)
