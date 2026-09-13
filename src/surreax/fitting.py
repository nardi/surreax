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

Derivative supervision, or Sobolev training, matches target Jacobians as well
as target values. Because a form linear in its parameters has a Jacobian that
is also linear in the weights, the derivative equations stack under the value
equations and the same closed-form solve applies.

A sparse fit keeps only the basis terms that matter, which controls the
combinatorial growth of a total-degree basis and lowers the structure cost.
Orthogonal matching pursuit does this greedily: at each step it adds the term
most correlated with the residual, refits the active terms, and continues to a
budget.
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


def _ridge_lstsq(
    design: Float[Array, "rows f"], targets: Float[Array, "rows k"], ridge: float
) -> Float[Array, "k f"]:
    """Solve `design @ weights.T = targets` with a ridge penalty, through lstsq.

    Stacking sqrt(ridge) times the identity below the design turns the ridge fit
    into a plain least squares, so lstsq solves it without forming the normal
    equations.
    """
    feature_count = design.shape[1]
    penalty = jnp.sqrt(ridge) * jnp.eye(feature_count, dtype=design.dtype)
    stacked_design = jnp.concatenate([design, penalty])
    stacked_targets = jnp.concatenate(
        [targets, jnp.zeros((feature_count, targets.shape[1]), dtype=targets.dtype)]
    )
    weights, *_ = jnp.linalg.lstsq(stacked_design, stacked_targets, rcond=None)
    return weights.T


def fit_linear(
    form: LinearForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    ridge: float,
) -> LinearModel:
    """Solve for the weights that best map `inputs` to `targets`."""
    design = jax.vmap(form.features)(inputs)
    return LinearModel(weights=_ridge_lstsq(design, targets, ridge), form=form)


def fit_linear_sobolev(
    form: LinearForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    jacobians: Float[Array, "n k d"],
    *,
    ridge: float,
    derivative_weight: float,
) -> LinearModel:
    """Fit against values and Jacobians together.

    The value equations `weights @ features(x) = target` stack above the
    derivative equations `weights @ jacobian(features)(x) = jacobian(target)`,
    weighted by `derivative_weight`, and the whole system is one least squares
    solve.
    """
    output_count = targets.shape[1]
    design = jax.vmap(form.features)(inputs)
    feature_count = design.shape[1]
    feature_jacobians = jax.vmap(jax.jacobian(form.features))(inputs)

    # Each derivative equation is weights @ Jphi = Jtarget. Transposing to
    # Jphi.T @ weights.T = Jtarget.T puts it in the same shape as the value
    # block, then the per-sample axis folds into the row axis.
    weight = jnp.sqrt(derivative_weight)
    derivative_design = jnp.transpose(feature_jacobians, (0, 2, 1)).reshape(-1, feature_count)
    derivative_targets = jnp.transpose(jacobians, (0, 2, 1)).reshape(-1, output_count)
    stacked_design = jnp.concatenate([design, weight * derivative_design])
    stacked_targets = jnp.concatenate([targets, weight * derivative_targets])
    return LinearModel(weights=_ridge_lstsq(stacked_design, stacked_targets, ridge), form=form)


def fit_linear_sparse(
    form: LinearForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    terms: int,
    ridge: float,
) -> LinearModel:
    """Fit keeping only `terms` basis functions, chosen by matching pursuit.

    One shared active set serves every output. Each step adds the feature with
    the largest total correlation with the residual across outputs, refits the
    active columns, and updates the residual.
    """
    design = jax.vmap(form.features)(inputs)
    feature_count = design.shape[1]
    output_count = targets.shape[1]
    budget = min(terms, feature_count)

    residual = targets
    active: list[int] = []
    coefficients = jnp.zeros((output_count, 0))
    for _ in range(budget):
        scores = jnp.sum(jnp.abs(design.T @ residual), axis=1)
        scores = scores.at[jnp.array(active, dtype=int)].set(-jnp.inf)
        active.append(int(jnp.argmax(scores)))
        columns = design[:, jnp.array(active)]
        coefficients = _ridge_lstsq(columns, targets, ridge)
        residual = targets - columns @ coefficients.T

    weights = jnp.zeros((output_count, feature_count), dtype=design.dtype)
    weights = weights.at[:, jnp.array(active)].set(coefficients)
    return LinearModel(weights=weights, form=form)
