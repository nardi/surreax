"""A smooth feedforward network form.

This wraps an Equinox multilayer perceptron with a smooth activation, so the
surrogate stays differentiable for a downstream optimizer. It is the accuracy
reference, showing how much accuracy the structured forms cost, and the fallback
when no structured form fits well.

The network is not linear in its parameters, so it fits by first-order
optimization through the optimistix backend driving optax. Because the target
Jacobian is available, derivative supervision extends to the network by adding a
Jacobian term to the loss, the same Sobolev idea as the linear forms but through
the gradient fit rather than a closed-form solve.
"""

from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray

_ModelT = TypeVar("_ModelT", bound=eqx.Module)
"""A trainable network model."""


class Network(eqx.Module):
    """A fitted multilayer perceptron mapping one input to one output vector."""

    mlp: eqx.nn.MLP

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        return self.mlp(x)


class NetworkForm(eqx.Module):
    """A request to fit a feedforward network of a given width and depth."""

    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True, default=2)
    activation: Callable[[Float[Array, " a"]], Float[Array, " a"]] = eqx.field(
        static=True, default=jax.nn.tanh
    )
    steps: int = eqx.field(static=True, default=500)
    learning_rate: float = eqx.field(static=True, default=0.01)


def parameter_count(model: eqx.Module) -> int:
    """Number of scalar parameters in a network model."""
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(leaf.size for leaf in leaves))


def minimize_squared_error(
    start: _ModelT,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    learning_rate: float,
    steps: int,
    jacobians: Float[Array, "n k d"] | None = None,
    derivative_weight: float = 0.0,
    derivative_columns: tuple[int, ...] | None = None,
) -> tuple[_ModelT, Float[Array, ""]]:
    """Train a model from `start` by Adam on the mean squared error.

    A positive `derivative_weight` with `jacobians` adds a Jacobian term, limited
    to the input coordinates in `derivative_columns` when it is given. Returns the
    trained model and its final value loss, so a caller can compare restarts.
    """
    params, static = eqx.partition(start, eqx.is_inexact_array)

    def value_loss(trained):
        model = eqx.combine(trained, static)
        return jnp.mean((jax.vmap(model)(inputs) - targets) ** 2)

    def loss(trained, _):
        residual = value_loss(trained)
        if jacobians is None:
            return residual
        model = eqx.combine(trained, static)
        model_jacobians = jax.vmap(jax.jacobian(model))(inputs)
        mismatch = model_jacobians - jacobians
        if derivative_columns is not None:
            mismatch = mismatch[:, :, jnp.asarray(derivative_columns, dtype=int)]
        return residual + derivative_weight * jnp.mean(mismatch**2)

    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=1e-8, atol=1e-8)
    fitted = optx.minimise(loss, solver, params, max_steps=steps, throw=False)
    return eqx.combine(fitted.value, static), value_loss(fitted.value)


def fit_network(
    form: NetworkForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
    jacobians: Float[Array, "n k d"] | None = None,
    derivative_weight: float = 0.0,
    derivative_columns: tuple[int, ...] | None = None,
    initial: "Network | None" = None,
) -> Network:
    """Fit a feedforward network by minimizing the mean squared error.

    A positive `derivative_weight` with `jacobians` adds a Jacobian term, so the
    network matches target derivatives as well as values, limited to the input
    coordinates in `derivative_columns` when it is given. An `initial` network
    seeds the optimizer, so a refit continues from a previous fit.
    """
    if initial is None:
        start = Network(
            mlp=eqx.nn.MLP(
                in_size=inputs.shape[1],
                out_size=targets.shape[1],
                width_size=form.width,
                depth=form.depth,
                activation=form.activation,
                key=key,
            )
        )
    else:
        start = initial
    fitted, _ = minimize_squared_error(
        start,
        inputs,
        targets,
        learning_rate=form.learning_rate,
        steps=form.steps,
        jacobians=jacobians,
        derivative_weight=derivative_weight,
        derivative_columns=derivative_columns,
    )
    return fitted
