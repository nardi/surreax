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

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray


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


def parameter_count(model: Network) -> int:
    """Number of scalar parameters in the network."""
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(leaf.size for leaf in leaves))


def fit_network(
    form: NetworkForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
    jacobians: Float[Array, "n k d"] | None = None,
    derivative_weight: float = 0.0,
    initial: "Network | None" = None,
) -> Network:
    """Fit a feedforward network by minimizing the mean squared error.

    A positive `derivative_weight` with `jacobians` adds a Jacobian term, so the
    network matches target derivatives as well as values. An `initial` network
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
    params, static = eqx.partition(start, eqx.is_inexact_array)

    def loss(trained, _):
        model = eqx.combine(trained, static)
        residual = jnp.mean((jax.vmap(model)(inputs) - targets) ** 2)
        if jacobians is None:
            return residual
        model_jacobians = jax.vmap(jax.jacobian(model))(inputs)
        return residual + derivative_weight * jnp.mean((model_jacobians - jacobians) ** 2)

    solver = optx.OptaxMinimiser(optax.adam(form.learning_rate), rtol=1e-8, atol=1e-8)
    fitted = optx.minimise(loss, solver, params, max_steps=form.steps, throw=False)
    return eqx.combine(fitted.value, static)
