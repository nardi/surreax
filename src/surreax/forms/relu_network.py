"""ReLU networks, which a solver can encode exactly.

A ReLU network is piecewise linear, so a mixed-integer program represents it with
one binary per unit. Two forms are offered, matching the two kinds of piecewise
linear function in the other forms.

`ConvexNetworkForm` is an input-convex neural network, from Amos, Xu and Kolter,
https://arxiv.org/abs/1609.07400. Its hidden-to-hidden and output weights are
nonnegative, and each layer also reads the input directly, so every output is a
convex piecewise linear function of the input. That is the function class of the
max-affine form, and a solver minimizes it as a linear program with one auxiliary
variable per unit. The nonnegative weights are the softplus of free parameters.

`ReluNetworkForm` is a plain multilayer perceptron with ReLU activations. It
represents every continuous piecewise linear function, the class of the
difference of max-affine form, and a solver needs a binary per unit. A deep
network can represent a function with far fewer units than the number of affine
pieces it has.

Neither form is linear in its parameters, so both fit by Adam through the shared
trainer of `network`, from several random starts, keeping the best.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from surreax.forms.network import _ModelT, minimize_squared_error


class ConvexNetwork(eqx.Module):
    """A fitted input-convex network, convex in its input for every output.

    Layer `l` computes `relu(softplus(hidden_raw[l]) @ z + input_weights[l] @ x +
    biases[l])`, with no hidden term in the first layer, and the output layer has
    no activation. `hidden_raw[l]` for the last entry feeds the output.
    """

    input_weights: tuple[Float[Array, "_ d"], ...]
    """Skip weights from the input, one matrix per hidden layer and one for the output."""

    biases: tuple[Float[Array, " _"], ...]
    """Bias per hidden layer and for the output."""

    hidden_raw: tuple[Float[Array, "_ h"], ...]
    """Free parameters of the nonnegative weights from the previous layer, one matrix
    per layer after the first and one for the output."""

    def hidden_weights(self) -> tuple[Float[Array, "_ h"], ...]:
        """The nonnegative weights from each layer to the next."""
        return tuple(jax.nn.softplus(raw) for raw in self.hidden_raw)

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        weights = self.hidden_weights()
        z = jax.nn.relu(self.input_weights[0] @ x + self.biases[0])
        for layer in range(1, len(self.input_weights) - 1):
            carried = weights[layer - 1] @ z
            z = jax.nn.relu(carried + self.input_weights[layer] @ x + self.biases[layer])
        return weights[-1] @ z + self.input_weights[-1] @ x + self.biases[-1]


class ConvexNetworkForm(eqx.Module):
    """A request to fit an input-convex network of a given width and depth.

    `depth` is the number of hidden layers.
    """

    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True, default=2)
    steps: int = eqx.field(static=True, default=1500)
    learning_rate: float = eqx.field(static=True, default=0.02)
    restarts: int = eqx.field(static=True, default=3)


class ReluNetwork(eqx.Module):
    """A fitted multilayer perceptron with ReLU activations."""

    mlp: eqx.nn.MLP

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        return self.mlp(x)


class ReluNetworkForm(eqx.Module):
    """A request to fit a ReLU network of a given width and number of hidden layers."""

    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True, default=2)
    steps: int = eqx.field(static=True, default=1500)
    learning_rate: float = eqx.field(static=True, default=0.01)
    restarts: int = eqx.field(static=True, default=3)


def _initial_convex_network(
    form: ConvexNetworkForm, inputs: int, outputs: int, key: PRNGKeyArray
) -> ConvexNetwork:
    """A random input-convex network whose kinks fall inside the normalized box."""
    layers = form.depth
    keys = jax.random.split(key, 3 * (layers + 1))
    sizes = [form.width] * layers + [outputs]
    input_weights = tuple(
        jax.random.normal(keys[i], (size, inputs)) / jnp.sqrt(inputs)
        for i, size in enumerate(sizes)
    )
    # Biases spread the kinks across the box, and the output bias starts at zero.
    spread = [
        jax.random.uniform(keys[layers + 1 + i], (size,), minval=-1.0, maxval=1.0)
        for i, size in enumerate(sizes[:-1])
    ]
    biases = (*spread, jnp.zeros(outputs))
    # A softplus of -1 is about 0.3, so the first outputs start at a modest scale.
    hidden_raw = tuple(
        -1.0 + 0.5 * jax.random.normal(keys[2 * layers + 2 + i], (size, form.width))
        for i, size in enumerate(sizes[1:])
    )
    return ConvexNetwork(input_weights=input_weights, biases=biases, hidden_raw=hidden_raw)


def _best_of_restarts(
    starts: list[_ModelT], train: Callable[[_ModelT], tuple[_ModelT, Float[Array, ""]]]
) -> _ModelT:
    """Train every start and keep the model with the smallest final value loss."""
    trained = [train(start) for start in starts]
    losses = jnp.stack([loss for _, loss in trained])
    return trained[int(jnp.argmin(losses))][0]


def fit_convex_network(
    form: ConvexNetworkForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
    jacobians: Float[Array, "n k d"] | None = None,
    derivative_weight: float = 0.0,
    derivative_columns: tuple[int, ...] | None = None,
    initial: ConvexNetwork | None = None,
) -> ConvexNetwork:
    """Fit an input-convex network by Adam, keeping the best of several random starts.

    An `initial` network joins the starts, so a refit never does worse than the
    network it continues from, measured by the loss after training.
    """
    keys = jax.random.split(key, form.restarts)
    starts = [_initial_convex_network(form, inputs.shape[1], targets.shape[1], k) for k in keys]

    def train(start: ConvexNetwork):
        return minimize_squared_error(
            start,
            inputs,
            targets,
            learning_rate=form.learning_rate,
            steps=form.steps,
            jacobians=jacobians,
            derivative_weight=derivative_weight,
            derivative_columns=derivative_columns,
        )

    if initial is not None:
        starts.append(initial)
    return _best_of_restarts(starts, train)


def fit_relu_network(
    form: ReluNetworkForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
    jacobians: Float[Array, "n k d"] | None = None,
    derivative_weight: float = 0.0,
    derivative_columns: tuple[int, ...] | None = None,
    initial: ReluNetwork | None = None,
) -> ReluNetwork:
    """Fit a ReLU network by Adam, keeping the best of several random starts.

    An `initial` network joins the starts, so a refit never does worse than the
    network it continues from, measured by the loss after training.
    """
    keys = jax.random.split(key, form.restarts)
    starts = [
        ReluNetwork(
            mlp=eqx.nn.MLP(
                in_size=inputs.shape[1],
                out_size=targets.shape[1],
                width_size=form.width,
                depth=form.depth,
                activation=jax.nn.relu,
                key=k,
            )
        )
        for k in keys
    ]

    def train(start: ReluNetwork):
        return minimize_squared_error(
            start,
            inputs,
            targets,
            learning_rate=form.learning_rate,
            steps=form.steps,
            jacobians=jacobians,
            derivative_weight=derivative_weight,
            derivative_columns=derivative_columns,
        )

    if initial is not None:
        starts.append(initial)
    return _best_of_restarts(starts, train)
