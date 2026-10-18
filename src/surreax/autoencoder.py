"""A reduced-order model on a learned manifold, trained on the residual.

A decoder maps a small latent to the full unknown, so the states it can produce
form a manifold. The reduced solve looks for the latent whose decoded state
leaves the smallest residual, by Gauss-Newton on `||g(x, D(z))||`, which is the
least-squares Petrov-Galerkin projection of Carlberg and colleagues,
https://arxiv.org/abs/2012.06073. That solve always has an answer, even while
the manifold is still poor, which keeps the training stable.

Training needs only forward evaluations of the residual. For sampled inputs it
reduce-solves for the latent, decodes, and lowers the residual that the decoded
state leaves, differentiating through the inner solve with the implicit function
theorem. By default no state is ever solved in the full space. A manifold with
a sharp moving feature is hard to find that way, so a few solved snapshots can
pretrain the autoencoder first, and a small learned guess starts each latent
solve near its answer, away from the local minima a sharp feature creates.

An encoder is trained alongside so that `E(D(z)) = z`, which makes it a left
inverse of the decoder, for encoding a known state into the latent. The
nonlinear-manifold setting follows Lee and Carlberg,
https://doi.org/10.1016/j.jcp.2019.108973.
"""

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import splineax
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from surreax.domain import Domain
from surreax.fitted import FittedSurrogate
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport
from surreax.sparse_solve import LinearSolver

Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""A residual `g(x, y)`, from an input and an unknown to the residual vector."""


class Autoencoder(eqx.Module):
    """A decoder from the latent to the unknown, and an encoder back.

    Each map is a linear part plus a small network, so an untrained autoencoder
    starts as a linear subspace and bends only as far as training asks.
    """

    center: Float[Array, " m"]
    """Offset of the manifold in the unknown space."""

    linear_decoder: Float[Array, "m r"]
    linear_encoder: Float[Array, "r m"]
    decoder: eqx.nn.MLP
    encoder: eqx.nn.MLP
    residual: Residual = eqx.field(static=True)

    def decode(self, z: Float[Array, " r"]) -> Float[Array, " m"]:
        """Map a latent to a full state on the manifold."""
        return self.center + self.linear_decoder @ z + self.decoder(z)

    def decode_rows(self, z: Float[Array, " r"], rows: Int[Array, " s"]) -> Float[Array, " s"]:
        """Map a latent to the state on `rows` only, without the other rows."""
        hidden = z
        for layer in self.decoder.layers[:-1]:
            hidden = self.decoder.activation(layer(hidden))
        last = self.decoder.layers[-1]
        # The decoder is built with a final bias, so the layer always carries one.
        bias = cast("Array", last.bias)
        network = last.weight[rows] @ hidden + bias[rows]
        return self.center[rows] + self.linear_decoder[rows] @ z + network

    def encode(self, y: Float[Array, " m"]) -> Float[Array, " r"]:
        """Map a full state to its latent."""
        offset = y - self.center
        return self.linear_encoder @ offset + self.encoder(offset)


def _initial_autoencoder(
    residual: Residual,
    unknown_dimension: int,
    rank: int,
    width: int,
    depth: int,
    key: PRNGKeyArray,
) -> Autoencoder:
    """A random orthonormal subspace with near-zero network corrections."""
    basis_key, decoder_key, encoder_key = jax.random.split(key, 3)
    basis, _ = jnp.linalg.qr(jax.random.normal(basis_key, (unknown_dimension, rank)))

    def quiet(mlp: eqx.nn.MLP) -> eqx.nn.MLP:
        # Shrink the last layer so the network starts as a small correction.
        return eqx.tree_at(lambda m: m.layers[-1].weight, mlp, mlp.layers[-1].weight * 1e-2)

    decoder = eqx.nn.MLP(rank, unknown_dimension, width, depth, jnp.tanh, key=decoder_key)
    encoder = eqx.nn.MLP(unknown_dimension, rank, width, depth, jnp.tanh, key=encoder_key)
    return Autoencoder(
        center=jnp.zeros(unknown_dimension),
        linear_decoder=basis,
        linear_encoder=basis.T,
        decoder=quiet(decoder),
        encoder=quiet(encoder),
        residual=residual,
    )


Objective = Callable[[Any, Float[Array, " n"], Float[Array, " r"]], Float[Array, " k"]]
"""A residual in the latent, `objective(model, x, z)`, that a latent solve drives down.

The model is any Equinox module whose arrays the objective reads, so `Any` is
the narrowest type that covers both the autoencoder and the hyper-reduced
solver without importing the latter here.
"""


def _stationarity(
    objective: Objective, model: Any, x: Float[Array, " n"], z: Float[Array, " r"]
) -> Float[Array, " r"]:
    """The least-squares normal equations, `J^T objective`, zero at the solve."""
    value, pullback = jax.vjp(lambda latent: objective(model, x, latent), z)
    return pullback(value)[0]


@eqx.filter_custom_jvp
def least_squares_latent(
    objective: Objective,
    model: Any,
    x: Float[Array, " n"],
    start: Float[Array, " r"],
    steps: int,
) -> Float[Array, " r"]:
    """Gauss-Newton on `||objective(model, x, z)||` in the latent, from `start`.

    The derivative comes from the implicit function theorem on the normal
    equations, so gradients reach the model and the input without
    differentiating the iterates.
    """
    latent = start
    for _ in range(steps):
        jacobian = jax.jacfwd(lambda z: objective(model, x, z))(latent)
        latent = latent - jnp.linalg.lstsq(jacobian, objective(model, x, latent))[0]
    return latent


@least_squares_latent.def_jvp
def _least_squares_latent_jvp(
    primals: tuple[Objective, Any, Float[Array, " n"], Float[Array, " r"], int],
    tangents: tuple[None, PyTree, Float[Array, " n"] | None, Float[Array, " r"] | None, None],
) -> tuple[Float[Array, " r"], Float[Array, " r"]]:
    """Differentiate the latent solve by the implicit function theorem.

    The solve zeroes the normal equations `q`, so a change in the model or the
    input moves the latent by `-(dq/dz)^{-1} q_dot`, where `q_dot` is the change
    in `q` at a fixed latent.
    """
    objective, model, x, start, steps = primals
    _, model_tangent, x_tangent, _, _ = tangents
    latent = least_squares_latent(objective, model, x, start, steps)
    arrays, static = eqx.partition(model, eqx.is_inexact_array)
    if x_tangent is None:
        x_tangent = jnp.zeros_like(x)
    # An unperturbed array arrives with a `None` tangent, which `jax.jvp` needs as
    # an explicit zero of the same shape.
    model_tangent = jax.tree.map(
        lambda array, tangent: None
        if array is None
        else (jnp.zeros_like(array) if tangent is None else tangent),
        arrays,
        model_tangent,
        is_leaf=lambda leaf: leaf is None,
    )
    latent_jacobian = jax.jacobian(lambda z: _stationarity(objective, model, x, z))(latent)
    _, stationarity_tangent = jax.jvp(
        lambda a, point: _stationarity(objective, eqx.combine(a, static), point, latent),
        (arrays, x),
        (model_tangent, x_tangent),
    )
    return latent, -jnp.linalg.solve(latent_jacobian, stationarity_tangent)


def _decoded_residual(
    autoencoder: Autoencoder, x: Float[Array, " n"], z: Float[Array, " r"]
) -> Float[Array, " m"]:
    """The full residual at the decoded latent, `g(x, D(z))`."""
    return autoencoder.residual(x, autoencoder.decode(z))


def _latent_solve(
    autoencoder: Autoencoder, x: Float[Array, " n"], start: Float[Array, " r"], steps: int
) -> Float[Array, " r"]:
    """Least-squares solve on the learned manifold, from `start`."""
    return least_squares_latent(_decoded_residual, autoencoder, x, start, steps)


class AutoencoderSolve(eqx.Module):
    """Fit a large-unknown target on a learned manifold.

    Trains a decoder and an encoder on the residual, reduce-solves every sample
    on the learned manifold, and distills the input-to-latent map into an
    explicit surrogate that decodes its output. By default it never solves the
    full system. A manifold with a sharp moving feature is hard to find from a
    random start, so a positive `snapshots` solves that many full states first
    and pretrains the autoencoder to reconstruct them, starting from their linear
    basis, before the residual training refines it. `linear_solver` and `reuse`
    set how those snapshot solves factor their Jacobian, and `convergence_tolerance`
    how close to a root they must get, as on `ExactSolve`.
    """

    form: LinearForm
    unknown_dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    width: int = eqx.field(static=True, default=32)
    depth: int = eqx.field(static=True, default=2)
    steps: int = eqx.field(static=True, default=1000)
    learning_rate: float = eqx.field(static=True, default=1e-2)
    batch: int = eqx.field(static=True, default=64)
    consistency_weight: float = eqx.field(static=True, default=0.1)
    reduced_steps: int = eqx.field(static=True, default=5)
    snapshots: int = eqx.field(static=True, default=0)
    pretrain_steps: int = eqx.field(static=True, default=2000)
    newton_steps: int = eqx.field(static=True, default=8)
    convergence_tolerance: float | None = eqx.field(static=True, default=1e-3)
    linear_solver: LinearSolver = eqx.field(static=True, default="sparse")
    reuse: splineax.HybridSettings | None = eqx.field(static=True, default=None)


class LatentGuess(eqx.Module):
    """A small network from the input to a starting latent for the reduced solve.

    A manifold with a sharp feature gives the least-squares solve local minima,
    so starting every solve at zero can land in the wrong one. The guess learns
    where the solve ends up, so each solve starts near its answer.
    """

    center: Float[Array, " n"]
    """Midpoint of the input box, subtracted before the network."""

    scale: Float[Array, " n"]
    """Half-width of the input box, divided out before the network."""

    network: eqx.nn.MLP

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " r"]:
        return self.network((x - self.center) / self.scale)


class AutoencoderSolver(eqx.Module):
    """A reduced-order solver on a learned manifold, a differentiable oracle.

    A call reduce-solves for the latent, starting from the learned guess, and
    decodes it to the full state. Its input derivative comes from the implicit
    function theorem.
    """

    autoencoder: Autoencoder
    guess: LatentGuess
    training_residual: Float[Array, ""]
    """Mean residual left on held-out inputs, relative to the residual at zero."""

    consistency_error: Float[Array, ""]
    """Mean distance between `E(D(z))` and `z` over sampled latents."""

    reduced_steps: int = eqx.field(static=True)

    def latent(self, x: Float[Array, " n"]) -> Float[Array, " r"]:
        """Reduce-solve for the latent at `x`."""
        return _latent_solve(self.autoencoder, x, self.guess(x), self.reduced_steps)

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.autoencoder.decode(self.latent(x))


class _Trainable(eqx.Module):
    """The parts that training updates together."""

    autoencoder: Autoencoder
    guess: LatentGuess


def _relative_residual(
    model: _Trainable, inputs: Float[Array, "n d"], steps: int
) -> Float[Array, ""]:
    """Mean residual the decoded solve leaves, over the mean residual at zero."""
    autoencoder = model.autoencoder
    latents = jax.vmap(lambda x: _latent_solve(autoencoder, x, model.guess(x), steps))(inputs)
    left = jax.vmap(
        lambda x, z: jnp.linalg.norm(autoencoder.residual(x, autoencoder.decode(z)))
    )(inputs, latents)
    unknown_dimension = autoencoder.center.shape[0]
    base = jax.vmap(
        lambda x: jnp.linalg.norm(autoencoder.residual(x, jnp.zeros(unknown_dimension)))
    )(inputs)
    scale = jnp.mean(base)
    return jnp.mean(left) / jnp.where(scale > 0, scale, 1.0)


def _optimize(
    model: _Trainable,
    loss: Callable[[_Trainable, PRNGKeyArray], Float[Array, ""]],
    steps: int,
    learning_rate: float,
    key: PRNGKeyArray,
) -> _Trainable:
    """Run Adam with a clipped gradient and a cosine schedule on a sampled loss."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(learning_rate, steps, alpha=1e-2)),
    )
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        current: _Trainable, state: optax.OptState, step_key: PRNGKeyArray
    ) -> tuple[_Trainable, optax.OptState]:
        gradient = eqx.filter_grad(loss)(current, step_key)
        updates, state = optimizer.update(
            gradient, state, eqx.filter(current, eqx.is_inexact_array)
        )
        return eqx.apply_updates(current, updates), state

    for step_key in jax.random.split(key, steps):
        model, state = step(model, state, step_key)
    return model


def _consistency(model: _Trainable, latents: Float[Array, "b r"]) -> Float[Array, ""]:
    """Mean squared distance between `E(D(z))` and `z`."""
    autoencoder = model.autoencoder
    return jnp.mean(
        jax.vmap(lambda z: jnp.sum((autoencoder.encode(autoencoder.decode(z)) - z) ** 2))(
            latents
        )
    )


def _pretrain(
    model: _Trainable,
    method: AutoencoderSolve,
    snapshot_inputs: Float[Array, "s n"],
    snapshot_states: Float[Array, "s m"],
    key: PRNGKeyArray,
) -> _Trainable:
    """Fit the autoencoder to reconstruct the snapshots, and the guess to encode them."""
    spread = jnp.mean(jnp.sum((snapshot_states - snapshot_states.mean(axis=0)) ** 2, axis=1))
    spread = jnp.where(spread > 0, spread, 1.0)

    def loss(current: _Trainable, step_key: PRNGKeyArray) -> Float[Array, ""]:
        autoencoder = current.autoencoder
        latents = jax.vmap(autoencoder.encode)(snapshot_states)
        rebuilt = jax.vmap(autoencoder.decode)(latents)
        reconstruction = jnp.mean(jnp.sum((rebuilt - snapshot_states) ** 2, axis=1)) / spread
        guessed = jax.vmap(current.guess)(snapshot_inputs)
        guess = jnp.mean(jnp.sum((guessed - jax.lax.stop_gradient(latents)) ** 2, axis=1))
        probes = jax.random.normal(step_key, (method.batch, method.rank))
        return reconstruction + guess + method.consistency_weight * _consistency(current, probes)

    return _optimize(model, loss, method.pretrain_steps, method.learning_rate * 0.3, key)


def train_autoencoder(
    residual: Residual,
    domain: Domain,
    method: AutoencoderSolve,
    *,
    held_out: Float[Array, "h n"],
    seed: int,
    snapshot_inputs: Float[Array, "s n"] | None = None,
    snapshot_states: Float[Array, "s m"] | None = None,
) -> AutoencoderSolver:
    """Train the autoencoder on the residual, after optional snapshot pretraining.

    Each residual step samples inputs, reduce-solves them from the guess, and
    lowers the residual the decoded latents leave, plus the gap between the guess
    and the solved latent and the consistency error of the encoder.
    """
    init_key, guess_key, pretrain_key, train_key = jax.random.split(jax.random.key(seed), 4)
    dimension = held_out.shape[1]
    lower, upper = domain.bounds()
    half = (upper - lower) / 2
    guess = LatentGuess(
        center=(upper + lower) / 2,
        scale=jnp.where(half > 0, half, 1.0),
        network=eqx.nn.MLP(dimension, method.rank, method.width, 2, jnp.tanh, key=guess_key),
    )
    autoencoder = _initial_autoencoder(
        residual, method.unknown_dimension, method.rank, method.width, method.depth, init_key
    )
    learning_rate = method.learning_rate
    if snapshot_inputs is not None and snapshot_states is not None:
        # Start from the linear basis of the snapshots, then let the network bend it.
        center = snapshot_states.mean(axis=0)
        _, _, directions = jnp.linalg.svd(snapshot_states - center, full_matrices=False)
        basis = directions[: method.rank].T
        autoencoder = eqx.tree_at(
            lambda a: (a.center, a.linear_decoder, a.linear_encoder),
            autoencoder,
            (center, basis, basis.T),
        )
        model = _pretrain(
            _Trainable(autoencoder, guess), method, snapshot_inputs, snapshot_states, pretrain_key
        )
        # A pretrained manifold only needs refining, so the residual phase runs slower.
        learning_rate = learning_rate * 0.1
    else:
        model = _Trainable(autoencoder, guess)

    # A fixed scale from the residual at zero keeps the loss near one at the start.
    base = jax.vmap(
        lambda x: jnp.sum(residual(x, jnp.zeros(method.unknown_dimension)) ** 2)
    )(held_out)
    scale = jnp.mean(base)
    scale = jnp.where(scale > 0, scale, 1.0)

    def loss(current: _Trainable, step_key: PRNGKeyArray) -> Float[Array, ""]:
        input_key, latent_key = jax.random.split(step_key)
        inputs = domain.sample_unit(jax.random.uniform(input_key, (method.batch, dimension)))
        manifold = current.autoencoder
        solved = jax.vmap(
            lambda x: _latent_solve(
                manifold, x, jax.lax.stop_gradient(current.guess(x)), method.reduced_steps
            )
        )(inputs)
        physics = jax.vmap(lambda x, z: jnp.sum(residual(x, manifold.decode(z)) ** 2))(
            inputs, solved
        )
        guessed = jax.vmap(current.guess)(inputs)
        guess_gap = jnp.mean(jnp.sum((guessed - jax.lax.stop_gradient(solved)) ** 2, axis=1))
        probes = jax.random.normal(latent_key, (method.batch, method.rank))
        consistency = method.consistency_weight * _consistency(current, probes)
        return jnp.mean(physics) / scale + guess_gap + consistency

    model = _optimize(model, loss, method.steps, learning_rate, train_key)

    probe = jax.random.normal(jax.random.key(seed + 1), (method.batch, method.rank))
    trained = model.autoencoder
    consistency_error = jnp.mean(
        jax.vmap(lambda z: jnp.linalg.norm(trained.encode(trained.decode(z)) - z))(probe)
    )
    return AutoencoderSolver(
        autoencoder=trained,
        guess=model.guess,
        training_residual=_relative_residual(model, held_out, method.reduced_steps),
        consistency_error=consistency_error,
        reduced_steps=method.reduced_steps,
    )


class DecodedSurrogate(eqx.Module):
    """An explicit surrogate that predicts the latent and decodes it.

    The latent map is a fitted surrogate of the input-to-latent map, and the
    decoder lifts its output to the full state, so inference needs no solve. The
    decoder is nonlinear, so the result is not a linear or mixed-integer program.
    """

    latent_map: FittedSurrogate
    solver: AutoencoderSolver
    report: ErrorReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.solver.autoencoder.decode(self.latent_map(x))
