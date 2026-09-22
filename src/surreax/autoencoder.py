"""A reduced-order model on a learned manifold, trained without solved states.

A decoder maps a small latent to the full unknown, so the states it can produce
form a manifold. The reduced solve looks for the latent whose decoded state
leaves the smallest residual, by Gauss-Newton on `||g(x, D(z))||`, which is the
least-squares Petrov-Galerkin projection of Carlberg and colleagues,
https://arxiv.org/abs/2012.06073. That solve always has an answer, even while
the manifold is still poor, which keeps the training stable.

Training needs only forward evaluations of the residual. For sampled inputs it
reduce-solves for the latent, decodes, and lowers the residual that the decoded
state leaves, differentiating through the inner solve with the implicit function
theorem. No state is ever solved in the full space. An encoder is trained
alongside so that `E(D(z)) = z`, which makes it a left inverse of the decoder,
for encoding a known state into the latent. The nonlinear-manifold setting
follows Lee and Carlberg, https://doi.org/10.1016/j.jcp.2019.108973.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from surreax.domain import Domain
from surreax.fitted import FittedSurrogate
from surreax.report import ErrorReport

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


def _stationarity(
    autoencoder: Autoencoder, x: Float[Array, " n"], z: Float[Array, " r"]
) -> Float[Array, " r"]:
    """The least-squares normal equations, `J_D^T J_g^T g`, zero at the solve."""
    residual_value, pullback = jax.vjp(
        lambda latent: autoencoder.residual(x, autoencoder.decode(latent)), z
    )
    return pullback(residual_value)[0]


@eqx.filter_custom_jvp
def _latent_solve(
    autoencoder: Autoencoder, x: Float[Array, " n"], start: Float[Array, " r"], steps: int
) -> Float[Array, " r"]:
    """Gauss-Newton on `||g(x, D(z))||` in the latent, from `start`."""

    def decoded_residual(latent: Float[Array, " r"]) -> Float[Array, " m"]:
        return autoencoder.residual(x, autoencoder.decode(latent))

    latent = start
    for _ in range(steps):
        jacobian = jax.jacfwd(decoded_residual)(latent)
        latent = latent - jnp.linalg.lstsq(jacobian, decoded_residual(latent))[0]
    return latent


@_latent_solve.def_jvp
def _latent_solve_jvp(
    primals: tuple[Autoencoder, Float[Array, " n"], Float[Array, " r"], int],
    tangents: tuple[PyTree, Float[Array, " n"] | None, Float[Array, " r"] | None, None],
) -> tuple[Float[Array, " r"], Float[Array, " r"]]:
    """Differentiate the latent solve by the implicit function theorem.

    The solve zeroes the normal equations `q`, so a change in the autoencoder or
    the input moves the latent by `-(dq/dz)^{-1} q_dot`, where `q_dot` is the
    change in `q` at a fixed latent. This avoids differentiating the Gauss-Newton
    iterates.
    """
    autoencoder, x, start, steps = primals
    autoencoder_tangent, x_tangent, _, _ = tangents
    latent = _latent_solve(autoencoder, x, start, steps)
    arrays, static = eqx.partition(autoencoder, eqx.is_inexact_array)
    if x_tangent is None:
        x_tangent = jnp.zeros_like(x)
    # An unperturbed array arrives with a `None` tangent, which `jax.jvp` needs as
    # an explicit zero of the same shape.
    autoencoder_tangent = jax.tree.map(
        lambda array, tangent: None if array is None else (
            jnp.zeros_like(array) if tangent is None else tangent
        ),
        arrays,
        autoencoder_tangent,
        is_leaf=lambda leaf: leaf is None,
    )
    latent_jacobian = jax.jacobian(lambda z: _stationarity(autoencoder, x, z))(latent)
    _, stationarity_tangent = jax.jvp(
        lambda a, point: _stationarity(eqx.combine(a, static), point, latent),
        (arrays, x),
        (autoencoder_tangent, x_tangent),
    )
    return latent, -jnp.linalg.solve(latent_jacobian, stationarity_tangent)


class AutoencoderSolver(eqx.Module):
    """A reduced-order solver on a learned manifold, a differentiable oracle.

    A call reduce-solves for the latent and decodes it to the full state. Its
    input derivative comes from the implicit function theorem.
    """

    autoencoder: Autoencoder
    training_residual: Float[Array, ""]
    """Mean residual left on held-out inputs, relative to the residual at zero."""

    consistency_error: Float[Array, ""]
    """Mean distance between `E(D(z))` and `z` over sampled latents."""

    reduced_steps: int = eqx.field(static=True)

    def latent(self, x: Float[Array, " n"]) -> Float[Array, " r"]:
        """Reduce-solve for the latent at `x`."""
        start = jnp.zeros(self.autoencoder.linear_decoder.shape[1])
        return _latent_solve(self.autoencoder, x, start, self.reduced_steps)

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.autoencoder.decode(self.latent(x))


def _relative_residual(
    autoencoder: Autoencoder, inputs: Float[Array, "n d"], steps: int
) -> Float[Array, ""]:
    """Mean residual the decoded solve leaves, over the mean residual at zero."""
    rank = autoencoder.linear_decoder.shape[1]
    latents = jax.vmap(lambda x: _latent_solve(autoencoder, x, jnp.zeros(rank), steps))(inputs)
    left = jax.vmap(
        lambda x, z: jnp.linalg.norm(autoencoder.residual(x, autoencoder.decode(z)))
    )(inputs, latents)
    unknown_dimension = autoencoder.center.shape[0]
    base = jax.vmap(
        lambda x: jnp.linalg.norm(autoencoder.residual(x, jnp.zeros(unknown_dimension)))
    )(inputs)
    scale = jnp.mean(base)
    return jnp.mean(left) / jnp.where(scale > 0, scale, 1.0)


def train_autoencoder(
    residual: Residual,
    domain: Domain,
    *,
    unknown_dimension: int,
    rank: int,
    width: int,
    depth: int,
    steps: int,
    learning_rate: float,
    batch: int,
    consistency_weight: float,
    reduced_steps: int,
    held_out: Float[Array, "h d"],
    seed: int,
) -> AutoencoderSolver:
    """Train the autoencoder on the residual alone, with no solved states.

    Each step samples inputs and latents, reduce-solves the inputs, and lowers
    the residual the decoded latents leave plus the consistency error of the
    encoder, through Adam with a clipped gradient.
    """
    key = jax.random.key(seed)
    init_key, key = jax.random.split(key)
    autoencoder = _initial_autoencoder(residual, unknown_dimension, rank, width, depth, init_key)
    dimension = held_out.shape[1]

    # A fixed scale from the residual at zero keeps the loss near one at the start.
    base = jax.vmap(lambda x: jnp.sum(residual(x, jnp.zeros(unknown_dimension)) ** 2))(held_out)
    scale = jnp.mean(base)
    scale = jnp.where(scale > 0, scale, 1.0)

    def loss(
        model: Autoencoder, inputs: Float[Array, "b d"], latents: Float[Array, "b r"]
    ) -> Float[Array, ""]:
        solved = jax.vmap(lambda x: _latent_solve(model, x, jnp.zeros(rank), reduced_steps))(
            inputs
        )
        physics = jax.vmap(lambda x, z: jnp.sum(residual(x, model.decode(z)) ** 2))(
            inputs, solved
        )
        consistency = jax.vmap(lambda z: jnp.sum((model.encode(model.decode(z)) - z) ** 2))(
            latents
        )
        return jnp.mean(physics) / scale + consistency_weight * jnp.mean(consistency)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(learning_rate, steps, alpha=1e-2)),
    )
    state = optimizer.init(eqx.filter(autoencoder, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        model: Autoencoder, state: optax.OptState, step_key: PRNGKeyArray
    ) -> tuple[Autoencoder, optax.OptState]:
        input_key, latent_key = jax.random.split(step_key)
        inputs = domain.sample_unit(jax.random.uniform(input_key, (batch, dimension)))
        latents = jax.random.normal(latent_key, (batch, rank))
        gradient = eqx.filter_grad(loss)(model, inputs, latents)
        updates, state = optimizer.update(
            gradient, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), state

    for step_key in jax.random.split(key, steps):
        autoencoder, state = step(autoencoder, state, step_key)

    probe = jax.random.normal(jax.random.key(seed + 1), (batch, rank))
    consistency = jnp.mean(
        jax.vmap(lambda z: jnp.linalg.norm(autoencoder.encode(autoencoder.decode(z)) - z))(probe)
    )
    return AutoencoderSolver(
        autoencoder=autoencoder,
        training_residual=_relative_residual(autoencoder, held_out, reduced_steps),
        consistency_error=consistency,
        reduced_steps=reduced_steps,
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
