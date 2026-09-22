"""Hyper-reduction: a reduced solve that reads only a few entries of the residual.

A reduced solve works in a small latent, but it still evaluates the full
residual at every Newton step, so its cost grows with the state size. The
discrete empirical interpolation method of Chaturantabut and Sorensen,
https://doi.org/10.1137/090766498, fixes that. Residual snapshots taken near the
manifold give a basis `U` for the residual, and a greedy choice of entries `P`
makes `U` recoverable from those entries alone. The latent solve then minimizes
the gappy residual `||pinv(U[P]) g_P(x, D(z))||`, a least-squares Petrov-Galerkin
solve on the sampled entries, the GNAT method of Carlberg and colleagues,
https://doi.org/10.1016/j.jcp.2013.02.028.

Evaluating `g_P` needs the state only on the stencils of the sampled entries, so
the solve decodes just those rows and reads the residual only there. Its cost
depends on the number of samples and the stencil width, not on the state size.
It needs a `StencilResidual`, since a residual written as one opaque function
evaluates every entry anyway. It slots under both the linear reduced-order
solver and the learned-manifold solver, since both decoders can produce a
chosen set of rows.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from surreax import sampling
from surreax.autoencoder import Autoencoder, AutoencoderSolver, LatentGuess, least_squares_latent
from surreax.domain import Domain
from surreax.implicit import ReducedSolver
from surreax.stencil import StencilResidual


class LinearDecoder(eqx.Module):
    """The decoder of a linear basis, `y = center + basis @ z`."""

    center: Float[Array, " m"]
    basis: Float[Array, "m r"]

    def decode(self, z: Float[Array, " r"]) -> Float[Array, " m"]:
        """Map a latent to the full state."""
        return self.center + self.basis @ z

    def decode_rows(self, z: Float[Array, " r"], rows: Int[Array, " s"]) -> Float[Array, " s"]:
        """Map a latent to the state on `rows` only."""
        return self.center[rows] + self.basis[rows] @ z


class HyperReducedSolver(eqx.Module):
    """A reduced-order solver whose latent solve reads only sampled residual entries.

    `sampled` are the residual entries the solve evaluates, `rows` the state
    entries their stencils touch, and `weights` the pseudo-inverse of the residual
    basis on the sampled entries. A call decodes the full state at the end, but
    the latent solve itself costs the same at any state size.
    """

    decoder: LinearDecoder | Autoencoder
    guess: LatentGuess | None
    sampled: Int[Array, " k"]
    """Residual entries the latent solve evaluates."""

    rows: Int[Array, " s"]
    """State entries the stencils of the sampled entries read."""

    local: Int[Array, "k w"]
    """Stencil of each sampled entry, as positions within `rows`."""

    weights: Float[Array, "q k"]
    """Pseudo-inverse of the residual basis on the sampled entries."""

    residual: StencilResidual = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    reduced_steps: int = eqx.field(static=True)

    def gappy_residual(self, x: Float[Array, " n"], z: Float[Array, " r"]) -> Float[Array, " q"]:
        """The residual recovered from the sampled entries, in the residual basis."""
        state = self.decoder.decode_rows(z, self.rows)
        entries = self.residual.entries(x, self.sampled, state[self.local])
        return self.weights @ entries

    def latent(self, x: Float[Array, " n"]) -> Float[Array, " r"]:
        """Solve for the latent at `x` from the sampled entries alone."""
        start = self.guess(x) if self.guess is not None else jnp.zeros(self.rank)
        return least_squares_latent(_gappy_objective, self, x, start, self.reduced_steps)

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.decoder.decode(self.latent(x))


def _gappy_objective(
    solver: HyperReducedSolver, x: Float[Array, " n"], z: Float[Array, " r"]
) -> Float[Array, " q"]:
    """The objective the latent solve drives to least squares."""
    return solver.gappy_residual(x, z)


def _deim_indices(basis: Float[Array, "m q"]) -> list[int]:
    """Greedy empirical interpolation points for the columns of `basis`.

    Each step picks the entry where the next basis vector is worst interpolated
    by the ones before it on the entries chosen so far.
    """
    chosen = [int(jnp.argmax(jnp.abs(basis[:, 0])))]
    for column in range(1, basis.shape[1]):
        picked = jnp.asarray(chosen)
        coefficients = jnp.linalg.solve(basis[picked, :column], basis[picked, column])
        leftover = basis[:, column] - basis[:, :column] @ coefficients
        chosen.append(int(jnp.argmax(jnp.abs(leftover))))
    return chosen


def hyper_reduce(
    solver: ReducedSolver | AutoencoderSolver,
    residual: StencilResidual,
    domain: Domain,
    *,
    basis_size: int = 16,
    samples: int = 24,
    snapshots: int = 256,
    spread: float = 0.3,
    reduced_steps: int = 6,
    seed: int = 0,
) -> HyperReducedSolver:
    """Hyper-reduce a reduced-order solver so its latent solve reads few entries.

    Solves the latent at `snapshots` sampled inputs, perturbs each latent by
    `spread` of its standard deviation, and takes the residual there, so the
    residual basis covers the states a latent solve passes through. Keeps
    `basis_size` residual modes, picks as many entries by empirical
    interpolation, and adds the entries of highest leverage up to `samples`, since
    a few spare entries make the least-squares solve better conditioned.
    """
    if samples < basis_size:
        raise ValueError("samples must be at least basis_size")
    if isinstance(solver, ReducedSolver):
        decoder: LinearDecoder | Autoencoder = LinearDecoder(solver.center, solver.basis.T)
        guess = None
        rank = solver.basis.shape[0]
    else:
        decoder = solver.autoencoder
        guess = solver.guess
        rank = solver.autoencoder.linear_decoder.shape[1]

    inputs = sampling.sample(domain, snapshots, seed=seed)
    latents = jax.vmap(solver.latent)(inputs)
    noise = jax.random.normal(jax.random.key(seed), latents.shape)
    perturbed = latents + spread * jnp.std(latents, axis=0) * noise
    residuals = jax.vmap(lambda x, z: residual(x, decoder.decode(z)))(inputs, perturbed)
    modes, _, _ = jnp.linalg.svd(residuals.T, full_matrices=False)
    modes = modes[:, :basis_size]

    chosen = _deim_indices(modes)
    leverage = np.array(jnp.sum(modes**2, axis=1))
    leverage[chosen] = -1.0
    spare = [int(i) for i in np.argsort(-leverage)[: samples - basis_size]]
    sampled = np.asarray(chosen + spare)

    stencil = np.asarray(residual.stencil)[sampled]
    rows = np.unique(stencil)
    local = np.searchsorted(rows, stencil)
    return HyperReducedSolver(
        decoder=decoder,
        guess=guess,
        sampled=jnp.asarray(sampled),
        rows=jnp.asarray(rows),
        local=jnp.asarray(local),
        weights=jnp.linalg.pinv(modes[sampled]),
        residual=residual,
        rank=rank,
        reduced_steps=reduced_steps,
    )
