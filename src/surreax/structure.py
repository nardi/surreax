"""Structural decomposition of the sensitivity, shared by both modes.

Which inputs drive which outputs is read from the Jacobian, not the source. The
core needs only a batch of output-to-input Jacobians, one per sample, so it runs
the same way in explicit mode, fed the target Jacobian, and in implicit mode,
fed the Jacobian of the inverse map. Averaging their absolute values gives a
coupling matrix between outputs and inputs.

From the coupling, an input that no output depends on is dropped, which is exact
rather than statistical, and the outputs and inputs split into blocks by the
connected components of the coupling graph, so a block-diagonal coupling means
the fit separates into independent pieces. Coupling below a tolerance is treated
as absent, which is how weak coupling is ignored.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PyTree
from scipy.sparse.csgraph import connected_components

from surreax import sampling
from surreax.domain import Domain
from surreax.flatten import ravel
from surreax.reduction import InputReducer, LinearReduction


def coupling_matrix(jacobians: Float[Array, "s k d"]) -> Float[Array, "k d"]:
    """Average absolute Jacobian over the samples, an output-to-input coupling."""
    return jnp.mean(jnp.abs(jacobians), axis=0)


def _active_mask(coupling: Float[Array, "k d"], tolerance: float) -> Bool[Array, " d"]:
    """Which inputs any output depends on, above the tolerance."""
    strength = jnp.max(coupling, axis=0)
    return strength > tolerance * jnp.max(strength)


def prune_inputs(*, tolerance: float = 1e-6) -> InputReducer:
    """An input reducer that drops inputs no output depends on.

    Reads the dependency from the coupling matrix, so it is exact pruning rather
    than a statistical reduction.
    """

    def build(inputs: Float[Array, "n d"], jacobians: Float[Array, "n k d"]) -> LinearReduction:
        active = np.asarray(_active_mask(coupling_matrix(jacobians), tolerance))
        indices = np.flatnonzero(active)
        selection = jnp.asarray(np.eye(active.shape[0])[indices])
        zeros_reduced = jnp.zeros(indices.shape[0])
        zeros_full = jnp.zeros(active.shape[0])
        return LinearReduction(
            encode_matrix=selection,
            encode_bias=zeros_reduced,
            decode_matrix=selection.T,
            decode_bias=zeros_full,
        )

    return build


class Structure(eqx.Module):
    """The dependency structure read from a target's Jacobian.

    `coupling` is the averaged output-to-input sensitivity. `input_block` and
    `output_block` label which independent block each input and output belongs
    to, so equal labels move together and different labels are decoupled.
    """

    coupling: Float[Array, "k d"]
    input_block: Int[Array, " d"]
    output_block: Int[Array, " k"]


def _blocks(
    coupling: Float[Array, "k d"], tolerance: float
) -> tuple[Int[Array, " d"], Int[Array, " k"]]:
    """Label inputs and outputs by the connected components of the coupling graph."""
    outputs, inputs = coupling.shape
    strong = np.asarray(coupling) > tolerance * float(jnp.max(coupling))
    size = outputs + inputs
    adjacency = np.zeros((size, size), dtype=bool)
    adjacency[:outputs, outputs:] = strong
    adjacency[outputs:, :outputs] = strong.T
    _, labels = connected_components(adjacency, directed=False)
    return jnp.asarray(labels[outputs:]), jnp.asarray(labels[:outputs])


def analyze(
    target: Callable[[PyTree[Array]], PyTree[Array]],
    domain: Domain,
    *,
    samples: int = 256,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> Structure:
    """Read the dependency structure of a target over a domain.

    Samples the domain, takes the target Jacobian at each point, and returns the
    coupling matrix and the block labels. The analysis is advisory.
    """
    inputs = sampling.sample(domain, samples, seed=seed)

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(domain.unravel(point)))

    jacobians = jax.vmap(jax.jacobian(flat_target))(inputs)
    coupling = coupling_matrix(jacobians)
    input_block, output_block = _blocks(coupling, tolerance)
    return Structure(coupling=coupling, input_block=input_block, output_block=output_block)
