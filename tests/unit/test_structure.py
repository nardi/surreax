"""Tests for the structural decomposition of the sensitivity."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.structure import coupling_matrix


def test_coupling_matrix_averages_absolute_jacobians() -> None:
    jacobians = jnp.array([[[1.0, -2.0]], [[-3.0, 2.0]]])
    coupling = coupling_matrix(jacobians)
    assert jnp.allclose(coupling, jnp.array([[2.0, 2.0]]))


def test_pruning_drops_inputs_no_output_depends_on() -> None:
    def target(x: jax.Array) -> jax.Array:
        # Depends on x0 and x2 only, of five inputs.
        return (2 * x[0] - x[2] ** 2)[None]

    domain = sx.box([-1.0] * 5, [1.0] * 5)
    surrogate = sx.fit(
        target,
        domain,
        sx.ChebyshevPolynomial(degree=2),
        samples=256,
        reduce_inputs=sx.prune_inputs(),
        worst_case_starts=0,
    )
    assert surrogate.input_reduction.encode_matrix.shape == (2, 5)
    assert float(surrogate.report.value_max) < 1e-3


def test_pruning_keeps_every_input_when_all_are_used() -> None:
    def target(x: jax.Array) -> jax.Array:
        return (x[0] + x[1] + x[2])[None]

    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    surrogate = sx.fit(
        target,
        domain,
        sx.Affine(),
        samples=128,
        reduce_inputs=sx.prune_inputs(),
        worst_case_starts=0,
    )
    assert surrogate.input_reduction.encode_matrix.shape == (3, 3)


def test_analyze_splits_a_separable_target_into_blocks() -> None:
    def separable(x: jax.Array) -> jax.Array:
        return jnp.array([x[0] + x[1], x[2] * x[3]])

    structure = sx.analyze(separable, sx.box([-1.0] * 4, [1.0] * 4), samples=128)
    # Two independent blocks: outputs and their inputs are labeled apart.
    assert structure.input_block[0] == structure.input_block[1]
    assert structure.input_block[2] == structure.input_block[3]
    assert structure.input_block[0] != structure.input_block[2]
    assert structure.output_block[0] != structure.output_block[1]


def test_analyze_finds_one_block_for_a_coupled_target() -> None:
    def coupled(x: jax.Array) -> jax.Array:
        return (x[0] * x[1] + x[2])[None]

    structure = sx.analyze(coupled, sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), samples=128)
    assert int(jnp.max(structure.input_block)) == 0
