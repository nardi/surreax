"""Sparse direct linear solves for the Newton steps of the large implicit methods.

A residual of a discretized physical model couples each unknown to only a few
others, so its Jacobian in the unknown is sparse. asdex reads that sparsity
pattern from the residual's jaxpr and colors it, so the Jacobian costs one
Jacobian-vector product per color. splineax then factors the sparse Jacobian
with a direct solver behind the lineax solver interface, and can reuse the
symbolic analysis and the pivot order from one Newton step to the next.

The solver follows the JAX precision the caller set, and this module never
changes the JAX configuration. On CPU with x64 enabled it is KLU, which reuses
its factorization. Without x64 it is splineax's `Spsolve`, which solves in
single precision and factors every system from scratch. On a CUDA GPU with the
splineax `cudss` extra installed it is cuDSS.
"""

import warnings
from collections.abc import Callable
from typing import Literal

import asdex
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import splineax
from jaxtyping import Array, Float, Int

from surreax.stencil import StencilResidual

Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""A residual `g(x, y)`, from an input and an unknown to the residual vector."""

NewtonStep = Callable[[Residual, Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""One Newton step `y -> y - (dg/dy)^{-1} g(x, y)` for a residual at an input."""

LinearSolver = Literal["sparse", "matrix_free", "dense"]
"""How a Newton step solves its Jacobian system.

`"sparse"` factors the sparse Jacobian with a direct solver, and falls back to
`"matrix_free"` when the residual has no usable sparsity pattern.
`"matrix_free"` runs GMRES on Jacobian-vector products and never forms the
Jacobian. `"dense"` forms the full Jacobian and factors it.
"""

DENSE_COLOR_FRACTION = 0.5
"""A coloring that needs more colors than this fraction of the unknowns is
treated as dense, since a factorization of it would gain nothing over a dense one."""


def _stencil_pattern(stencil: Int[Array, "m w"]) -> asdex.SparsityPattern:
    """The Jacobian pattern a stencil table states, entry `i` reading `stencil[i]`.

    A padded slot points at some valid index, so it adds an entry that holds an
    explicit zero, and an entry listed twice is kept once.
    """
    table = np.asarray(stencil)
    rows = np.repeat(np.arange(table.shape[0]), table.shape[1])
    entries = np.unique(np.stack([rows, table.reshape(-1)], axis=1), axis=0)
    size = table.shape[0]
    return asdex.SparsityPattern.from_coo(entries[:, 0], entries[:, 1], (size, size))


def residual_pattern(
    residual: Residual,
    input_example: Float[Array, " n"],
    unknown_example: Float[Array, " m"],
) -> asdex.ColoredPattern | None:
    """Read and color the sparsity pattern of `dg/dy`, or return None if it is unusable.

    A `StencilResidual` states its pattern in its stencil table, so it needs no
    tracing. Any other residual is traced by asdex, whose pattern holds for every
    input and unknown. The result is None when asdex cannot read a primitive in
    the residual, or when the coloring needs so many colors that the Jacobian is
    dense in practice. Runs on the host, so call it outside `jax.jit`.
    """
    # asdex warns about a dense coloring, which the color count below handles.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        if isinstance(residual, StencilResidual):
            coloring = asdex.jacobian_coloring_from_sparsity(_stencil_pattern(residual.stencil))
        else:
            try:
                coloring = asdex.jacobian_coloring(
                    lambda unknown: residual(input_example, unknown), unknown_example
                )
            except NotImplementedError:
                return None
    if coloring.num_colors > DENSE_COLOR_FRACTION * unknown_example.shape[0]:
        return None
    return coloring


def sparse_solver(reuse: splineax.HybridSettings | None) -> splineax.AutoSparseLinearSolver:
    """The direct solver for a Newton step, with optional reuse of a stale factorization.

    Iterative refinement is off, since the next Newton step corrects an inexact
    solve, and its default tolerance is out of reach for an ill-conditioned
    Jacobian such as a fine discretization. A `reuse` setting keeps the
    factorization of an earlier step as the preconditioner of an iterative solve.
    """
    return splineax.AutoSparseLinearSolver(iterative=False if reuse is None else reuse)


def sparse_newton_step(
    pattern: asdex.ColoredPattern, reuse: splineax.HybridSettings | None
) -> NewtonStep:
    """A Newton step that factors the sparse Jacobian through the given coloring."""
    tag = splineax.sparsity_coloring_tag(pattern)
    solver = sparse_solver(reuse)

    def step(
        residual: Residual, x: Float[Array, " n"], y: Float[Array, " m"]
    ) -> Float[Array, " m"]:
        operator = lineax.JacobianLinearOperator(
            lambda unknown, _: residual(x, unknown), y, tags=tag
        )
        return y - lineax.linear_solve(operator, residual(x, y), solver).value

    return step


def solve_samples_threaded(
    step: NewtonStep,
    residual: Residual,
    inputs: Float[Array, "s n"],
    starts: Float[Array, "s m"],
    newton_steps: int,
) -> Float[Array, "s m"]:
    """Newton-solve every sample, threading one solver state through its steps.

    The state carries the symbolic analysis and the pivot order from one step to
    the next, so later steps only refactor the values.
    """

    def solve_one(x: Float[Array, " n"], start: Float[Array, " m"]) -> Float[Array, " m"]:
        unknown = start
        for _ in range(newton_steps):
            unknown = step(residual, x, unknown)
        return unknown

    threaded = splineax.stateful_solve_transform(solve_one)
    # Under `vmap`, splineax 0.4.0 turns its reuse check into a select that also
    # runs a fresh factorization for every sample, which is slower than not
    # reusing at all. KLU batches one sample at a time anyway, so `lax.map` loses
    # no parallelism. Change this to `jax.vmap` once splineax shares the reuse
    # predicate across the batch, see design/splineax-issue-batched-reuse.md.
    return jax.lax.map(lambda pair: threaded(*pair), (inputs, starts))


def adjoint_solver(
    linear_solver: LinearSolver, pattern: asdex.ColoredPattern | None
) -> tuple[lineax.AbstractLinearSolver, frozenset[object]]:
    """The lineax solver and operator tags for a transposed Jacobian solve."""
    if pattern is not None:
        return sparse_solver(None), frozenset({splineax.sparsity_coloring_tag(pattern)})
    if linear_solver == "dense":
        return lineax.LU(), frozenset()
    return lineax.GMRES(rtol=1e-6, atol=1e-6), frozenset()


def reduced_state_jacobian(
    residual: Residual,
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    basis: Float[Array, "r m"],
    solver: lineax.AbstractLinearSolver,
    tags: frozenset[object],
) -> Float[Array, "r n"]:
    """The Jacobian of the solved state in the input, projected onto `basis`.

    By the implicit function theorem `dy/dx = -(dg/dy)^{-1} dg/dx`, so row `i` of
    `basis @ dy/dx` is `-w_i^T dg/dx` with `(dg/dy)^T w_i = basis[i]`. That is one
    transposed solve per basis row and one vector-Jacobian product in the input,
    so the full `m` by `n` Jacobian is never formed.
    """
    operator = lineax.JacobianLinearOperator(lambda unknown, _: residual(x, unknown), y, tags=tags)
    _, pullback = jax.vjp(lambda point: residual(point, y), x)
    rows = []
    for direction in basis:
        adjoint = lineax.linear_solve(operator.T, direction, solver).value
        rows.append(-pullback(adjoint)[0])
    return jnp.stack(rows)


def reduced_state_jacobians(
    residual: Residual,
    inputs: Float[Array, "s n"],
    unknowns: Float[Array, "s m"],
    basis: Float[Array, "r m"],
    linear_solver: LinearSolver,
    pattern: asdex.ColoredPattern | None,
) -> Float[Array, "s r n"]:
    """`reduced_state_jacobian` at every solved sample.

    With a sparse pattern the transposed solves of one sample share one solver
    state, and the samples run through `lax.map` for the reason given in
    `solve_samples_threaded`.
    """
    solver, tags = adjoint_solver(linear_solver, pattern)

    def one(x: Float[Array, " n"], y: Float[Array, " m"]) -> Float[Array, "r n"]:
        return reduced_state_jacobian(residual, x, y, basis, solver, tags)

    if pattern is None:
        return jax.vmap(one)(inputs, unknowns)
    threaded = splineax.stateful_solve_transform(one)
    return jax.lax.map(lambda pair: threaded(*pair), (inputs, unknowns))
