"""Equality constraints that a surrogate satisfies by construction.

A fitted surrogate breaks a known equation, such as a mass or current balance,
by its fit error. DC3 of Donti, Rolnick and Kolter, https://arxiv.org/abs/2104.12225,
avoids that by predicting only some of the outputs and completing the rest from
the equation. Here a `Constraint` names an equation `h(x, y) = 0` of `k`
equations and the `k` dependent outputs it determines. A `CompletedSurrogate`
takes the free outputs from a base surrogate and solves `h` for the dependent
ones at call time, by Newton from the base's own prediction of them.

An affine constraint completes in one Newton step, exactly. The derivative of
the completion comes from the implicit function theorem, so the completed
surrogate differentiates exactly without unrolling Newton.
"""

from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from surreax.flatten import Unraveler, ravel, ravel_example
from surreax.report import ErrorReport

ConstraintFunction = Callable[[PyTree[Array], Float[Array, " k"]], Float[Array, " c"]]
"""An equation `h(x, y)` from an input and the flat output vector to its residual."""


class Constraint(eqx.Module):
    """An equation `h(x, y) = 0` that determines the `dependent` flat outputs.

    `h` returns one residual entry per dependent output. `steps` is the number of
    Newton steps the completion takes. Left as None, `complete` picks one step
    for a constraint affine in the dependent outputs and eight otherwise.
    """

    function: ConstraintFunction = eqx.field(static=True)
    dependent: tuple[int, ...] = eqx.field(static=True)
    steps: int | None = eqx.field(static=True, default=None)


def _assemble(
    free: Float[Array, " f"],
    dependent_values: Float[Array, " c"],
    free_index: tuple[int, ...],
    dependent_index: tuple[int, ...],
) -> Float[Array, " k"]:
    """The flat output vector from its free and dependent parts."""
    size = len(free_index) + len(dependent_index)
    output = jnp.zeros(size, dtype=free.dtype)
    output = output.at[jnp.asarray(free_index, dtype=int)].set(free)
    return output.at[jnp.asarray(dependent_index, dtype=int)].set(dependent_values)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _completed(
    function: ConstraintFunction,
    free_index: tuple[int, ...],
    dependent_index: tuple[int, ...],
    steps: int,
    x: PyTree[Array],
    free: Float[Array, " f"],
    guess: Float[Array, " c"],
) -> Float[Array, " c"]:
    """Solve `h(x, y) = 0` for the dependent outputs by Newton from `guess`."""

    def residual(values: Float[Array, " c"]) -> Float[Array, " c"]:
        return function(x, _assemble(free, values, free_index, dependent_index))

    values = guess
    for _ in range(steps):
        values = values - jnp.linalg.solve(jax.jacobian(residual)(values), residual(values))
    return values


@_completed.defjvp
def _completed_jvp(
    function: ConstraintFunction,
    free_index: tuple[int, ...],
    dependent_index: tuple[int, ...],
    steps: int,
    primals: tuple,
    tangents: tuple,
) -> tuple[Float[Array, " c"], Float[Array, " c"]]:
    """Differentiate the completion by the implicit function theorem.

    The solution does not depend on the starting guess, so its tangent is
    dropped, and `d dependent = -(dh/d dependent)^{-1} (dh/dx dx + dh/d free d free)`.
    """
    x, free, guess = primals
    x_dot, free_dot, _ = tangents
    values = _completed(function, free_index, dependent_index, steps, x, free, guess)

    def residual(point: PyTree[Array], free_part: Float[Array, " f"], dep: Float[Array, " c"]):
        return function(point, _assemble(free_part, dep, free_index, dependent_index))

    in_dependent = jax.jacobian(residual, argnums=2)(x, free, values)
    _, pushed = jax.jvp(lambda p, f: residual(p, f, values), (x, free), (x_dot, free_dot))
    return values, -jnp.linalg.solve(in_dependent, pushed)


class CompletedSurrogate(eqx.Module):
    """A surrogate whose dependent outputs are solved from a constraint at call time.

    The base surrogate supplies the free outputs and a starting guess for the
    dependent ones, and the result satisfies the constraint to solver precision.
    """

    base: Callable[[PyTree[Array]], PyTree[Array]]
    constraint: Constraint
    free_index: tuple[int, ...] = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    output_unravel: Unraveler = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: PyTree[Array]) -> PyTree[Array]:
        predicted = ravel(self.base(x))
        free = predicted[jnp.asarray(self.free_index, dtype=int)]
        guess = predicted[jnp.asarray(self.constraint.dependent, dtype=int)]
        values = _completed(
            self.constraint.function,
            self.free_index,
            self.constraint.dependent,
            self.steps,
            x,
            free,
            guess,
        )
        return self.output_unravel(
            _assemble(free, values, self.free_index, self.constraint.dependent)
        )


def complete(
    base: Callable[[PyTree[Array]], PyTree[Array]],
    constraint: Constraint,
    inputs: PyTree[Array],
    report: ErrorReport,
    *,
    tolerance: float = 1e-6,
) -> CompletedSurrogate:
    """Wrap a fitted surrogate so that it satisfies `constraint` exactly.

    `inputs` is a batch of inputs, in the surrogate's input structure, that the
    checks run on. The completion needs `dh/d dependent` to be invertible, so a
    choice of dependent outputs whose Jacobian has a reciprocal condition number
    below `tolerance` at any of them is rejected. Without a given step count, the
    completion takes one Newton step when that Jacobian is the same at every
    input, which means the constraint is affine in the dependent outputs.
    """
    first_input = jax.tree.map(lambda leaf: leaf[0], inputs)
    example_output, output_unravel = ravel_example(base(first_input))
    _check_dependent_indices(constraint, example_output.shape[0])
    dependent = constraint.dependent
    free_index = tuple(i for i in range(example_output.shape[0]) if i not in dependent)

    def dependent_jacobian(x: PyTree[Array]) -> Float[Array, "c c"]:
        predicted = ravel(base(x))
        free = predicted[jnp.asarray(free_index, dtype=int)]

        def residual(values: Float[Array, " c"]) -> Float[Array, " c"]:
            return constraint.function(x, _assemble(free, values, free_index, dependent))

        return jax.jacobian(residual)(predicted[jnp.asarray(dependent, dtype=int)])

    jacobians = jax.vmap(dependent_jacobian)(inputs)
    singular = jnp.linalg.svd(jacobians, compute_uv=False)
    conditioning = singular[:, -1] / jnp.maximum(singular[:, 0], 1e-30)
    if float(jnp.min(conditioning)) < tolerance:
        raise ValueError(
            "the constraint does not determine the dependent outputs, its Jacobian in "
            "them is singular at some input"
        )
    steps = constraint.steps
    if steps is None:
        spread = jnp.max(jnp.abs(jacobians - jacobians[0]))
        steps = 1 if float(spread) <= 1e-6 * float(jnp.max(jnp.abs(jacobians))) else 8
    return CompletedSurrogate(
        base=base,
        constraint=constraint,
        free_index=free_index,
        steps=steps,
        output_unravel=output_unravel,
        report=report,
    )


def constraint_residuals(
    surrogate: Callable[[PyTree[Array]], PyTree[Array]],
    constraint: Constraint,
    inputs: PyTree[Array],
) -> Float[Array, " n"]:
    """The norm of the constraint residual a surrogate leaves at each input."""

    def one(x: PyTree[Array]) -> Float[Array, ""]:
        return jnp.linalg.norm(constraint.function(x, ravel(surrogate(x))))

    return jax.vmap(one)(inputs)


def _check_dependent_indices(constraint: Constraint, output_size: int) -> None:
    """Reject dependent indices that repeat or lie outside the outputs."""
    indices = np.asarray(constraint.dependent)
    if indices.size != np.unique(indices).size or np.any(indices >= output_size):
        raise ValueError("dependent outputs must be distinct flat output indices")
