"""The polynomial degree of a target, read from its jaxpr.

The numerical kind check in `strategy` can tell that an output is curved, but not
which polynomial degree would fit it exactly. Reading the program can. A small
interpreter walks the target's jaxpr and keeps, for each value, a degree per
element in the continuous input coordinates. An input coordinate has degree one
and a constant degree zero. An addition takes the larger degree, a
multiplication the sum, and `integer_pow` multiplies by the exponent. Any
operation that is not a polynomial in a non-constant operand, such as `sin` or a
division by a non-constant, gives infinity, which reads as "not a polynomial".

Operations that only move elements, such as a reshape, a slice or a gather, run
on the degree arrays themselves, which places each degree where its element
goes. The result is an upper bound on the degree, exact unless terms cancel.
"""

import math
from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Primitive, Var
from jaxtyping import Array, ArrayLike, Float, PyTree

from surreax.domain import Domain
from surreax.flatten import ravel

Degrees = np.ndarray
"""A float array of per-element degrees, infinity where a value is not a polynomial."""

_SUM_LIKE = frozenset({"add", "sub", "add_any", "neg", "convert_element_type", "copy", "copy_p"})
"""Elementwise operations whose result has the largest degree of their operands."""

_IDENTITY = frozenset({"stop_gradient", "reduce_precision", "optimization_barrier"})
"""Operations that pass their operand's degree through unchanged."""

_STRUCTURAL = frozenset(
    {
        "reshape",
        "broadcast_in_dim",
        "squeeze",
        "expand_dims",
        "transpose",
        "slice",
        "concatenate",
        "pad",
        "rev",
        "stack",
        "split",
        "gather",
        "dynamic_slice",
        "dynamic_update_slice",
        "scatter",
        "scatter-add",
        "select_n",
    }
)
"""Operations that move elements without combining them, run on the degree arrays.

A `scatter-add` that lands two elements on one place adds their degrees, which
overestimates, so the result stays an upper bound.
"""

_INDEX_OPERANDS = {
    "gather": (1,),
    "dynamic_slice": None,
    "dynamic_update_slice": None,
    "scatter": (1,),
    "scatter-add": (1,),
    "select_n": (0,),
}
"""Operands whose concrete values, not degrees, the operation needs. None means
every operand after the data operands, which `_data_operand_count` gives."""

_CALLS = frozenset(
    {"pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call"}
)
"""Operations that wrap a sub-jaxpr, which the interpreter enters."""


def _data_operand_count(name: str) -> int:
    """How many leading operands of a slicing operation are data."""
    return 2 if name == "dynamic_update_slice" else 1


def _index_positions(name: str, count: int) -> tuple[int, ...]:
    """Positions of the operands an operation reads as concrete indices."""
    positions = _INDEX_OPERANDS.get(name, ())
    if positions is None:
        return tuple(range(_data_operand_count(name), count))
    return positions


class _Value:
    """What the interpreter knows about one jaxpr value.

    `concrete` holds the value itself when it is a constant, and `degree` the
    per-element degrees, zero for a constant.
    """

    __slots__ = ("concrete", "degree")

    def __init__(self, degree: Degrees, concrete: Array | None = None) -> None:
        self.degree = degree
        self.concrete = concrete


def _constant(value: ArrayLike) -> _Value:
    return _Value(np.zeros(np.shape(value)), jnp.asarray(value))


def _broadcast_max(degrees: Sequence[Degrees], shape: tuple[int, ...]) -> Degrees:
    """The largest degree per element over operands broadcast to one shape."""
    largest = np.zeros(shape)
    for degree in degrees:
        largest = np.maximum(largest, np.broadcast_to(degree, shape))
    return np.asarray(largest, dtype=float)


def _not_polynomial(degrees: Sequence[Degrees], shape: tuple[int, ...]) -> Degrees:
    """Degree zero where every operand is constant, infinity elsewhere.

    An operand that does not broadcast to the result, such as the input of a
    reduction, counts through its largest degree.
    """
    fitted = []
    for degree in degrees:
        try:
            fitted.append(np.broadcast_to(degree, shape))
        except ValueError:
            fitted.append(np.full(shape, np.max(degree, initial=0.0)))
    return np.where(_broadcast_max(fitted, shape) == 0, 0.0, np.inf)


def _max_plus_dot(lhs: Degrees, rhs: Degrees, dimension_numbers: tuple) -> Degrees:
    """The degree of a `dot_general`, the largest sum of degrees over the contraction."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_free = [axis for axis in range(lhs.ndim) if axis not in (*lhs_contract, *lhs_batch)]
    rhs_free = [axis for axis in range(rhs.ndim) if axis not in (*rhs_contract, *rhs_batch)]
    batch_shape = [lhs.shape[axis] for axis in lhs_batch]
    lhs_free_shape = [lhs.shape[axis] for axis in lhs_free]
    rhs_free_shape = [rhs.shape[axis] for axis in rhs_free]
    size = math.prod(lhs.shape[axis] for axis in lhs_contract)
    left = np.transpose(lhs, (*lhs_batch, *lhs_free, *lhs_contract)).reshape(
        math.prod(batch_shape), math.prod(lhs_free_shape), size
    )
    right = np.transpose(rhs, (*rhs_batch, *rhs_free, *rhs_contract)).reshape(
        math.prod(batch_shape), math.prod(rhs_free_shape), size
    )
    combined = np.max(left[:, :, None, :] + right[:, None, :, :], axis=3, initial=0.0)
    return combined.reshape((*batch_shape, *lhs_free_shape, *rhs_free_shape))


def _apply(
    primitive: Primitive, params: dict, operands: list[_Value], shapes: list[tuple]
) -> list[Degrees]:
    """The degrees of an operation's results from its operands' degrees."""
    results = _rules(primitive, params, operands, shapes)
    return [np.asarray(result, dtype=float) for result in results]


def _rules(
    primitive: Primitive, params: dict, operands: list[_Value], shapes: list[tuple]
) -> Sequence[Degrees | np.floating]:
    """The degree rules, whose numpy results may come back as scalars."""
    name = primitive.name
    degrees = [operand.degree for operand in operands]
    if name in _SUM_LIKE or name in _IDENTITY:
        return [_broadcast_max(degrees, shapes[0])]
    if name == "mul":
        return [np.broadcast_to(degrees[0], shapes[0]) + np.broadcast_to(degrees[1], shapes[0])]
    if name == "div":
        return [np.where(np.broadcast_to(degrees[1], shapes[0]) == 0, degrees[0], np.inf)]
    if name == "square":
        return [2.0 * degrees[0]]
    if name == "integer_pow":
        power = params["y"]
        if power >= 0:
            return [power * degrees[0]]
        return [np.where(degrees[0] == 0, 0.0, np.inf)]
    if name == "pow" and operands[1].concrete is not None:
        exponent = np.asarray(operands[1].concrete)
        if np.all(exponent == np.round(exponent)) and np.all(exponent >= 0):
            return [np.broadcast_to(degrees[0], shapes[0]) * exponent]
    if name == "reduce_sum":
        return [np.max(degrees[0], axis=tuple(params["axes"]), initial=0.0)]
    if name == "dot_general":
        return [_max_plus_dot(degrees[0], degrees[1], params["dimension_numbers"])]
    if name in _STRUCTURAL:
        return _apply_structural(primitive, params, operands, shapes)
    return [_not_polynomial(degrees, shape) for shape in shapes]


def _apply_structural(
    primitive: Primitive, params: dict, operands: list[_Value], shapes: list[tuple]
) -> list[Degrees]:
    """Run an element-moving operation on the degree arrays.

    Its index operands must be constants, read by their values. A non-constant
    index, such as a predicate that depends on the input, makes the result
    piecewise, so it is not a polynomial.
    """
    indices = _index_positions(primitive.name, len(operands))
    arguments = []
    for position, operand in enumerate(operands):
        if position in indices:
            if operand.concrete is None:
                return [_not_polynomial([o.degree for o in operands], s) for s in shapes]
            arguments.append(operand.concrete)
        else:
            arguments.append(jnp.asarray(operand.degree, dtype=jnp.float32))
    results = primitive.bind(*arguments, **params)
    results = results if primitive.multiple_results else [results]
    return [np.asarray(result, dtype=float) for result in results]


def _evaluate(jaxpr: Jaxpr, consts: Sequence[Array], inputs: list[_Value]) -> list[_Value]:
    """Interpret a jaxpr on what is known about its inputs."""
    environment: dict[Var, _Value] = {}

    def read(atom: Var | Literal) -> _Value:
        if isinstance(atom, Literal):
            return _constant(atom.val)
        return environment[atom]

    for variable, value in zip(jaxpr.constvars, consts, strict=True):
        environment[variable] = _constant(value)
    for variable, value in zip(jaxpr.invars, inputs, strict=True):
        environment[variable] = value

    for equation in jaxpr.eqns:
        operands = [read(atom) for atom in equation.invars]
        name = equation.primitive.name
        shapes = [tuple(cast(ShapedArray, variable.aval).shape) for variable in equation.outvars]
        if all(operand.concrete is not None for operand in operands):
            outputs = equation.primitive.bind(
                *[operand.concrete for operand in operands], **equation.params
            )
            outputs = outputs if equation.primitive.multiple_results else [outputs]
            results = [_constant(output) for output in outputs]
        elif name in _CALLS:
            inner = _inner_jaxpr(equation.params)
            results = _evaluate(inner.jaxpr, inner.consts, operands)
        else:
            results = [
                _Value(degree)
                for degree in _apply(equation.primitive, dict(equation.params), operands, shapes)
            ]
        for variable, result in zip(equation.outvars, results, strict=True):
            environment[variable] = result
    return [read(atom) for atom in jaxpr.outvars]


def _inner_jaxpr(params: dict) -> ClosedJaxpr:
    """The sub-jaxpr a call-like operation wraps, under whichever name it uses."""
    inner = next(params[key] for key in ("jaxpr", "call_jaxpr", "fun_jaxpr") if key in params)
    return inner if isinstance(inner, ClosedJaxpr) else ClosedJaxpr(inner, ())


def output_degrees(
    function: Callable[[Float[Array, " d"]], Float[Array, " k"]],
    example: Float[Array, " d"],
    input_degrees: Degrees,
) -> tuple[int | None, ...]:
    """The polynomial degree of each output of a flat function, or None.

    `input_degrees` gives each input coordinate a degree, one for a coordinate
    the polynomial is read in and zero for one held constant.
    """
    closed = jax.make_jaxpr(function)(example)
    (result,) = _evaluate(closed.jaxpr, closed.consts, [_Value(np.asarray(input_degrees, float))])
    flat = np.asarray(result.degree, dtype=float).reshape(-1)
    return tuple(int(degree) if np.isfinite(degree) else None for degree in flat)


def polynomial_degrees(
    target: Callable[[PyTree[Array]], PyTree[Array]], domain: Domain
) -> tuple[int | None, ...]:
    """The polynomial degree of each flat output of a target, or None if it has none.

    The degree is read in the continuous flat coordinates of the domain, with
    the discrete coordinates held constant, so an output that is a cubic of a
    `LogRange` input's value is not a polynomial in its logarithm.
    """

    def flat_target(point: Float[Array, " d"]) -> Float[Array, " k"]:
        return ravel(target(domain.unravel(point)))

    lower, upper = domain.bounds()
    example = (lower + upper) / 2
    return output_degrees(flat_target, example, (~domain.discrete_mask).astype(float))
