"""Linear and mixed-integer programs built from exported piecewise linear surrogates.

The library exports arrays and leaves the encoding to the user, so these helpers
are the encodings a user would write. Each one minimizes a linear cost of the input
over a box, with bounds on the outputs of the surrogate, and returns the minimizer
in the coordinates of the export.
"""

import numpy as np
from scipy import optimize

import surreax as sx


def _solve(
    program: optimize.LinearConstraint,
    cost: np.ndarray,
    bounds: optimize.Bounds,
    integrality: np.ndarray,
) -> np.ndarray:
    result = optimize.milp(cost, constraints=program, bounds=bounds, integrality=integrality)
    assert result.status == 0, result.message
    return result.x


def _stack(rows: list[np.ndarray], count: int) -> np.ndarray:
    return np.vstack(rows) if rows else np.zeros((0, count))


def minimize_convex(
    export: sx.MaxAffineExport | sx.ConvexNetworkExport,
    cost: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Minimize `cost . x` over the box subject to `output_k(x) <= upper_bounds[k]`.

    An output that is convex in the input has a convex sublevel set, so the
    constraint is a linear program. A maximum of affine pieces is at most a bound
    when every piece is, and a convex network replaces each ReLU by a variable that
    is at least its pre-activation. The nonnegative hidden weights make that
    relaxation exact, since a smaller variable only loosens the constraint.
    """
    d = len(cost)
    if isinstance(export, sx.MaxAffineExport):
        slopes = np.asarray(export.slopes).reshape(-1, d)
        offsets = np.asarray(export.intercepts).reshape(-1)
        outputs = np.repeat(np.arange(export.slopes.shape[0]), export.slopes.shape[1])
        matrix = slopes
        bound = upper_bounds[outputs] - offsets
        program = optimize.LinearConstraint(matrix, -np.inf, bound)
        return _solve(program, cost, optimize.Bounds(lower, upper), np.zeros(d))

    widths = [w.shape[0] for w in export.input_weights[:-1]]
    starts = np.cumsum([d, *widths])
    total = int(starts[-1])
    rows: list[np.ndarray] = []
    bounds_above: list[float] = []
    for layer, width in enumerate(widths):
        # The pre-activation minus the variable stands at most at minus the bias.
        block = np.zeros((width, total))
        block[:, :d] = np.asarray(export.input_weights[layer])
        if layer > 0:
            previous = slice(starts[layer - 1], starts[layer])
            block[:, previous] = np.asarray(export.hidden_weights[layer - 1])
        block[:, starts[layer] : starts[layer + 1]] -= np.eye(width)
        rows.append(block)
        bounds_above.extend(-np.asarray(export.biases[layer]))
    output = np.zeros((len(upper_bounds), total))
    output[:, :d] = np.asarray(export.input_weights[-1])
    output[:, starts[-2] : starts[-1]] = np.asarray(export.hidden_weights[-1])
    rows.append(output)
    bounds_above.extend(upper_bounds - np.asarray(export.biases[-1]))
    full_cost = np.concatenate([cost, np.zeros(total - d)])
    variable_lower = np.concatenate([lower, np.zeros(total - d)])
    variable_upper = np.concatenate([upper, np.full(total - d, np.inf)])
    program = optimize.LinearConstraint(np.vstack(rows), -np.inf, np.array(bounds_above))
    solution = _solve(
        program,
        full_cost,
        optimize.Bounds(variable_lower, variable_upper),
        np.zeros(total),
    )
    return solution[:d]


def preactivation_bounds(
    export: sx.ReluNetworkExport, lower: np.ndarray, upper: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bounds on every hidden pre-activation over the box, by interval arithmetic."""
    bounds = []
    low, high = lower, upper
    for weights, biases in zip(export.weights[:-1], export.biases[:-1], strict=True):
        matrix = np.asarray(weights)
        positive, negative = np.maximum(matrix, 0.0), np.minimum(matrix, 0.0)
        pre_low = positive @ low + negative @ high + np.asarray(biases)
        pre_high = positive @ high + negative @ low + np.asarray(biases)
        bounds.append((pre_low, pre_high))
        low, high = np.maximum(pre_low, 0.0), np.maximum(pre_high, 0.0)
    return bounds


def minimize_relu(
    export: sx.ReluNetworkExport,
    cost: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
) -> np.ndarray:
    """Minimize `cost . x` over the box subject to bounds on every network output.

    Each hidden unit `a = relu(p)` has a binary `s` and the bounds `l <= p <= u`
    from interval arithmetic, so `a >= p`, `a >= 0`, `a <= p - l (1 - s)` and
    `a <= u s`. The binary is 1 exactly when the unit is active.
    """
    d = len(cost)
    pre = preactivation_bounds(export, lower, upper)
    widths = [w.shape[0] for w in export.weights[:-1]]
    activation_starts = np.cumsum([d, *widths])
    binary_start = int(activation_starts[-1])
    total = binary_start + sum(widths)
    binary_starts = np.cumsum([binary_start, *widths])
    rows: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []

    def add(block: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
        rows.append(block)
        lows.append(low)
        highs.append(high)

    for layer, width in enumerate(widths):
        weights = np.asarray(export.weights[layer])
        biases = np.asarray(export.biases[layer])
        low, high = pre[layer]
        active = slice(activation_starts[layer], activation_starts[layer + 1])
        binary = slice(binary_starts[layer], binary_starts[layer + 1])
        # The pre-activation is `p = weights @ previous + biases`, where the
        # previous layer is the input for the first layer.
        feed = np.zeros((width, total))
        if layer == 0:
            feed[:, :d] = weights
        else:
            feed[:, activation_starts[layer - 1] : activation_starts[layer]] = weights
        eye = np.eye(width)
        # a - p >= 0.
        block = -feed
        block[:, active] += eye
        add(block, biases, np.full(width, np.inf))
        # a <= p - l (1 - s), which is a - p - l s <= -l.
        block = -feed
        block[:, active] += eye
        block[:, binary] -= np.diag(low)
        add(block, np.full(width, -np.inf), biases - low)
        # a - u s <= 0.
        block = np.zeros((width, total))
        block[:, active] = eye
        block[:, binary] = -np.diag(high)
        add(block, np.full(width, -np.inf), np.zeros(width))
    output = np.zeros((export.weights[-1].shape[0], total))
    output[:, activation_starts[-2] : activation_starts[-1]] = np.asarray(export.weights[-1])
    shift = np.asarray(export.biases[-1])
    add(output, output_lower - shift, output_upper - shift)
    full_cost = np.concatenate([cost, np.zeros(total - d)])
    variable_lower = np.concatenate([lower, np.zeros(total - d)])
    variable_upper = np.concatenate([upper, np.full(total - d, np.inf)])
    variable_upper[binary_start:] = 1.0
    integrality = np.zeros(total)
    integrality[binary_start:] = 1
    program = optimize.LinearConstraint(
        np.vstack(rows), np.concatenate(lows), np.concatenate(highs)
    )
    solution = _solve(
        program, full_cost, optimize.Bounds(variable_lower, variable_upper), integrality
    )
    return solution[:d]
