"""Physical implicit systems for testing the reduced-order methods.

Each system is a residual `g(x, y) = 0` with a small input `x`, a few knobs such
as heater powers or a load, and a large unknown `y`, such as the temperature at
every point along a rod or the voltage at every node of a circuit. Every entry of
the residual reads only its neighbors, so each is written as a
`StencilResidual`, and the Jacobian is sparse. The steady state depends on only
the few knobs, so it lies on a low-dimensional manifold.

The chain systems use unscaled differences, so the conduction operator keeps a
bounded condition number and the matrix-free solve converges in a few iterations
at any number of points. Each builder takes the number of points or nodes, so a
test can run a system small for correctness and large for scaling.
"""

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

import surreax as sx


class ImplicitSystem(NamedTuple):
    """A residual, the input box it is fitted over, and a box for solve starts."""

    residual: sx.StencilResidual
    domain: sx.Box
    unknown_domain: sx.Box


def _chain(points: int, offsets: Sequence[int]) -> tuple[Int[Array, "m w"], Float[Array, "m w"]]:
    """Neighbor indices along a chain, and a mask that is zero past either end."""
    index = np.arange(points)[:, None] + np.asarray(offsets)[None, :]
    inside = (index >= 0) & (index < points)
    return jnp.asarray(np.clip(index, 0, points - 1)), jnp.asarray(inside, dtype=float)


def _heater_profiles(points: int, heaters: int) -> Float[Array, "k m"]:
    """Smooth heat deposits along the rod, one per heater."""
    position = (jnp.arange(points) + 1) / (points + 1)
    return jnp.stack([jnp.sin((k + 1) * jnp.pi * position) for k in range(heaters)])


def _radiation(temperature: Float[Array, ""]) -> Float[Array, ""]:
    """A fourth-power heat loss, signed so that a colder point gains heat."""
    return temperature * jnp.abs(temperature) ** 3


def heated_rod(points: int, heaters: int = 4) -> ImplicitSystem:
    """A rod held at a steady temperature by a few heaters, its ends at ambient.

    `g_i = -(y_{i-1} - 2 y_i + y_{i+1}) + 0.5 y_i - sum_j x_j phi_j(i)`, where the
    `0.5 y_i` term is heat lost to the air. The residual is affine in `y` and `x`,
    so the temperature is an exact linear map of the heater powers and the
    manifold is a linear subspace of dimension `heaters`.
    """
    stencil, inside = _chain(points, (-1, 0, 1))
    profiles = _heater_profiles(points, heaters)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 3"]
    ) -> Float[Array, ""]:
        left, here, right = near * inside[i]
        return -(left - 2.0 * here + right) + 0.5 * here - x @ profiles[:, i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0] * heaters, [1.0] * heaters),
        sx.box([-2.0] * points, [2.0] * points),
    )


def radiating_rod(points: int, heaters: int = 4) -> ImplicitSystem:
    """The heated rod, also losing heat by radiation along its whole length.

    `g_i = -(y_{i-1} - 2 y_i + y_{i+1}) + 0.5 y_i + y_i^4 - sum_j x_j phi_j(i)`.
    The fourth power makes the Jacobian depend on the state, so the temperature is
    a curved function of the heater powers and the manifold bends.
    """
    stencil, inside = _chain(points, (-1, 0, 1))
    profiles = _heater_profiles(points, heaters)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 3"]
    ) -> Float[Array, ""]:
        left, here, right = near * inside[i]
        conduction = -(left - 2.0 * here + right) + 0.5 * here
        return conduction + _radiation(here) - x @ profiles[:, i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0] * heaters, [1.0] * heaters),
        sx.box([-2.0] * points, [2.0] * points),
    )


THERMOSTAT_SETPOINT = 0.5
"""Temperature at which the thermostat heaters of the thermostat rod are half on."""

THERMOSTAT_WIDTH = 0.2
"""Temperature span over which a thermostat heater switches from off to fully on."""

THERMOSTAT_POWER = 1.0
"""Heat per point of a fully on thermostat heater."""


def thermostat_rod(points: int) -> ImplicitSystem:
    """A rod with a thermostat-controlled heater at every point, in a varying ambient.

    `g_i = -(y_{i-1} - 2 y_i + y_{i+1}) + 0.5 (y_i - a) - P s((T - y_i) / w) - q phi_i`,
    with ambient temperature `a = x_0`, a load `q = x_1` on a patch `phi`, and
    `s` the logistic function. The ends sit at the ambient temperature. A warm
    ambient leaves the heaters off and the rod follows it. As the ambient cools
    the heaters ramp up and hold the rod near the setpoint `T`, until they
    saturate and the rod follows the ambient again, about `2 P` warmer. The
    temperature is a function of the ambient with slope 1, then about 0, then 1,
    so it is concave at one kink and convex at the other. The Jacobian is
    positive definite, since the heaters push back against a temperature change,
    so the solution is unique.
    """
    stencil, inside = _chain(points, (-1, 0, 1))
    position = (jnp.arange(points) + 1) / (points + 1)
    patch = jnp.exp(-(((position - 0.7) / 0.1) ** 2))

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 3"]
    ) -> Float[Array, ""]:
        ambient, load = x
        # A neighbor past either end is at the ambient temperature.
        left, here, right = jnp.where(inside[i] > 0, near, ambient)
        conduction = -(left - 2.0 * here + right) + 0.5 * (here - ambient)
        heater = THERMOSTAT_POWER * jax.nn.sigmoid((THERMOSTAT_SETPOINT - here) / THERMOSTAT_WIDTH)
        return conduction - heater - load * patch[i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([-2.5, 0.0], [2.0, 1.0]),
        sx.box([-3.0] * points, [3.0] * points),
    )


def radiating_plate(side: int, heaters: int = 4) -> ImplicitSystem:
    """A square plate on a `side` by `side` grid, heated and radiating like the rod.

    Each point exchanges heat with its four grid neighbors, so the Jacobian has
    five nonzeros per row. Unlike the banded chain, its factorization fills in
    between the rows of the grid, which is the case where reusing a stale
    factorization can be cheaper than factoring again.
    """
    rows, cols = np.divmod(np.arange(side * side), side)
    offsets = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
    neighbor_rows = rows[:, None] + np.array([r for r, _ in offsets])[None, :]
    neighbor_cols = cols[:, None] + np.array([c for _, c in offsets])[None, :]
    inside = (neighbor_rows >= 0) & (neighbor_rows < side)
    inside &= (neighbor_cols >= 0) & (neighbor_cols < side)
    index = np.clip(neighbor_rows, 0, side - 1) * side + np.clip(neighbor_cols, 0, side - 1)
    stencil, mask = jnp.asarray(index), jnp.asarray(inside, dtype=float)
    along = _heater_profiles(side, heaters)
    across = _heater_profiles(side, 1)[0]
    profiles = (along[:, :, None] * across[None, None, :]).reshape(heaters, side * side)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 5"]
    ) -> Float[Array, ""]:
        here, *others = near * mask[i]
        conduction = 4.0 * here - sum(others) + 0.5 * here
        return conduction + _radiation(here) - x @ profiles[:, i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0] * heaters, [1.0] * heaters),
        sx.box([-2.0] * side * side, [2.0] * side * side),
    )


def radiating_block(side: int, heaters: int = 4) -> ImplicitSystem:
    """The radiating plate in 3D, a cube on a `side` by `side` by `side` grid.

    Each point exchanges heat with its six grid neighbors, so the Jacobian has
    seven nonzeros per row, and its factorization fills in far more than the
    plate's as the grid grows.
    """
    points = side**3
    layers, rest = np.divmod(np.arange(points), side * side)
    rows, cols = np.divmod(rest, side)
    offsets = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    shifts = np.array(offsets)
    neighbor = np.stack([layers, rows, cols], axis=1)[:, None, :] + shifts[None, :, :]
    inside = np.all((neighbor >= 0) & (neighbor < side), axis=2)
    clipped = np.clip(neighbor, 0, side - 1)
    index = (clipped[:, :, 0] * side + clipped[:, :, 1]) * side + clipped[:, :, 2]
    stencil, mask = jnp.asarray(index), jnp.asarray(inside, dtype=float)
    along = _heater_profiles(side, heaters)
    across = _heater_profiles(side, 1)[0]
    profiles = (along[:, :, None, None] * across[None, None, :, None] * across).reshape(
        heaters, points
    )

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 7"]
    ) -> Float[Array, ""]:
        here, *others = near * mask[i]
        conduction = 6.0 * here - sum(others) + 0.5 * here
        return conduction + _radiation(here) - x @ profiles[:, i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0] * heaters, [1.0] * heaters),
        sx.box([-2.0] * points, [2.0] * points),
    )


PATCHES = (0.25, 0.5, 0.75)
"""Positions of the bare patches along the rod with hot spots, as fractions."""


def rod_with_hot_spots(points: int, heaters: int = 4) -> ImplicitSystem:
    """An insulated rod that radiates only at a few bare patches.

    `g_i = -(y_{i-1} - 2 y_i + y_{i+1}) + 0.5 y_i + 3 m_i y_i^4 - sum_j x_j phi_j(i)`,
    where the patch mask `m_i` is nonzero at only a few points, so the nonlinear
    part of the residual lives on a small set of entries.
    """
    stencil, inside = _chain(points, (-1, 0, 1))
    profiles = _heater_profiles(points, heaters)
    position = jnp.arange(points) / (points - 1)
    width = 1.5 / points
    bumps = sum(jnp.exp(-(((position - patch) / width) ** 2)) for patch in PATCHES)
    mask = jnp.where(bumps > 1e-3, bumps, 0.0)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " 3"]
    ) -> Float[Array, ""]:
        left, here, right = near * inside[i]
        conduction = -(left - 2.0 * here + right) + 0.5 * here
        return conduction + 3.0 * mask[i] * _radiation(here) - x @ profiles[:, i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0] * heaters, [1.0] * heaters),
        sx.box([-2.0] * points, [2.0] * points),
    )


def hot_fluid_pipe(points: int) -> ImplicitSystem:
    """Fluid flowing through a pipe past a narrow heater at a movable position.

    With the heater position `x_0` and the flow rate `x_1`, the steady balance of
    advection, diffusion, and a little wall cooling is
    `g_i = v (y_i - y_{i-1}) - 0.05 (y_{i-1} - 2 y_i + y_{i+1}) + 0.01 y_i - q_i`,
    with cold inflow and a heater deposit `q_i` centered at the heater. The fluid
    is cold upstream of the heater and warm after it, so the solution is a sharp
    step that slides along the pipe with the heater. The manifold has only two
    dimensions, but a linear basis needs about one mode per step position.
    """
    stencil, inside = _chain(points, (-1, 0, 1))
    margin = points / 8

    def entry(
        x: Float[Array, " 2"], i: Int[Array, ""], near: Float[Array, " 3"]
    ) -> Float[Array, ""]:
        # The inflow before the first point is cold, and the clipped index past
        # the last point reads that point, so the outflow has no gradient.
        upstream = near[0] * inside[i, 0]
        here, downstream = near[1], near[2]
        spot = margin + x[0] * (points - 2 * margin)
        speed = 1.0 + x[1]
        heat = jnp.exp(-(((i - spot) / 1.5) ** 2))
        diffusion = upstream - 2.0 * here + downstream
        return speed * (here - upstream) - 0.05 * diffusion + 0.01 * here - heat

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0, 0.0], [1.0, 1.0]),
        sx.box([-1.0] * points, [2.0] * points),
    )


def buckling_beam(points: int = 24) -> ImplicitSystem:
    """A column pressed from its ends, which buckles past a critical load.

    With the deflection `y`, the load `x_0` as a fraction of the critical load,
    and a small sideways push `x_1`, the beam balances bending against the load
    with a cubic geometric term,
    `g = K y / l^2 + x_0 D y / l + y^3 - x_1`,
    where `K` is the fourth difference, `D` the second difference, and `l` the
    lowest eigenvalue of `-D`. The ends are pinned, so the lowest bending mode has
    stiffness `1 - x_0`. Past `x_0 = 1` the straight beam loses that stiffness,
    the Jacobian turns singular, and two bowed shapes appear. The fourth
    difference cancels in single precision on a fine grid, so the beam keeps to a
    few dozen points.
    """
    offsets = np.arange(-2, 3)
    index = np.arange(points)[:, None] + offsets[None, :]
    # The supports sit one step past each end and hold the beam at zero. A pinned
    # end has no curvature, so the point beyond a support mirrors the first point.
    sign = np.where((index == -1) | (index == points), 0.0, 1.0)
    sign = np.where((index == -2) | (index == points + 1), -1.0, sign)
    stencil = jnp.asarray(np.clip(index, 0, points - 1))
    signs = jnp.asarray(sign)
    lowest = 2.0 - 2.0 * np.cos(np.pi / (points + 1))

    def entry(
        x: Float[Array, " 2"], i: Int[Array, ""], near: Float[Array, " 5"]
    ) -> Float[Array, ""]:
        far_left, left, here, right, far_right = near * signs[i]
        second = left - 2.0 * here + right
        fourth = far_left - 4.0 * left + 6.0 * here - 4.0 * right + far_right
        return fourth / lowest**2 + x[0] * second / lowest + here**3 - x[1]

    return ImplicitSystem(
        sx.StencilResidual(stencil=stencil, entry=entry),
        sx.box([0.0, -0.05], [2.0, 0.05]),
        sx.box([-2.0] * points, [2.0] * points),
    )


class _Network(NamedTuple):
    """A wiring graph as padded neighbor slots per node."""

    stencil: Int[Array, "m w"]
    edge: Int[Array, "m w"]
    """Edge index of each slot, `-1` for the node itself and for padding."""

    edges: np.ndarray
    """Endpoint pairs, one row per edge."""


def _network(rows: int, cols: int, long_wires: int, seed: int) -> _Network:
    """A grid of nodes wired to their neighbors, plus a few random long wires."""
    rng = np.random.default_rng(seed)
    nodes = rows * cols
    pairs = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                pairs.append((node, node + 1))
            if r + 1 < rows:
                pairs.append((node, node + cols))
    for _ in range(long_wires):
        a, b = rng.choice(nodes, size=2, replace=False)
        pairs.append((int(a), int(b)))
    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(nodes)]
    for index, (a, b) in enumerate(pairs):
        neighbors[a].append((b, index))
        neighbors[b].append((a, index))
    width = 1 + max(len(slots) for slots in neighbors)
    stencil = np.zeros((nodes, width), dtype=int)
    edge = -np.ones((nodes, width), dtype=int)
    for node, slots in enumerate(neighbors):
        stencil[node] = node
        for slot, (other, index) in enumerate(slots, start=1):
            stencil[node, slot] = other
            edge[node, slot] = index
    return _Network(jnp.asarray(stencil), jnp.asarray(edge), np.asarray(pairs))


def resistor_network(rows: int, cols: int, sources: int = 4, seed: int = 0) -> ImplicitSystem:
    """A mesh of resistors driven by current injected at a few nodes.

    Kirchhoff's current law at node `i` gives
    `g_i = sum_j w_ij (y_i - y_j) + 0.1 y_i - (E x)_i`, where the `0.1 y_i` term is
    a small leak to ground. It is the graph Laplacian in place of the rod's chain,
    affine in `y` and `x`, so the voltages are an exact linear map of the sources.
    The long wires make the Jacobian sparse without being banded.
    """
    network = _network(rows, cols, long_wires=max(1, rows * cols // 16), seed=seed)
    rng = np.random.default_rng(seed + 1)
    conductance = rng.uniform(0.5, 2.0, len(network.edges))
    weight = jnp.asarray(np.where(np.asarray(network.edge) >= 0, conductance[network.edge], 0.0))
    nodes = rows * cols
    injection = np.zeros((nodes, sources))
    for k, node in enumerate(rng.choice(nodes, size=sources, replace=False)):
        injection[node, k] = 1.0
    injection_array = jnp.asarray(injection)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " w"]
    ) -> Float[Array, ""]:
        here = near[0]
        return jnp.sum(weight[i] * (here - near)) + 0.1 * here - x @ injection_array[i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=network.stencil, entry=entry),
        sx.box([-1.0] * sources, [1.0] * sources),
        sx.box([-5.0] * nodes, [5.0] * nodes),
    )


def diode_network(rows: int, cols: int, sources: int = 4, seed: int = 0) -> ImplicitSystem:
    """The resistor mesh with a diode in place of every third wire.

    A diode from node `a` to node `b` carries `I_s (exp((y_a - y_b) / V_T) - 1)`,
    the Shockley law. The exponential makes the voltages a curved function of the
    sources, and the nonlinearity sits only on the diode wires.
    """
    network = _network(rows, cols, long_wires=max(1, rows * cols // 16), seed=seed)
    rng = np.random.default_rng(seed + 1)
    conductance = rng.uniform(0.5, 2.0, len(network.edges))
    is_diode = np.arange(len(network.edges)) % 3 == 0
    edge = np.asarray(network.edge)
    stencil = np.asarray(network.stencil)
    wired = edge >= 0
    resistor = jnp.asarray(np.where(wired & ~is_diode[edge], conductance[edge], 0.0))
    # A slot's sign is +1 where its node is the diode's anode and -1 at the cathode.
    node = np.arange(stencil.shape[0])[:, None]
    anode = np.asarray(network.edges)[edge, 0] == node
    sign = jnp.asarray(np.where(wired & is_diode[edge], np.where(anode, 1.0, -1.0), 0.0))
    nodes = rows * cols
    injection = np.zeros((nodes, sources))
    for k, source in enumerate(rng.choice(nodes, size=sources, replace=False)):
        injection[source, k] = 1.0
    injection_array = jnp.asarray(injection)

    def entry(
        x: Float[Array, " k"], i: Int[Array, ""], near: Float[Array, " w"]
    ) -> Float[Array, ""]:
        here = near[0]
        drop = here - near
        diode = sign[i] * 0.05 * (jnp.exp(sign[i] * drop / 0.5) - 1.0)
        leak = 0.1 * here
        return jnp.sum(resistor[i] * drop + diode) + leak - x @ injection_array[i]

    return ImplicitSystem(
        sx.StencilResidual(stencil=network.stencil, entry=entry),
        sx.box([-1.0] * sources, [1.0] * sources),
        sx.box([-5.0] * nodes, [5.0] * nodes),
    )


def _inverter(
    voltage: Float[Array, ""], gain: Float[Array, ""], bias: Float[Array, ""]
) -> Float[Array, ""]:
    """An inverter's output, high for a low input and low for a high one."""
    return 0.5 * (1.0 - jnp.tanh(gain * (voltage - 0.5 - bias)))


def latch_ring(latches: int = 4) -> ImplicitSystem:
    """A ring of memory latches, each a pair of cross-coupled inverters.

    Node `2k` and node `2k + 1` of latch `k` drive each other,
    `g_a = y_a - f(y_b) + c (2 y_a - y_a' - y_a'')` and `g_b = y_b - f(y_a)`, with a
    weak coupling `c` to the neighboring latches. With the loop gain `x_0` above
    two, each latch holds a stored zero or a stored one, so the ring has many
    solutions, and they merge at a fold as the bias `x_1` is swept.
    """
    nodes = 2 * latches
    stencil = np.zeros((nodes, 4), dtype=int)
    for k in range(latches):
        a, b = 2 * k, 2 * k + 1
        stencil[a] = (a, b, 2 * ((k - 1) % latches), 2 * ((k + 1) % latches))
        stencil[b] = (b, a, b, b)

    def entry(
        x: Float[Array, " 2"], i: Int[Array, ""], near: Float[Array, " 4"]
    ) -> Float[Array, ""]:
        here, partner, before, after = near
        coupling = jnp.where(i % 2 == 0, 0.05 * (2.0 * here - before - after), 0.0)
        return here - _inverter(partner, x[0], x[1]) + coupling

    return ImplicitSystem(
        sx.StencilResidual(stencil=jnp.asarray(stencil), entry=entry),
        sx.box([3.0, -0.1], [6.0, 0.1]),
        sx.box([0.0] * nodes, [1.0] * nodes),
    )
