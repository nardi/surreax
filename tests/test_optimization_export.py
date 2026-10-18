"""Constrained optimization over exported surrogates, checked on the full systems.

Each test fits a surrogate to the sensor temperatures of a reduced solver, exports
it, solves a linear or mixed-integer program with `scipy.optimize.milp`, and then
solves the full implicit system at the minimizer. The surrogate is only an
approximation, so the full system meets the constraints up to the fit error, and
the cost lands near the best cost over a grid of full solves.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import surreax as sx
from tests import implicit_systems
from tests.optimization_encodings import minimize_convex, minimize_relu
from tests.piecewise_cases import Case, build

_POINTS = 32

_ROD_SENSORS = jnp.array([15, 7])
_ROD_TARGET = np.array([0.8, 0.8])

_THERMOSTAT_SENSORS = jnp.array([22, 4])
_THERMOSTAT_LIMIT = 0.7
_THERMOSTAT_FLOOR = 0.4
_THERMOSTAT_COST = np.array([-0.3, -1.0])


def _full_states(residual: sx.Residual, inputs: np.ndarray) -> np.ndarray:
    """Solve the full residual by Newton from the zero state at every input."""

    def solve(x: jax.Array) -> jax.Array:
        def step(y: jax.Array, _: None) -> tuple[jax.Array, None]:
            jacobian = jax.jacobian(residual, argnums=1)(x, y)
            return y - jnp.linalg.solve(jacobian, residual(x, y)), None

        return jax.lax.scan(step, jnp.zeros(_POINTS), None, length=40)[0]

    states = jax.jit(jax.vmap(solve))(jnp.asarray(inputs))
    # A state that is not a root would make the checks below meaningless.
    residuals = jax.vmap(residual)(jnp.asarray(inputs), states)
    assert float(jnp.max(jnp.abs(residuals))) < 1e-4
    return np.asarray(states)


def _grid(domain: sx.Box, counts: tuple[int, ...]) -> np.ndarray:
    axes = [
        np.linspace(low, high, count)
        for low, high, count in zip(domain.lower, domain.upper, counts, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


@pytest.fixture(scope="module")
def rod() -> Case:
    return build("rod")


@pytest.mark.parametrize(
    "form",
    [
        sx.MaxAffineForm(pieces=16, restarts=4, iterations=25),
        sx.ConvexNetworkForm(width=12, depth=2),
    ],
    ids=["max_affine", "convex_network"],
)
def test_the_least_heater_power_plan_of_a_convex_surrogate_holds_on_the_rod(
    rod: Case, form: sx.MaxAffineForm | sx.ConvexNetworkForm
) -> None:
    # Radiation makes the temperature a concave function of the heater powers, so
    # the negated temperature is convex and "at least the target" is a convex set.
    surrogate = sx.fit(
        lambda x: -rod.solver(x)[_ROD_SENSORS],
        rod.domain,
        form,
        samples=512,
        worst_case_starts=0,
    )
    lower, upper = np.asarray(rod.domain.lower), np.asarray(rod.domain.upper)
    export = sx.export(surrogate)
    assert isinstance(export, sx.MaxAffineExport | sx.ConvexNetworkExport)
    power = minimize_convex(export, np.ones(2), lower, upper, -_ROD_TARGET)

    # The program is solved for the surrogate itself, whatever the fit error is.
    assert np.all(power >= lower - 1e-9) and np.all(power <= upper + 1e-9)
    assert np.all(-np.asarray(surrogate(jnp.asarray(power))) >= _ROD_TARGET - 1e-4)

    # The full rod at that power meets the target up to the fit error.
    sensors = _full_states(rod.residual, power[None])[0][np.asarray(_ROD_SENSORS)]
    assert np.all(sensors >= _ROD_TARGET - 0.04)

    # The cheapest power on a grid of full solves is a reference for the cost.
    grid = _grid(rod.domain, (41, 41))
    temperatures = _full_states(rod.residual, grid)[:, np.asarray(_ROD_SENSORS)]
    feasible = np.all(temperatures >= _ROD_TARGET, axis=1)
    cheapest = grid[feasible].sum(axis=1).min()
    assert abs(power.sum() - cheapest) < 0.1 * cheapest


class ThermostatOracle:
    """A cheap solver of the thermostat rod, with the system it approximates."""

    def __init__(
        self,
        system: implicit_systems.ImplicitSystem,
        field: Callable[[jax.Array], jax.Array],
    ) -> None:
        self.domain = system.domain
        self.residual = system.residual
        self.field = field


def _reduced_oracle() -> ThermostatOracle:
    case = build("thermostat")
    return ThermostatOracle(implicit_systems.thermostat_rod(_POINTS), case.solver)


def _autoencoder_oracle() -> ThermostatOracle:
    # No full state is solved, the manifold is learned from the residual alone.
    system = implicit_systems.thermostat_rod(_POINTS)
    method = sx.AutoencoderSolve(form=sx.Affine(), unknown_dimension=_POINTS, rank=3, steps=800)
    solver = sx.autoencoder_solver(system.residual, system.domain, method)
    assert float(solver.training_residual) < 0.05
    return ThermostatOracle(system, solver)


@pytest.fixture(scope="module", params=["reduced", "autoencoder"])
def thermostat(request: pytest.FixtureRequest) -> ThermostatOracle:
    return _reduced_oracle() if request.param == "reduced" else _autoencoder_oracle()


def test_the_mixed_integer_plan_of_a_relu_network_holds_on_the_thermostat_rod(
    thermostat: ThermostatOracle,
) -> None:
    # The thermostat holds the rod near its setpoint and then lets go, so the
    # temperature is not convex in the ambient and the load, and a ReLU network is
    # needed. The plan maximizes the load and the ambient subject to a process
    # window: the patch must stay below a limit and the far end above a floor.
    # The oracle is a reduced solver on a linear basis of solved states in one
    # run and an autoencoder trained on the residual alone in the other.
    surrogate = sx.fit(
        lambda x: thermostat.field(x)[_THERMOSTAT_SENSORS],
        thermostat.domain,
        sx.ReluNetworkForm(width=16, depth=2),
        samples=512,
        worst_case_starts=0,
    )
    lower, upper = np.asarray(thermostat.domain.lower), np.asarray(thermostat.domain.upper)
    output_lower = np.array([-np.inf, _THERMOSTAT_FLOOR])
    output_upper = np.array([_THERMOSTAT_LIMIT, np.inf])
    export = sx.export(surrogate)
    assert isinstance(export, sx.ReluNetworkExport)
    plan = minimize_relu(export, _THERMOSTAT_COST, lower, upper, output_lower, output_upper)

    # The program is exact for the surrogate: it meets both bounds, and no point
    # of a fine grid of the surrogate does better.
    predicted = np.asarray(surrogate(jnp.asarray(plan)))
    assert predicted[0] <= _THERMOSTAT_LIMIT + 1e-4
    assert predicted[1] >= _THERMOSTAT_FLOOR - 1e-4
    grid = _grid(thermostat.domain, (181, 81))
    values = np.asarray(jax.vmap(surrogate)(jnp.asarray(grid)))
    window = (values[:, 0] <= _THERMOSTAT_LIMIT) & (values[:, 1] >= _THERMOSTAT_FLOOR)
    assert _THERMOSTAT_COST @ plan <= (grid[window] @ _THERMOSTAT_COST).min() + 1e-6

    # The full rod at the plan meets the window up to the fit error.
    sensors = _full_states(thermostat.residual, plan[None])[0][np.asarray(_THERMOSTAT_SENSORS)]
    assert sensors[0] <= _THERMOSTAT_LIMIT + 0.03
    assert sensors[1] >= _THERMOSTAT_FLOOR - 0.03

    # The best plan on a grid of full solves is a reference for the objective.
    coarse = _grid(thermostat.domain, (46, 21))
    states = _full_states(thermostat.residual, coarse)[:, np.asarray(_THERMOSTAT_SENSORS)]
    inside = (states[:, 0] <= _THERMOSTAT_LIMIT) & (states[:, 1] >= _THERMOSTAT_FLOOR)
    best = (coarse[inside] @ _THERMOSTAT_COST).min()
    assert abs(_THERMOSTAT_COST @ plan - best) < 0.1
