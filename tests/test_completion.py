"""Equality constraints satisfied by construction, in the manner of DC3."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx

_DOMAIN = sx.box([0.5, 0.0], [2.0, 1.5])


def _inflow(x: jax.Array) -> jax.Array:
    """An inflow outside the polynomial basis.

    A linear least-squares fit keeps any linear relation among its outputs whose
    right-hand side lies in its basis, so an inflow in the basis would balance
    without any help.
    """
    return jnp.exp(x[0])


def _balanced(x: jax.Array) -> jax.Array:
    """Three flows that always sum to the inflow."""
    first = x[0] * jnp.sin(x[1]) ** 2 / 2
    second = x[0] * jnp.cos(x[1]) * x[1] / 3
    return jnp.stack([first, second, _inflow(x) - first - second])


def _balance(x: jax.Array, y: jax.Array) -> jax.Array:
    return (jnp.sum(y) - _inflow(x))[None]


def _held_out() -> jax.Array:
    unit = jax.random.uniform(jax.random.key(4), (32, 2))
    return _DOMAIN.sample_unit(unit)


def test_a_balance_holds_exactly_while_the_free_outputs_stay_put() -> None:
    constraint = sx.Constraint(_balance, dependent=(2,))
    form = sx.ChebyshevPolynomial(degree=3)
    completed = sx.fit_constrained(_balanced, _DOMAIN, constraint, form, samples=256)
    plain = sx.fit(_balanced, _DOMAIN, form, samples=256, worst_case_starts=0)
    points = _held_out()
    assert float(jnp.max(sx.constraint_residuals(completed, constraint, points))) < 1e-5
    assert float(jnp.max(sx.constraint_residuals(plain, constraint, points))) > 1e-4
    # The balance is affine in the dependent flow, so one Newton step solves it.
    assert completed.steps == 1
    free, base_free = jax.vmap(completed)(points)[:, :2], jax.vmap(completed.base)(points)[:, :2]
    assert jnp.allclose(free, base_free)
    assert float(completed.report.value_max) < 0.1
    assert jnp.allclose(eqx.filter_jit(completed)(points[0]), completed(points[0]))


def _on_a_circle(x: jax.Array) -> jax.Array:
    radius = jnp.sqrt(1.0 + x[0] ** 2)
    return jnp.stack([radius * jnp.cos(x[1]), radius * jnp.sin(x[1])])


def _circle(x: jax.Array, y: jax.Array) -> jax.Array:
    return (y[0] ** 2 + y[1] ** 2 - 1.0 - x[0] ** 2)[None]


@pytest.mark.parametrize(
    ("target", "constraint"),
    [(_balanced, sx.Constraint(_balance, dependent=(2,))),
     (_on_a_circle, sx.Constraint(_circle, dependent=(1,)))],
    ids=["affine", "nonlinear"],
)
def test_the_completed_jacobian_matches_finite_differences(
    target: object, constraint: sx.Constraint
) -> None:
    domain = sx.box([0.5, 0.3], [2.0, 1.2])
    completed = sx.fit_constrained(
        target, domain, constraint, sx.ChebyshevPolynomial(degree=3), samples=128
    )
    point = jnp.array([1.2, 0.7])
    exact = jax.jacobian(completed)(point)
    step = 1e-3
    columns = [
        (completed(point.at[i].add(step)) - completed(point.at[i].add(-step))) / (2 * step)
        for i in range(2)
    ]
    assert jnp.allclose(exact, jnp.stack(columns, axis=1), atol=2e-2)
    assert float(jnp.abs(constraint.function(point, completed(point))[0])) < 1e-4


def test_a_dependent_choice_the_constraint_does_not_fix_is_rejected() -> None:
    def partial_balance(x: jax.Array, y: jax.Array) -> jax.Array:
        return (y[0] + y[1] - x[0])[None]

    with pytest.raises(ValueError, match="singular"):
        sx.fit_constrained(
            _balanced, _DOMAIN, sx.Constraint(partial_balance, dependent=(2,)), sx.Affine()
        )
    with pytest.raises(ValueError, match="distinct"):
        sx.fit_constrained(_balanced, _DOMAIN, sx.Constraint(_balance, dependent=(1, 1)))


def test_an_implicit_fit_satisfies_its_first_equation_exactly() -> None:
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.stack([y[0] + y[1] - x[0], y[0] - y[1] ** 3 - x[1]])

    def first_equation(x: jax.Array, y: jax.Array) -> jax.Array:
        return residual(x, y)[:1]

    domain = sx.box([0.0, -0.5], [1.0, 0.5])
    inner = sx.InverseMap(
        form=sx.ChebyshevPolynomial(degree=3), unknown_domain=sx.box([-2.0, -2.0], [2.0, 2.0])
    )
    method = sx.Completed(method=inner, constraint=sx.Constraint(first_equation, dependent=(0,)))
    surrogate = sx.fit_implicit(residual, domain, method, samples=256)
    plain = sx.fit_implicit(residual, domain, inner, samples=256)
    assert isinstance(surrogate, sx.CompletedSurrogate)
    points = domain.sample_unit(jax.random.uniform(jax.random.key(5), (32, 2)))
    left = sx.constraint_residuals(surrogate, method.constraint, points)
    assert float(jnp.max(left)) < 1e-5
    # The full residual stays about as small as the inner method leaves it.
    assert float(surrogate.report.value_mean) < 2.0 * float(plain.report.value_mean)


def test_a_given_step_count_overrides_the_detected_one() -> None:
    domain = sx.box([0.5, 0.3], [2.0, 1.2])
    constraint = sx.Constraint(_circle, dependent=(1,), steps=4)
    completed = sx.fit_constrained(
        _on_a_circle, domain, constraint, sx.ChebyshevPolynomial(degree=3), samples=128
    )
    assert completed.steps == 4
    detected = sx.fit_constrained(
        _on_a_circle,
        domain,
        sx.Constraint(_circle, dependent=(1,)),
        sx.ChebyshevPolynomial(degree=3),
        samples=128,
    )
    assert detected.steps == 8
