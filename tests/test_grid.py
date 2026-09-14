"""Tests for the regular-grid form and its export."""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.interpolate import RegularGridInterpolator

import surreax as sx


def _smooth(x: jax.Array) -> jax.Array:
    return (jnp.sin(2 * x[0]) + x[1] ** 2)[None]


def test_grid_error_falls_as_the_resolution_grows() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    coarse = sx.fit(_smooth, domain, sx.Grid(resolution=4), samples=512, worst_case_starts=0)
    fine = sx.fit(_smooth, domain, sx.Grid(resolution=16), samples=512, worst_case_starts=0)
    assert float(fine.report.value_mean) < float(coarse.report.value_mean)
    assert float(fine.report.value_mean) < 0.02


def test_grid_surrogate_survives_jit_vmap_and_grad() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(_smooth, domain, sx.Grid(resolution=8), samples=512, worst_case_starts=0)
    point = jnp.array([0.3, -0.4])
    plain = surrogate(point)
    assert jnp.allclose(jax.jit(lambda m, z: m(z))(surrogate, point), plain)
    assert jax.vmap(surrogate)(jnp.stack([point, point])).shape == (2, 1)
    assert jax.grad(lambda z: surrogate(z)[0])(point).shape == (2,)


def test_grid_export_matches_the_surrogate() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(_smooth, domain, sx.Grid(resolution=8), samples=512, worst_case_starts=0)
    export = sx.export(surrogate)
    assert isinstance(export, sx.GridExport)
    assert tuple(axis.shape[0] for axis in export.axes) == (8, 8)
    assert export.values.shape == (1, 8, 8)

    interpolate = RegularGridInterpolator(
        export.axes, export.values[0], method="linear", bounds_error=False, fill_value=None
    )
    points = jax.random.uniform(jax.random.key(2), (30, 2), minval=-1.0, maxval=1.0)
    by_hand = jax.vmap(lambda x: interpolate(x[None])[0])(points)
    assert jnp.allclose(by_hand, jax.vmap(surrogate)(points)[:, 0], atol=1e-4)


def test_grid_export_refuses_a_reduced_input() -> None:
    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    surrogate = sx.fit(
        _smooth,
        domain,
        sx.Grid(resolution=6),
        samples=256,
        worst_case_starts=0,
        reduce_inputs=sx.pca_inputs(rank=2),
    )
    with pytest.raises(ValueError, match="identity input reduction"):
        sx.export(surrogate)
