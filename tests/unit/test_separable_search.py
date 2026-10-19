"""The worst-case search per block of a separable fit."""

import jax
import jax.numpy as jnp
import pytest

import surreax as sx


@pytest.mark.slow
def test_a_separable_fit_finds_its_worst_block() -> None:
    def target(x: jax.Array) -> jax.Array:
        # A narrow bump at 0.6 in the first block, which a quadratic misses.
        bump = jnp.exp(-(((x[0] - 0.6) / 0.05) ** 2))
        return jnp.stack([x[0] ** 2 + bump, 3.0 * x[1] - x[2]])

    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    composite = sx.fit_separable(
        target, domain, sx.ChebyshevPolynomial(degree=2), samples=128, worst_case_starts=8
    )
    blocks = [float(block.report.worst_case) for block in composite.blocks]
    assert float(composite.report.worst_case) > 0.5
    assert abs(float(composite.report.worst_case) - jnp.sqrt(sum(b**2 for b in blocks))) < 1e-5
    assert abs(float(composite.report.worst_case_point[0]) - 0.6) < 0.1
    skipped = sx.fit_separable(
        target, domain, sx.ChebyshevPolynomial(degree=2), samples=128, worst_case_starts=0
    )
    assert float(skipped.report.worst_case) == 0.0
