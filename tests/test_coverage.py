"""Coverage honesty: reading how far a query lies from the data a domain came from."""

import equinox as eqx
import jax
import jax.numpy as jnp

import surreax as sx


def _diagonal(count: int = 200) -> jax.Array:
    """Points along the diagonal of the unit square, with a little noise across it."""
    along = jax.random.uniform(jax.random.key(0), (count,))
    across = 0.01 * jax.random.normal(jax.random.key(1), (count,))
    return jnp.stack([along + across, along - across], axis=1)


def test_an_empty_corner_of_the_box_is_not_covered() -> None:
    domain = sx.domain_from_data(_diagonal())
    assert domain.basis.shape == (1, 2)
    assert float(domain.coverage(jnp.array([0.5, 0.5]))) <= 1.0
    # The corner lies inside the bounding box but far from the data.
    assert float(domain.coverage(jnp.array([0.95, 0.05]))) > 1.0
    assert float(sx.coverage(sx.box_from_data(_diagonal()), jnp.array([0.95, 0.05]))) == 0.0


def test_the_gap_between_two_clusters_is_not_covered() -> None:
    noise = 0.05 * jax.random.normal(jax.random.key(2), (100, 2))
    left, right = noise + jnp.array([-1.0, 0.0]), noise[:50] + jnp.array([1.0, 0.0])
    clusters = jnp.concatenate([left, right])
    domain = sx.domain_from_data(clusters)
    assert float(domain.coverage(jnp.array([-1.0, 0.0]))) <= 1.0
    assert float(domain.coverage(jnp.array([0.0, 0.0]))) > 1.0
    points = jnp.array([[-1.0, 0.0], [0.0, 0.0]])
    jitted = eqx.filter_jit(jax.vmap(domain.coverage))(points)
    assert jnp.allclose(jitted, jax.vmap(domain.coverage)(points))


def test_a_box_covers_its_inside_and_grows_outside() -> None:
    region = sx.box([0.0, 0.0], [1.0, 2.0])
    assert float(sx.coverage(region, jnp.array([0.5, 1.0]))) == 0.0
    assert float(sx.coverage(region, jnp.array([1.5, 1.0]))) == 1.5


def test_a_fit_reports_coverage_and_the_surrogate_reads_it() -> None:
    data = _diagonal()

    def target(x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)[None]

    surrogate = sx.fit_data(data, jax.vmap(target)(data), sx.ChebyshevPolynomial(degree=2))
    assert isinstance(surrogate.domain, sx.DataDomain)
    assert float(surrogate.report.coverage_max) < 2.0
    assert float(surrogate.coverage(jnp.array([0.95, 0.05]))) > 1.0
    # Fitting over the inferred domain samples its whole box, so the report
    # shows held-out points that lie away from the data.
    fitted = sx.fit(
        target, sx.domain_from_data(data), sx.Affine(), samples=64, worst_case_starts=0
    )
    assert float(fitted.report.coverage_max) > 1.0
