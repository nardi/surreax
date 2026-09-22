"""Tests for the reduced-order model on a learned manifold."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax import implicit

_GRID = jnp.linspace(0.0, 1.0, 32)
_MODES = jnp.stack(
    [
        jnp.sin(2 * jnp.pi * _GRID),
        jnp.cos(2 * jnp.pi * _GRID),
        jnp.sin(4 * jnp.pi * _GRID),
        jnp.cos(4 * jnp.pi * _GRID),
    ]
)
_DOMAIN = sx.box([-1.0] * 4, [1.0] * 4)


def _radiating_rod(x: jax.Array, y: jax.Array) -> jax.Array:
    # A rod heated by four source modes that also sheds heat with a cubic term,
    # so the steady temperature is a curved function of the sources.
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y + 0.3 * y**3 - x @ _MODES


def _relative_residual(surrogate: eqx.Module) -> float:
    points = jax.random.uniform(jax.random.key(4), (64, 4), minval=-1.0, maxval=1.0)
    left = jax.vmap(lambda x: jnp.linalg.norm(_radiating_rod(x, surrogate(x))))(points)
    base = jax.vmap(lambda x: jnp.linalg.norm(_radiating_rod(x, jnp.zeros(32))))(points)
    return float(jnp.mean(left) / jnp.mean(base))


def _method(**overrides: object) -> sx.AutoencoderSolve:
    settings: dict[str, object] = {
        "form": sx.ChebyshevPolynomial(degree=3),
        "unknown_dimension": 32,
        "rank": 4,
        "steps": 800,
    }
    settings.update(overrides)
    return sx.AutoencoderSolve(**settings)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def solver() -> sx.AutoencoderSolver:
    return sx.autoencoder_solver(_radiating_rod, _DOMAIN, _method())


def test_training_needs_no_solved_states(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the training solved the full residual")

    # Any full-space solve would go through these, so forbid them.
    monkeypatch.setattr(implicit, "_solve_independently_with", refuse)
    monkeypatch.setattr(implicit, "_newton_step", refuse)
    trained = sx.autoencoder_solver(_radiating_rod, _DOMAIN, _method(steps=50))
    assert trained.training_residual.shape == ()


def test_training_reaches_a_small_residual(solver: sx.AutoencoderSolver) -> None:
    assert float(solver.training_residual) < 0.1
    assert _relative_residual(solver) < 0.1


def test_encoder_inverts_the_decoder(solver: sx.AutoencoderSolver) -> None:
    assert float(solver.consistency_error) < 0.1
    latent = solver.latent(jnp.array([0.4, -0.2, 0.6, 0.0]))
    decoded = solver.autoencoder.decode(latent)
    assert jnp.allclose(solver.autoencoder.encode(decoded), latent, atol=0.1)


def test_latent_derivative_matches_finite_differences(solver: sx.AutoencoderSolver) -> None:
    point = jnp.array([0.5, -0.3, 0.7, 0.1])
    implicit_jacobian = jax.jacobian(solver.latent)(point)
    eps = 1e-3
    finite = jnp.stack(
        [
            (solver.latent(point.at[i].add(eps)) - solver.latent(point.at[i].add(-eps)))
            / (2 * eps)
            for i in range(4)
        ],
        axis=1,
    )
    assert jnp.allclose(implicit_jacobian, finite, atol=1e-3)


def test_fit_implicit_distills_a_decoded_surrogate() -> None:
    surrogate = sx.fit_implicit(_radiating_rod, _DOMAIN, _method(), samples=256)
    assert isinstance(surrogate, sx.DecodedSurrogate)
    assert _relative_residual(surrogate) < 0.15

    point = jnp.array([0.3, -0.4, 0.2, 0.1])
    plain = surrogate(point)
    assert jnp.allclose(eqx.filter_jit(lambda m, z: m(z))(surrogate, point), plain)
    assert jax.vmap(surrogate)(jnp.stack([point, point])).shape == (2, 32)
    assert jax.grad(lambda z: surrogate(z)[0])(point).shape == (4,)
