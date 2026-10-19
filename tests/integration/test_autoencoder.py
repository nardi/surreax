"""Tests for the reduced-order model on a learned manifold."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import surreax as sx
from surreax import implicit
from tests import implicit_systems

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
    form = sx.ChebyshevPolynomial(degree=3)
    surrogate = sx.fit_implicit(_radiating_rod, _DOMAIN, _method(), form, samples=256)
    assert isinstance(surrogate, sx.DecodedSurrogate)
    assert _relative_residual(surrogate) < 0.15

    point = jnp.array([0.3, -0.4, 0.2, 0.1])
    plain = surrogate(point)
    assert jnp.allclose(eqx.filter_jit(lambda m, z: m(z))(surrogate, point), plain, atol=1e-5)
    assert jax.vmap(surrogate)(jnp.stack([point, point])).shape == (2, 32)
    assert jax.grad(lambda z: surrogate(z)[0])(point).shape == (4,)


def test_learned_manifold_beats_a_linear_basis_on_a_moving_front() -> None:
    # The pipe's warm region starts at a movable heater, so the solution is a step
    # that slides along the pipe, which a linear basis spreads over many modes.
    pipe = implicit_systems.hot_fluid_pipe(64)
    points = jax.random.uniform(jax.random.key(4), (64, 2))
    base = jnp.mean(jax.vmap(lambda x: jnp.linalg.norm(pipe.residual(x, jnp.zeros(64))))(points))

    def relative(solver: eqx.Module) -> float:
        left = jax.vmap(lambda x: jnp.linalg.norm(pipe.residual(x, solver(x))))(points)
        return float(jnp.mean(left) / base)

    linear = sx.reduced_solver(
        pipe.residual,
        pipe.domain,
        sx.ReducedSolve(
            unknown_domain=pipe.unknown_domain,
            rank=8,
            snapshots=64,
            newton_steps=3,
            reduced_steps=4,
        ),
        samples=128,
    )
    learned = sx.autoencoder_solver(
        pipe.residual,
        pipe.domain,
        sx.AutoencoderSolve(
            unknown_dimension=64,
            rank=2,
            width=64,
            snapshots=16,
            newton_steps=3,
        ),
    )
    # Two latents and a quarter of the snapshots beat eight linear modes.
    assert relative(learned) < 0.35
    assert relative(learned) < relative(linear) / 2


def test_recommender_routes_by_the_linear_energy() -> None:
    rod = implicit_systems.heated_rod(64)
    pipe = implicit_systems.hot_fluid_pipe(64)

    def request(system: implicit_systems.ImplicitSystem, rank: int) -> sx.ReducedSolve:
        return sx.ReducedSolve(
            unknown_domain=system.unknown_domain,
            rank=rank,
            snapshots=32,
            newton_steps=3,
        )

    # Four heaters span the rod's manifold exactly, so the linear model stays.
    rod_request = request(rod, 4)
    assert sx.recommend_reduced_method(rod.residual, rod.domain, rod_request) is rod_request
    # Two linear modes leave most of the moving front out.
    routed = sx.recommend_reduced_method(pipe.residual, pipe.domain, request(pipe, 2))
    assert isinstance(routed, sx.AutoencoderSolve)
    assert routed.snapshots == 32
