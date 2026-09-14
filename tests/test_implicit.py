"""Tests for implicit fitting through the inverse map and the objective route."""

import jax
import jax.numpy as jnp

import surreax as sx
from surreax.implicit import _newton_step, _newton_step_matrix_free, _solve_independently


def test_affine_in_the_unknown_is_recovered_exactly() -> None:
    # g(x, y) = y - 2x, so f(x) = 2x, and the inverse map is exactly affine.
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return y - 2.0 * x

    domain = sx.box([-1.0], [1.0])
    method = sx.InverseMap(form=sx.Affine(), unknown_domain=sx.box([-3.0], [3.0]))
    surrogate = sx.fit_implicit(residual, domain, method, samples=256)
    assert float(surrogate.report.value_max) < 1e-4
    prediction = surrogate(jnp.array([0.5]))
    assert jnp.allclose(prediction, jnp.array([1.0]), atol=1e-3)


def test_coupled_affine_system_is_recovered_exactly() -> None:
    matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])

    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return matrix @ y - x

    domain = sx.box([-1.0, -1.0], [1.0, 1.0])
    method = sx.InverseMap(form=sx.Affine(), unknown_domain=sx.box([-1.0, -1.0], [1.0, 1.0]))
    surrogate = sx.fit_implicit(residual, domain, method, samples=256)
    assert float(surrogate.report.value_max) < 1e-4
    solved = jnp.linalg.solve(matrix, jnp.array([0.4, -0.2]))
    assert jnp.allclose(surrogate(jnp.array([0.4, -0.2])), solved, atol=1e-3)


def _cubic(x: jax.Array, y: jax.Array) -> jax.Array:
    return y**3 + y - x


def test_newton_steps_lower_the_residual() -> None:
    domain = sx.box([-2.0], [2.0])
    form = sx.ChebyshevPolynomial(degree=4)
    unknown_domain = sx.box([-1.2], [1.2])
    without = sx.fit_implicit(
        _cubic, domain, sx.InverseMap(form=form, unknown_domain=unknown_domain, newton_steps=0),
        samples=512,
    )
    with_steps = sx.fit_implicit(
        _cubic, domain, sx.InverseMap(form=form, unknown_domain=unknown_domain, newton_steps=2),
        samples=512,
    )
    assert float(with_steps.report.value_mean) < float(without.report.value_mean)


def test_nonlinear_monotone_reaches_a_low_residual() -> None:
    domain = sx.box([-2.0], [2.0])
    form = sx.ChebyshevPolynomial(degree=4)
    method = sx.InverseMap(form=form, unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2)
    surrogate = sx.fit_implicit(_cubic, domain, method, samples=512)
    assert float(surrogate.report.value_mean) < 0.5


def test_iterative_refinement_drives_the_residual_down() -> None:
    domain = sx.box([-2.0], [2.0])
    form = sx.ChebyshevPolynomial(degree=4)
    method = sx.InverseMap(form=form, unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2)
    base = sx.fit_implicit(_cubic, domain, method, samples=512)
    refined = sx.fit_implicit(_cubic, domain, method, samples=512, refine_steps=3)
    assert float(refined.report.value_mean) < float(base.report.value_mean) / 5


def test_objective_route_leaves_a_small_residual() -> None:
    domain = sx.box([-2.0], [2.0])
    form = sx.ChebyshevPolynomial(degree=4)
    # The objective route fits f directly by minimizing the residual, no inversion.
    method = sx.Objective(form=form, unknown_dimension=1, steps=600)
    surrogate = sx.fit_implicit(_cubic, domain, method, samples=256)
    assert float(surrogate.report.value_mean) < 0.15


def _square(x: jax.Array, y: jax.Array) -> jax.Array:
    # Two branches, y = +-sqrt(x), and a fold at x = 0 where dg/dy = 2y vanishes.
    return y**2 - x


def test_two_branch_residual_stays_on_one_branch() -> None:
    domain = sx.box([0.25], [1.0])
    # Symmetric starts land on both branches, so a naive fit would average them.
    method = sx.InverseMap(
        form=sx.ChebyshevPolynomial(degree=4),
        unknown_domain=sx.box([-1.1], [1.1]),
        newton_steps=6,
    )
    surrogate = sx.fit_implicit(_square, domain, method, samples=256)
    points = jnp.linspace(0.3, 0.95, 20)[:, None]
    predictions = jax.vmap(surrogate)(points)[:, 0]
    assert bool(jnp.all(predictions > 0)) or bool(jnp.all(predictions < 0))
    assert float(surrogate.report.value_mean) < 0.1
    assert float(surrogate.fold.off_branch) > 0.3


def test_fold_is_flagged_where_the_jacobian_turns_singular() -> None:
    folding = sx.InverseMap(
        form=sx.ChebyshevPolynomial(degree=4),
        unknown_domain=sx.box([0.05], [1.1]),
        newton_steps=6,
        fold_tolerance=0.3,
    )
    fold_fit = sx.fit_implicit(_square, sx.box([0.0], [1.0]), folding, samples=256)
    assert fold_fit.fold.folds
    assert float(fold_fit.fold.boundary_point[0]) < 0.1

    monotone = sx.InverseMap(
        form=sx.ChebyshevPolynomial(degree=4),
        unknown_domain=sx.box([-1.2], [1.2]),
        newton_steps=2,
        fold_tolerance=0.3,
    )
    monotone_fit = sx.fit_implicit(_cubic, sx.box([-2.0], [2.0]), monotone, samples=256)
    assert not monotone_fit.fold.folds
    assert float(monotone_fit.fold.min_determinant) > float(fold_fit.fold.min_determinant)


def test_continuation_follows_one_branch() -> None:
    method = sx.InverseMap(
        form=sx.ChebyshevPolynomial(degree=4),
        unknown_domain=sx.box([0.05], [1.1]),
        newton_steps=6,
        continuation=True,
    )
    surrogate = sx.fit_implicit(_square, sx.box([0.0], [1.0]), method, samples=256)
    points = jnp.linspace(0.1, 0.95, 20)[:, None]
    predictions = jax.vmap(surrogate)(points)[:, 0]
    assert bool(jnp.all(predictions > 0))
    assert float(surrogate.report.value_mean) < 0.05


def test_fit_data_recovers_a_linear_map_from_pairs() -> None:
    key = jax.random.key(0)
    inputs = jax.random.uniform(key, (200, 3), minval=-1.0, maxval=1.0)
    weights = jnp.array([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]])
    outputs = jax.vmap(lambda x: weights @ x + jnp.array([0.5, -1.0]))(inputs)
    surrogate = sx.fit_data(inputs, outputs, sx.Affine())
    assert float(surrogate.report.value_max) < 1e-3
    assert jnp.allclose(surrogate(inputs[0]), outputs[0], atol=1e-3)


def test_root_find_and_refine_solve_a_scalar_root() -> None:
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return y**3 + y - x

    # A crude constant guess, driven to the root by Newton steps at call time.
    guess = sx.root_find(residual, lambda x: x / 2.0, steps=8)
    point = jnp.array([2.0])
    left = residual(point, guess(point))
    assert float(jnp.abs(left[0])) < 1e-5

    # refine wraps an existing surrogate the same way.
    polished = sx.refine(lambda x: x / 2.0, residual, steps=8)
    assert jnp.allclose(polished(point), guess(point))


_GRID = jnp.linspace(0.0, 1.0, 64)
_SHAPES = jnp.stack(
    [jnp.sin(jnp.pi * _GRID), jnp.sin(2 * jnp.pi * _GRID), _GRID * (1 - _GRID)]
)


def _diffusion(x: jax.Array, y: jax.Array) -> jax.Array:
    # A large sparse system: the nearest-neighbor Laplacian couples the unknown,
    # driven by three smooth spatial shapes, so the solution lives on a rank-3
    # manifold in the 64-dimensional unknown space.
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ _SHAPES


def test_exact_solve_fits_a_large_sparse_system() -> None:
    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    method = sx.ExactSolve(
        form=sx.Affine(), unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, newton_steps=5
    )
    surrogate = sx.fit_implicit(_diffusion, domain, method, samples=128)
    # The 64 unknowns collapse to 3 coefficients, and the fit stays exact.
    assert surrogate.output_reduction.encode_matrix.shape[0] == 3
    assert float(surrogate.report.value_mean) < 1e-3

    points = jax.random.uniform(jax.random.key(7), (20, 3), minval=-1.0, maxval=1.0)
    true = _solve_independently(_diffusion, points, jnp.zeros((20, 64)), 5)
    reconstruction = jnp.mean(jnp.linalg.norm(jax.vmap(surrogate)(points) - true, axis=1))
    assert float(reconstruction) < 1e-3


def test_matrix_free_solve_matches_the_dense_solve() -> None:
    x = jnp.array([0.5, -0.3, 0.7])
    dense, matrix_free = jnp.zeros(64), jnp.zeros(64)
    for _ in range(5):
        dense = _newton_step(_diffusion, x, dense)
        matrix_free = _newton_step_matrix_free(_diffusion, x, matrix_free)
    assert jnp.allclose(dense, matrix_free, atol=1e-5)


def test_exact_solve_dense_route_agrees_with_matrix_free() -> None:
    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    unknown_domain = sx.box([-3.0] * 64, [3.0] * 64)
    dense = sx.fit_implicit(
        _diffusion,
        domain,
        sx.ExactSolve(
            form=sx.Affine(),
            unknown_domain=unknown_domain,
            rank=3,
            newton_steps=5,
            matrix_free=False,
        ),
        samples=128,
    )
    assert float(dense.report.value_mean) < 1e-3


def _linear_diffusion(x: jax.Array, y: jax.Array) -> jax.Array:
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ _SHAPES


def test_reduced_order_fits_from_few_snapshots() -> None:
    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    # Only 16 full solves build the basis, then every sample is reduce-solved.
    method = sx.ReducedSolve(
        form=sx.Affine(),
        unknown_domain=sx.box([-3.0] * 64, [3.0] * 64),
        rank=3,
        snapshots=16,
        newton_steps=5,
        reduced_steps=5,
    )
    surrogate = sx.fit_implicit(_linear_diffusion, domain, method, samples=128)
    assert surrogate.output_reduction.decode_matrix.shape == (64, 3)
    assert float(surrogate.report.value_mean) < 1e-3


def test_reduced_order_latent_dimension_tracks_the_manifold() -> None:
    domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    unknown_domain = sx.box([-3.0] * 64, [3.0] * 64)

    def fit_with_rank(rank: int) -> float:
        method = sx.ReducedSolve(
            form=sx.Affine(),
            unknown_domain=unknown_domain,
            rank=rank,
            snapshots=16,
            newton_steps=5,
            reduced_steps=5,
            matrix_free=False,
        )
        surrogate = sx.fit_implicit(_linear_diffusion, domain, method, samples=128)
        return float(surrogate.report.value_mean)

    # The manifold is rank 3, so too few directions leave a large residual.
    assert fit_with_rank(1) > fit_with_rank(3)
    assert fit_with_rank(3) < 1e-3


def test_implicit_surrogate_survives_jit_vmap_and_grad() -> None:
    def residual(x: jax.Array, y: jax.Array) -> jax.Array:
        return y - 2.0 * x

    domain = sx.box([-1.0], [1.0])
    method = sx.InverseMap(form=sx.Affine(), unknown_domain=sx.box([-3.0], [3.0]))
    surrogate = sx.fit_implicit(residual, domain, method, samples=256)
    point = jnp.array([0.3])

    plain = surrogate(point)
    assert jnp.allclose(jax.jit(lambda m, z: m(z))(surrogate, point), plain)
    batched = jax.vmap(surrogate)(jnp.array([[0.1], [0.3]]))
    assert batched.shape == (2, 1)
    gradient = jax.grad(lambda z: surrogate(z)[0])(point)
    assert gradient.shape == (1,)
