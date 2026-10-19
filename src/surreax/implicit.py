"""Implicit fitting: fitting a target defined by a residual `g(x, y) = 0`.

The target `f` has `g(x, f(x))` near zero. Fitting it reuses the explicit engine
through small conversions. `fit_data` fits a form to given input-output pairs,
`root_find` and `refine` turn a residual, or a residual and a surrogate, into a
solve-at-call approximator, and `fit_implicit` orchestrates these through a
strategy `method`.

The default `InverseMap` method inverts the residual: writing `r = g(x, y)`, the
map `(x, y) -> (x, r)` inverts to `y = h(x, r)`, and `f(x) = h(x, 0)`. The pairs
`((x, r), y)` come from sampling `(x, y)` and evaluating `g` forward, so `h` is
an ordinary regression and no solve is involved. The `Objective` method fits `f`
directly by minimizing the residual it leaves.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import asdex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optax
import optimistix as optx
import splineax
from jaxtyping import Array, Bool, Float

from surreax import normalize, reduction, sampling, sparse_solve, strategy
from surreax.autoencoder import (
    AutoencoderSolve,
    AutoencoderSolver,
    DecodedSurrogate,
    train_autoencoder,
)
from surreax.completion import CompletedSurrogate, Constraint, complete
from surreax.domain import Box, Domain, domain_from_data
from surreax.fitted import FittedSurrogate
from surreax.fitting import LinearModel, fit_linear, fit_linear_sobolev
from surreax.flatten import ravel_example
from surreax.forms import Affine, ChebyshevPolynomial
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport
from surreax.sparse_solve import LinearSolver, NewtonStep

Residual = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]]
"""A residual `g(x, y)`, from an input and an unknown to the residual vector."""

Surrogate = Callable[[Float[Array, " n"]], Float[Array, " m"]]
"""Any approximator of the target, from a flat input to the unknown."""

_START_SEED = 104729
"""Seed for the unknown starts, distinct from the input seed so the residual
coordinate does not come out collinear with the input."""


def _zero_report(dimension: int) -> ErrorReport:
    zero = jnp.asarray(0.0)
    return _value_report(zero, zero, dimension, 0)


def _value_report(
    value_mean: Float[Array, ""],
    value_max: Float[Array, ""],
    dimension: int,
    structure_cost: int,
    coverage_max: Float[Array, ""] | None = None,
) -> ErrorReport:
    zero = jnp.asarray(0.0)
    return ErrorReport(
        value_mean=value_mean,
        value_max=value_max,
        derivative_mean=zero,
        derivative_max=zero,
        projection_mean=zero,
        projection_max=zero,
        worst_case=zero,
        worst_case_point=jnp.zeros(dimension),
        structure_cost=structure_cost,
        coverage_max=zero if coverage_max is None else coverage_max,
    )


def _holdout_count(samples: int, holdout: float) -> int:
    """How many trailing samples a holdout fraction sets aside, at least one."""
    return max(1, round(samples * holdout))


def fit_data(
    inputs: Float[Array, "s d"],
    outputs: Float[Array, "s k"],
    form: LinearForm,
    *,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    reduce_outputs: reduction.OutputReducer | None = None,
    output_jacobians: Float[Array, "s r d"] | None = None,
    derivative_weight: float = 0.0,
) -> FittedSurrogate:
    """Fit a linear form to given input-output pairs, with no target callable.

    Normalizes from sample moments, since the pairs carry no domain box, and
    reports the held-out prediction error. This is the array counterpart of
    `fit`, used where the data is pairs rather than a callable. A `reduce_outputs`
    builder fits the map into a reduced output space, so a large output collapses
    to a handful of coefficients before the fit.

    With `output_jacobians` and a positive `derivative_weight`, the fit also
    matches the Jacobians of the outputs in the inputs, through the Sobolev fit.
    There is one Jacobian per sample, of the outputs after `reduce_outputs` has
    encoded them.
    """
    holdout_count = _holdout_count(inputs.shape[0], holdout)
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]
    train_outputs, holdout_outputs = outputs[:-holdout_count], outputs[-holdout_count:]

    input_reduction = reduction.identity(inputs.shape[1])
    output_reduction = (
        reduction.identity(outputs.shape[1])
        if reduce_outputs is None
        else reduce_outputs(train_outputs)
    )
    input_normalizer = normalize.from_moments(train_inputs)
    encoded_outputs = jax.vmap(output_reduction.forward)(train_outputs)
    output_normalizer = normalize.from_moments(encoded_outputs)
    normalized_inputs = jax.vmap(input_normalizer.forward)(train_inputs)
    normalized_outputs = jax.vmap(output_normalizer.forward)(encoded_outputs)
    if output_jacobians is not None and derivative_weight > 0.0:
        # Rescale the Jacobians into the normalized input and output frame.
        scale = input_normalizer.scale[None, None, :] / output_normalizer.scale[None, :, None]
        model = fit_linear_sobolev(
            form,
            normalized_inputs,
            normalized_outputs,
            output_jacobians[:-holdout_count] * scale,
            ridge=ridge,
            derivative_weight=derivative_weight,
        )
    else:
        model = fit_linear(form, normalized_inputs, normalized_outputs, ridge=ridge)
    _, output_unravel = ravel_example(train_outputs[0])

    surrogate = FittedSurrogate(
        input_reduction=input_reduction,
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=output_normalizer,
        output_reduction=output_reduction,
        output_unravel=output_unravel,
        domain=domain_from_data(train_inputs),
        report=_zero_report(inputs.shape[1]),
    )
    predicted = jax.vmap(surrogate)(holdout_inputs)
    residual = jnp.linalg.norm(predicted - holdout_outputs, axis=1)
    report = _value_report(
        jnp.mean(residual),
        jnp.max(residual),
        inputs.shape[1],
        model.weights.shape[1],
        jnp.max(jax.vmap(surrogate.coverage)(holdout_inputs)),
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)


def _newton_step(
    residual: Residual, x: Float[Array, " n"], y: Float[Array, " m"]
) -> Float[Array, " m"]:
    """One Newton step in the unknown, `y - (dg/dy)^{-1} g(x, y)`."""
    jacobian = jax.jacobian(lambda unknown: residual(x, unknown))(y)
    return y - jnp.linalg.solve(jacobian, residual(x, y))


def _newton_step_matrix_free(
    residual: Residual, x: Float[Array, " n"], y: Float[Array, " m"]
) -> Float[Array, " m"]:
    """One Newton step that solves the Jacobian system without forming it.

    The step needs `(dg/dy)^{-1} g`, which lineax finds by GMRES from the
    Jacobian-vector product alone, so a large unknown never pays for a dense
    `m` by `m` Jacobian. This is the scalable route for a big, sparse system.
    """
    residual_value = residual(x, y)

    def jacobian_vector(direction: Float[Array, " m"]) -> Float[Array, " m"]:
        return jax.jvp(lambda unknown: residual(x, unknown), (y,), (direction,))[1]

    operator = lineax.FunctionLinearOperator(
        jacobian_vector, jax.ShapeDtypeStruct(y.shape, y.dtype)
    )
    step = lineax.linear_solve(operator, residual_value, solver=lineax.GMRES(rtol=1e-6, atol=1e-6))
    return y - step.value


class FoldReport(eqx.Module):
    """Where the residual folds and how tightly one branch was kept.

    The residual folds where its Jacobian in the unknown turns singular, so no
    function of the input exists across the fold. The fit stays on one branch and
    flags the fold rather than averaging through it.
    """

    min_determinant: Float[Array, ""]
    """Smallest `|det(dg/dy)|` seen over the samples, near zero at a fold."""

    boundary_point: Float[Array, " d"]
    """Flat input where that smallest determinant occurs."""

    off_branch: Float[Array, ""]
    """Fraction of samples dropped for lying on a different branch than the anchor."""

    folds: bool = eqx.field(static=True)
    """Whether the smallest determinant fell below the fold tolerance."""


class ImplicitSurrogate(eqx.Module):
    """The zero-residual slice of a fitted inverse map, `f(x) = h(x, 0)`.

    Feed-forward, so it is an explicit approximator of the target built from an
    explicit surrogate of the inverse map. `root_find` and `refine` turn it into
    a solve-at-call approximator.
    """

    inverse_map: FittedSurrogate
    residual_dimension: int = eqx.field(static=True)
    report: ErrorReport
    fold: FoldReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        augmented = jnp.concatenate([x, jnp.zeros(self.residual_dimension)])
        return self.inverse_map(augmented)


class RefinedSurrogate(eqx.Module):
    """A surrogate polished by Newton steps on the residual at call time.

    It approximates the target indirectly: the base surrogate gives the starting
    guess, and the Newton steps drive `g(x, f(x))` toward zero. Both an explicit
    surrogate and a reduced-residual solver use this to correct their output.
    """

    base: Surrogate
    steps: int = eqx.field(static=True)
    residual: Residual = eqx.field(static=True)
    report: ErrorReport

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        prediction = self.base(x)
        for _ in range(self.steps):
            prediction = _newton_step(self.residual, x, prediction)
        return prediction


def refine(surrogate: Surrogate, residual: Residual, *, steps: int) -> RefinedSurrogate:
    """Polish a surrogate's output with Newton steps on the residual at call time."""
    report = getattr(surrogate, "report", _zero_report(1))
    return RefinedSurrogate(base=surrogate, steps=steps, residual=residual, report=report)


def root_find(residual: Residual, initial: Surrogate, *, steps: int) -> RefinedSurrogate:
    """Solve `residual(x, .) = 0` at call, warm-started from `initial`.

    An explicit surrogate of the residual, wrapped this way, becomes an implicit
    surrogate of the target.
    """
    return refine(initial, residual, steps=steps)


class InverseMap(eqx.Module):
    """Fit the target by inverting the residual and reading the zero slice.

    The fit keeps one solution branch, judged by the sign of `det(dg/dy)`, so a
    residual with several branches is not averaged. Turning `continuation` on
    orders the samples along the first input coordinate and warm-starts each
    solve from the previous solution, which follows one branch through a fold.
    """

    unknown_domain: Domain
    newton_steps: int = eqx.field(static=True, default=1)
    continuation: bool = eqx.field(static=True, default=False)
    fold_tolerance: float = eqx.field(static=True, default=1e-3)


class Objective(eqx.Module):
    """Fit the target directly by minimizing the residual it leaves.

    Experimental: the objective is indifferent between solution branches, so it
    can converge to a biased mixture where the residual has several solutions.
    """

    form: LinearForm
    unknown_dimension: int = eqx.field(static=True)
    steps: int = eqx.field(static=True, default=500)
    learning_rate: float = eqx.field(static=True, default=0.01)


def _solve_independently(
    residual: Residual,
    inputs: Float[Array, "n d"],
    starts: Float[Array, "n m"],
    newton_steps: int,
) -> Float[Array, "n m"]:
    """Newton-solve every sample from its own start, in parallel."""
    setup = NewtonSetup(step=_newton_step, pattern=None)
    return _solve_independently_with(setup, residual, inputs, starts, newton_steps)


def _solve_by_continuation(
    residual: Residual,
    inputs: Float[Array, "n d"],
    starts: Float[Array, "n m"],
    newton_steps: int,
) -> Float[Array, "n m"]:
    """Newton-solve along the first input coordinate, warm-started from the last.

    Ordering the samples into a path and carrying the solution forward keeps the
    solve on one branch through a fold, since each start is already near the next
    solution. Returns the unknowns in the original sample order.
    """
    order = jnp.argsort(inputs[:, 0])
    ordered_inputs = inputs[order]

    def step(previous, x):
        unknown = previous
        for _ in range(max(1, newton_steps)):
            unknown = _newton_step(residual, x, unknown)
        return unknown, unknown

    _, ordered_unknowns = jax.lax.scan(step, starts[order][0], ordered_inputs)
    return ordered_unknowns[jnp.argsort(order)]


def _branch_mask(
    determinants: Float[Array, " n"],
) -> tuple[Bool[Array, " n"], Float[Array, ""]]:
    """Keep the samples whose determinant sign matches the most interior sample.

    The anchor is the sample with the largest `|det|`, the furthest from a fold,
    so its branch is the reliable one to keep.
    """
    anchor_sign = jnp.sign(determinants[jnp.argmax(jnp.abs(determinants))])
    keep = determinants * anchor_sign > 0
    return keep, jnp.mean(~keep)


class ExactSolve(eqx.Module):
    """Fit a large-unknown target by solving the residual then reducing the output.

    Suits a system where the input is small but the unknown is large. It solves
    `g = 0` for the unknown at each sample, collapses the solved unknowns onto
    their leading principal directions, and fits the map into that reduced space,
    so the surrogate predicts a handful of coefficients rather than the full
    unknown. By default each Newton step factors the sparse Jacobian, which is
    what lets it reach a large unknown, see `sparse_solve`.

    A positive `derivative_weight` also fits the Jacobian of the reduced
    coefficients in the input, which the implicit function theorem gives from
    one transposed solve per coefficient at each solved sample.

    Each solve must reach a residual of at most `convergence_tolerance` times
    the residual at the zero state, or the fit raises an error. Newton diverges from a
    start far from a root of a stiff residual, and a basis built from states
    that did not converge is wrong without any other sign. Setting it to None
    skips the check.
    """

    unknown_domain: Domain
    rank: int | None = eqx.field(static=True, default=None)
    variance: float = eqx.field(static=True, default=0.99)
    newton_steps: int = eqx.field(static=True, default=8)
    convergence_tolerance: float | None = eqx.field(static=True, default=1e-3)
    linear_solver: LinearSolver = eqx.field(static=True, default="sparse")
    reuse: splineax.HybridSettings | None = eqx.field(static=True, default=None)
    derivative_weight: float = eqx.field(static=True, default=0.0)


class NewtonSetup(NamedTuple):
    """The Newton step chosen for a residual, and the sparsity pattern it uses."""

    step: NewtonStep
    """The step `y -> y - (dg/dy)^{-1} g(x, y)`."""

    pattern: asdex.ColoredPattern | None
    """The colored Jacobian pattern of a sparse step, in which case the solves of
    one sample thread a solver state, or None for a dense or matrix-free step."""


def newton_setup(
    residual: Residual,
    linear_solver: LinearSolver,
    reuse: splineax.HybridSettings | None,
    input_example: Float[Array, " n"],
    unknown_example: Float[Array, " m"],
) -> NewtonSetup:
    """Choose the Newton step for a residual from the requested linear solver.

    A `"sparse"` request reads the Jacobian pattern, and falls back to the
    matrix-free step when the residual has no usable pattern.
    """
    if linear_solver == "dense":
        return NewtonSetup(step=_newton_step, pattern=None)
    if linear_solver == "sparse":
        pattern = sparse_solve.residual_pattern(residual, input_example, unknown_example)
        if pattern is not None:
            return NewtonSetup(
                step=sparse_solve.sparse_newton_step(pattern, reuse), pattern=pattern
            )
    return NewtonSetup(step=_newton_step_matrix_free, pattern=None)


def _fit_exact_solve(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: ExactSolve,
    form: LinearForm,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Solve the residual for every sample, then fit the reduced output map."""
    starts = sampling.sample(method.unknown_domain, inputs.shape[0], seed=_START_SEED)
    setup = newton_setup(residual, method.linear_solver, method.reuse, inputs[0], starts[0])
    unknowns = _solve_to_convergence(
        setup, residual, inputs, starts, method.newton_steps, method.convergence_tolerance
    )
    reduce_outputs = reduction.pca_outputs(rank=method.rank, variance=method.variance)
    if method.derivative_weight <= 0.0:
        return fit_data(
            inputs,
            unknowns,
            form,
            holdout=holdout,
            ridge=ridge,
            reduce_outputs=reduce_outputs,
        )
    # The Jacobians need the reduced basis, so build it here from the same
    # training rows `fit_data` would use, and hand it over unchanged.
    projection = reduce_outputs(unknowns[: -_holdout_count(inputs.shape[0], holdout)])
    jacobians = sparse_solve.reduced_state_jacobians(
        residual, inputs, unknowns, projection.encode_matrix, method.linear_solver, setup.pattern
    )
    return fit_data(
        inputs,
        unknowns,
        form,
        holdout=holdout,
        ridge=ridge,
        reduce_outputs=lambda outputs: projection,
        output_jacobians=jacobians,
        derivative_weight=method.derivative_weight,
    )


def _solve_to_convergence(
    setup: "NewtonSetup",
    residual: Residual,
    inputs: Float[Array, "n d"],
    starts: Float[Array, "n m"],
    newton_steps: int,
    tolerance: float | None,
) -> Float[Array, "n m"]:
    """Newton-solve every sample, and raise if any solve did not reach a root.

    A solve has converged when its residual norm is at most `tolerance` times the
    residual norm at the zero state, plus a floor of a hundred machine epsilons for
    an input whose zero state is already a root. The zero state sets the scale
    because the residual at a far start can be astronomically large, which would
    make any ratio to it small. The check runs inside `eqx.error_if`, so it also
    works under `jit`. A `tolerance` of None skips it.
    """
    unknowns = _solve_independently_with(setup, residual, inputs, starts, newton_steps)
    if tolerance is None:
        return unknowns
    left = jnp.linalg.norm(jax.vmap(residual)(inputs, unknowns), axis=1)
    scale = jnp.linalg.norm(jax.vmap(residual)(inputs, jnp.zeros_like(unknowns)), axis=1)
    floor = 100.0 * jnp.finfo(unknowns.dtype).eps
    return eqx.error_if(
        unknowns,
        jnp.any(left > tolerance * scale + floor),
        "The Newton solves for the full states did not converge. Start closer to a "
        "root with a smaller unknown_domain, raise newton_steps, or set "
        "convergence_tolerance to None to skip this check.",
    )


def _solve_independently_with(
    setup: NewtonSetup,
    residual: Residual,
    inputs: Float[Array, "n d"],
    starts: Float[Array, "n m"],
    newton_steps: int,
) -> Float[Array, "n m"]:
    """Iterate a Newton step over every sample from its own start.

    A sparse step threads one solver state through each sample's steps, see
    `sparse_solve.solve_samples_threaded`. The other steps run every sample in
    parallel.
    """
    if setup.pattern is not None:
        return sparse_solve.solve_samples_threaded(
            setup.step, residual, inputs, starts, newton_steps
        )
    unknowns = starts
    for _ in range(newton_steps):
        unknowns = jax.vmap(lambda x, y: setup.step(residual, x, y))(inputs, unknowns)
    return unknowns


class ReducedSolve(eqx.Module):
    """Reduced-order model: solve a projected residual in a small latent space.

    For a large unknown whose solutions lie on a low-dimensional manifold, this
    builds a proper orthogonal decomposition basis from a few solved snapshots,
    projects the residual onto that basis in the manner of Galerkin, and solves
    the reduced residual by Newton in the latent. Each reduced step is an `r` by
    `r` solve, independent of the unknown size, so the online solve is cheap. It
    then distills the input-to-latent map into an explicit surrogate, the non-
    intrusive reduced-order model, so inference needs no solve. The basis is
    linear here, which suits a manifold whose Kolmogorov n-width decays fast.

    With `greedy` set, the snapshots are chosen one at a time where the current
    basis leaves the most residual, in the manner of the reduced-basis greedy of
    Prud'homme and colleagues, https://doi.org/10.1115/1.1448332, so it reaches a
    given accuracy with fewer full solves than the first `snapshots` samples.

    See Benner, Gugercin, and Willcox on projection-based model-order reduction,
    https://doi.org/10.1137/130932715, and Hesthaven and Ubbiali on the distilled
    POD-NN, https://doi.org/10.1016/j.jcp.2018.02.037.
    """

    unknown_domain: Domain
    rank: int = eqx.field(static=True)
    snapshots: int = eqx.field(static=True, default=64)
    newton_steps: int = eqx.field(static=True, default=8)
    reduced_steps: int = eqx.field(static=True, default=8)
    convergence_tolerance: float | None = eqx.field(static=True, default=1e-3)
    linear_solver: LinearSolver = eqx.field(static=True, default="sparse")
    reuse: splineax.HybridSettings | None = eqx.field(static=True, default=None)
    derivative_weight: float = eqx.field(static=True, default=0.0)
    greedy: bool = eqx.field(static=True, default=False)
    snapshot_tolerance: float = eqx.field(static=True, default=0.0)


class LatentReport(eqx.Module):
    """How trustworthy the reduced solve is over the samples.

    A reduced residual can have several latent roots, and its latent Jacobian can
    turn singular near a fold, so the reduced solve is reported rather than
    trusted blindly.
    """

    min_conditioning: Float[Array, ""]
    """Smallest reciprocal condition number of the latent Jacobian over the
    samples, near zero where the reduced problem is ill-conditioned."""

    boundary_point: Float[Array, " d"]
    """Flat input where that worst conditioning occurs."""

    multiple_roots: bool = eqx.field(static=True)
    """Whether a second start reached a different latent root on some sample."""


def _pod_basis(
    snapshots: Float[Array, "s m"], rank: int
) -> tuple[Float[Array, " m"], Float[Array, "r m"]]:
    """The mean and the leading `rank` principal directions of the snapshots."""
    center = jnp.mean(snapshots, axis=0)
    _, _, directions = jnp.linalg.svd(snapshots - center, full_matrices=False)
    return center, directions[:rank]


def _reduced_residual(
    residual: Residual,
    x: Float[Array, " n"],
    z: Float[Array, " r"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
) -> Float[Array, " r"]:
    """The Galerkin projection of the residual, `basis @ g(x, center + basis.T z)`."""
    return basis @ residual(x, center + basis.T @ z)


def _reduced_newton_step(
    residual: Residual,
    x: Float[Array, " n"],
    z: Float[Array, " r"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
) -> Float[Array, " r"]:
    """One Newton step on the reduced residual, an `r` by `r` solve in the latent."""

    def reduced(latent: Float[Array, " r"]) -> Float[Array, " r"]:
        return _reduced_residual(residual, x, latent, center, basis)

    return z - jnp.linalg.solve(jax.jacobian(reduced)(z), reduced(z))


def _latent_newton(
    residual: Residual,
    x: Float[Array, " n"],
    start: Float[Array, " r"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
    steps: int,
) -> Float[Array, " r"]:
    """Run the reduced Newton from a start, without implicit differentiation."""
    latent = start
    for _ in range(steps):
        latent = _reduced_newton_step(residual, x, latent, center, basis)
    return latent


@partial(jax.custom_jvp, nondiff_argnums=(0, 4))
def _reduced_root(
    residual: Residual,
    x: Float[Array, " n"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
    steps: int,
) -> Float[Array, " r"]:
    """Solve the reduced residual for the latent by Newton from zero."""
    return _latent_newton(residual, x, jnp.zeros(basis.shape[0]), center, basis, steps)


@_reduced_root.defjvp
def _reduced_root_jvp(
    residual: Residual,
    steps: int,
    primals: tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, "r m"]],
    tangents: tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, "r m"]],
) -> tuple[Float[Array, " r"], Float[Array, " r"]]:
    """Differentiate the reduced solve by the implicit function theorem.

    At the root the reduced residual `p` is zero, so its total derivative gives
    `dz = -(dp/dz)^{-1} p_dot`, where `p_dot` is the change in `p` at fixed `z`.
    This is exact and avoids differentiating the Newton iterates.
    """
    x, center, basis = primals
    z = _reduced_root(residual, x, center, basis, steps)
    latent_jacobian = jax.jacobian(
        lambda latent: _reduced_residual(residual, x, latent, center, basis)
    )(z)
    _, p_dot = jax.jvp(lambda a, b, c: _reduced_residual(residual, a, z, b, c), primals, tangents)
    return z, -jnp.linalg.solve(latent_jacobian, p_dot)


class ReducedSolver(eqx.Module):
    """The reduced-order solver, a cheap differentiable oracle for the target.

    A call solves the Galerkin-projected residual in the latent by Newton, an `r`
    by `r` solve independent of the unknown size, and decodes to the full field.
    Its input derivative comes from the implicit function theorem rather than
    differentiating the Newton iterates. `report` records where the reduced
    problem is ill-conditioned or has several roots.
    """

    center: Float[Array, " m"]
    basis: Float[Array, "r m"]
    report: LatentReport
    residual: Residual = eqx.field(static=True)
    reduced_steps: int = eqx.field(static=True)

    def latent(self, x: Float[Array, " n"]) -> Float[Array, " r"]:
        """Solve the reduced residual for the latent at `x`."""
        return _reduced_root(self.residual, x, self.center, self.basis, self.reduced_steps)

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.center + self.basis.T @ self.latent(x)


def _latent_report(
    residual: Residual,
    inputs: Float[Array, "n d"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
    steps: int,
) -> LatentReport:
    """Read the conditioning and root multiplicity of the reduced solve."""
    zeros_start = jnp.zeros(basis.shape[0])

    def solve_from(start: Float[Array, " r"], x: Float[Array, " n"]) -> Float[Array, " r"]:
        return _latent_newton(residual, x, start, center, basis, steps)

    latents = jax.vmap(lambda x: solve_from(zeros_start, x))(inputs)

    def conditioning(x: Float[Array, " n"], z: Float[Array, " r"]) -> Float[Array, ""]:
        def reduced(latent: Float[Array, " r"]) -> Float[Array, " r"]:
            return _reduced_residual(residual, x, latent, center, basis)

        singular = jnp.linalg.svd(jax.jacobian(reduced)(z), compute_uv=False)
        return singular[-1] / singular[0]

    conditions = jax.vmap(conditioning)(inputs, latents)
    worst = jnp.argmin(conditions)

    # A second start on a different root shows the reduced problem is multivalued.
    alternate = jax.vmap(lambda x: solve_from(zeros_start + 1.0, x))(inputs)
    spread = jnp.max(jnp.linalg.norm(latents - alternate, axis=1))
    return LatentReport(
        min_conditioning=conditions[worst],
        boundary_point=inputs[worst],
        multiple_roots=bool(spread > 1e-3),
    )


def _snapshot_setup(
    residual: Residual, snapshot_inputs: Float[Array, "s d"], method: ReducedSolve
) -> NewtonSetup:
    """The Newton step for the snapshot solves of a reduced-order method."""
    unknown_example = sampling.sample(method.unknown_domain, 1, seed=_START_SEED)[0]
    return newton_setup(
        residual, method.linear_solver, method.reuse, snapshot_inputs[0], unknown_example
    )


def _solve_snapshots(
    residual: Residual,
    snapshot_inputs: Float[Array, "s d"],
    method: ReducedSolve,
    setup: NewtonSetup | None = None,
) -> Float[Array, "s m"]:
    """Full-solve the residual at the snapshot inputs.

    A `setup` from an earlier call skips reading the Jacobian pattern again.
    """
    if setup is None:
        setup = _snapshot_setup(residual, snapshot_inputs, method)
    starts = sampling.sample(method.unknown_domain, snapshot_inputs.shape[0], seed=_START_SEED)
    return _solve_to_convergence(
        setup,
        residual,
        snapshot_inputs,
        starts,
        method.newton_steps,
        method.convergence_tolerance,
    )


def _greedy_states(
    residual: Residual, candidates: Float[Array, "n d"], method: ReducedSolve
) -> Float[Array, "s m"]:
    """Pick snapshots one at a time where the current basis leaves the most residual.

    Each round scores every candidate by the residual its reduced solve leaves,
    which is cheap, and full-solves only the worst, so the full solves stay at the
    snapshot budget while the basis targets where it is weakest.
    """

    def leftover(
        x: Float[Array, " d"], center: Float[Array, " m"], basis: Float[Array, "r m"]
    ) -> Float[Array, ""]:
        latent = _latent_newton(
            residual, x, jnp.zeros(basis.shape[0]), center, basis, method.reduced_steps
        )
        return jnp.linalg.norm(residual(x, center + basis.T @ latent))

    setup = _snapshot_setup(residual, candidates, method)
    states = _solve_snapshots(residual, candidates[:1], method, setup)
    for _ in range(method.snapshots - 1):
        center, basis = _pod_basis(states, min(method.rank, states.shape[0]))
        scores = jax.vmap(leftover, in_axes=(0, None, None))(candidates, center, basis)
        worst = int(jnp.argmax(scores))
        if float(scores[worst]) < method.snapshot_tolerance:
            break
        chosen = _solve_snapshots(residual, candidates[worst : worst + 1], method, setup)
        states = jnp.concatenate([states, chosen])
    return states


def _build_reduced_solver(
    residual: Residual, candidates: Float[Array, "n d"], method: ReducedSolve
) -> ReducedSolver:
    """Choose snapshots, take the POD basis, and read the latent report."""
    if method.greedy:
        states = _greedy_states(residual, candidates, method)
    else:
        states = _solve_snapshots(residual, candidates[: method.snapshots], method)
    center, basis = _pod_basis(states, min(method.rank, states.shape[0]))
    report = _latent_report(
        residual, candidates[: method.snapshots], center, basis, method.reduced_steps
    )
    return ReducedSolver(
        center=center,
        basis=basis,
        report=report,
        residual=residual,
        reduced_steps=method.reduced_steps,
    )


def reduced_solver(
    residual: Residual,
    domain: Domain,
    method: ReducedSolve,
    *,
    samples: int = 128,
    seed: int = 0,
) -> ReducedSolver:
    """Build the reduced-order solver for a residual over a domain.

    Samples the domain for the snapshot candidates, builds the basis, and returns
    the solver on its own, so a caller has the cheap latent solve and its
    derivative without the distilled surrogate.
    """
    inputs = sampling.sample(domain, samples, seed=seed)
    return _build_reduced_solver(residual, inputs, method)


def _fit_reduced_order(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: ReducedSolve,
    form: LinearForm,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Build the reduced solver, reduce-solve every sample, and distill."""
    solver = _build_reduced_solver(residual, inputs[: method.snapshots], method)
    latents = jax.vmap(solver.latent)(inputs)
    projection = reduction.LinearReduction(
        encode_matrix=solver.basis,
        encode_bias=-solver.basis @ solver.center,
        decode_matrix=solver.basis.T,
        decode_bias=solver.center,
    )
    return _distill_latent_map(
        solver,
        inputs,
        latents,
        projection,
        form,
        holdout,
        ridge,
        method.derivative_weight,
    )


def _distill_latent_map(
    solver: ReducedSolver,
    inputs: Float[Array, "n d"],
    latents: Float[Array, "n r"],
    projection: reduction.LinearReduction,
    form: LinearForm,
    holdout: float,
    ridge: float,
    derivative_weight: float,
) -> FittedSurrogate:
    """Fit the input-to-latent map and decode through the POD projection.

    A positive `derivative_weight` supervises the map's Jacobian with the solver's
    own, taken through the implicit function theorem, so the distilled map matches
    the solver's slope as well as its value.
    """
    holdout_count = max(1, round(inputs.shape[0] * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]
    train_latents, holdout_latents = latents[:-holdout_count], latents[-holdout_count:]

    input_normalizer = normalize.from_moments(train_inputs)
    latent_normalizer = normalize.from_moments(train_latents)
    normalized_inputs = jax.vmap(input_normalizer.forward)(train_inputs)
    normalized_latents = jax.vmap(latent_normalizer.forward)(train_latents)
    if derivative_weight > 0.0:
        # Rescale the solver Jacobian into the normalized input and latent frame.
        raw = jax.vmap(jax.jacobian(solver.latent))(train_inputs)
        scale = input_normalizer.scale[None, None, :] / latent_normalizer.scale[None, :, None]
        model = fit_linear_sobolev(
            form,
            normalized_inputs,
            normalized_latents,
            raw * scale,
            ridge=ridge,
            derivative_weight=derivative_weight,
        )
    else:
        model = fit_linear(form, normalized_inputs, normalized_latents, ridge=ridge)
    _, output_unravel = ravel_example(projection.inverse(train_latents[0]))

    surrogate = FittedSurrogate(
        input_reduction=reduction.identity(inputs.shape[1]),
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=latent_normalizer,
        output_reduction=projection,
        output_unravel=output_unravel,
        domain=domain_from_data(train_inputs),
        report=_zero_report(inputs.shape[1]),
    )
    # Score the distilled map against the reduced solver in the latent space.
    predicted = jax.vmap(lambda x: latent_normalizer.inverse(model(input_normalizer.forward(x))))(
        holdout_inputs
    )
    residual = jnp.linalg.norm(predicted - holdout_latents, axis=1)
    report = _value_report(
        jnp.mean(residual), jnp.max(residual), inputs.shape[1], model.weights.shape[1]
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)


def _fit_inverse_map(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: InverseMap,
    form: LinearForm,
    holdout: float,
    ridge: float,
) -> ImplicitSurrogate:
    """Solve for the unknowns, keep one branch, and fit the inverse map.

    The Jacobian in the unknown is computed for the solve anyway, so its
    determinant comes for free and drives both the branch filter and the fold
    flag.
    """
    starts = sampling.sample(method.unknown_domain, inputs.shape[0], seed=_START_SEED)
    solve = _solve_by_continuation if method.continuation else _solve_independently
    unknowns = solve(residual, inputs, starts, method.newton_steps)

    determinants = jax.vmap(
        lambda x, y: jnp.linalg.det(jax.jacobian(lambda u: residual(x, u))(y))
    )(inputs, unknowns)
    keep, off_branch = _branch_mask(determinants)
    smallest = jnp.argmin(jnp.abs(determinants))
    fold = FoldReport(
        min_determinant=jnp.abs(determinants[smallest]),
        boundary_point=inputs[smallest],
        off_branch=off_branch,
        folds=bool(jnp.abs(determinants[smallest]) < method.fold_tolerance),
    )

    # Boolean indexing runs eagerly here, so the kept subset can shrink freely.
    kept = np.asarray(keep)
    on_branch_inputs, on_branch_unknowns = inputs[kept], unknowns[kept]
    residuals = jax.vmap(residual)(on_branch_inputs, on_branch_unknowns)
    augmented = jnp.concatenate([on_branch_inputs, residuals], axis=1)
    inverse_map = fit_data(augmented, on_branch_unknowns, form, holdout=holdout, ridge=ridge)
    return ImplicitSurrogate(
        inverse_map=inverse_map,
        residual_dimension=residuals.shape[1],
        report=_value_report(
            jnp.asarray(0.0), jnp.asarray(0.0), inputs.shape[1], inverse_map.report.structure_cost
        ),
        fold=fold,
    )


def _fit_objective(
    residual: Residual, domain: Domain, inputs: Float[Array, "n d"], method: Objective
) -> FittedSurrogate:
    """Fit a linear form directly by minimizing the residual it leaves."""
    lower, upper = domain.bounds()
    input_normalizer = normalize.from_bounds(lower, upper)
    normalized = jax.vmap(input_normalizer.forward)(inputs)
    feature_count = method.form.features(normalized[0]).shape[0]

    def loss(weights, _):
        residuals = jax.vmap(lambda x, z: residual(x, weights @ method.form.features(z)))(
            inputs, normalized
        )
        return jnp.mean(residuals**2)

    solver = optx.OptaxMinimiser(optax.adam(method.learning_rate), rtol=1e-8, atol=1e-8)
    start = jnp.zeros((method.unknown_dimension, feature_count))
    weights = optx.minimise(loss, solver, start, max_steps=method.steps, throw=False).value

    identity_output = normalize.Normalizer(
        center=jnp.zeros(method.unknown_dimension), scale=jnp.ones(method.unknown_dimension)
    )
    _, output_unravel = ravel_example(jnp.zeros(method.unknown_dimension))
    return FittedSurrogate(
        input_reduction=reduction.identity(inputs.shape[1]),
        input_normalizer=input_normalizer,
        model=LinearModel(weights=weights, form=method.form),
        output_normalizer=identity_output,
        output_reduction=reduction.identity(method.unknown_dimension),
        output_unravel=output_unravel,
        domain=domain,
        report=_value_report(jnp.asarray(0.0), jnp.asarray(0.0), inputs.shape[1], feature_count),
    )


def autoencoder_solver(
    residual: Residual,
    domain: Domain,
    method: AutoencoderSolve,
    *,
    seed: int = 0,
) -> AutoencoderSolver:
    """Train the learned-manifold solver for a residual over a domain.

    Returns the solver on its own, so a caller has the reduced solve on the
    learned manifold and its derivative without the distilled surrogate. With
    `snapshots` at zero the training evaluates the residual forward only and
    never solves the full system. Otherwise it first solves that many full states
    from zero, to pretrain on.
    """
    held_out = sampling.sample(domain, method.batch, seed=seed)
    if method.snapshots == 0:
        return train_autoencoder(residual, domain, method, held_out=held_out, seed=seed)
    snapshot_inputs = sampling.sample(domain, method.snapshots, seed=seed + 1)
    starts = jnp.zeros((method.snapshots, method.unknown_dimension))
    setup = newton_setup(
        residual, method.linear_solver, method.reuse, snapshot_inputs[0], starts[0]
    )
    states = _solve_to_convergence(
        setup,
        residual,
        snapshot_inputs,
        starts,
        method.newton_steps,
        method.convergence_tolerance,
    )
    return train_autoencoder(
        residual,
        domain,
        method,
        held_out=held_out,
        seed=seed,
        snapshot_inputs=snapshot_inputs,
        snapshot_states=states,
    )


def _fit_autoencoder(
    residual: Residual,
    domain: Domain,
    inputs: Float[Array, "n d"],
    method: AutoencoderSolve,
    form: LinearForm,
    holdout: float,
    ridge: float,
    seed: int,
) -> DecodedSurrogate:
    """Train the solver, reduce-solve every sample, and distill the latent map."""
    solver = autoencoder_solver(residual, domain, method, seed=seed)
    latents = jax.vmap(solver.latent)(inputs)
    latent_map = fit_data(inputs, latents, form, holdout=holdout, ridge=ridge)
    return DecodedSurrogate(latent_map=latent_map, solver=solver, report=latent_map.report)


def recommend_reduced_method(
    residual: Residual,
    domain: Domain,
    method: ReducedSolve,
    *,
    tolerance: float = 1e-3,
    seed: int = 0,
) -> ReducedSolve | AutoencoderSolve:
    """Choose between a linear basis and a learned manifold from the snapshots.

    Solves the snapshots `method` asks for and reads how much of their variation
    `method.rank` linear modes capture. When a linear basis leaves out less than
    `tolerance`, the linear reduced-order model is simpler and as accurate, so
    `method` comes back unchanged. Otherwise the manifold bends or carries a
    moving feature, and an `AutoencoderSolve` at the same rank, pretrained on the
    same number of snapshots, is returned.
    """
    inputs = sampling.sample(domain, method.snapshots, seed=seed)
    states = _solve_snapshots(residual, inputs, method)
    if strategy.linear_energy(states, method.rank) >= 1.0 - tolerance:
        return method
    return AutoencoderSolve(
        unknown_dimension=method.unknown_domain.dimension,
        rank=method.rank,
        snapshots=method.snapshots,
        newton_steps=method.newton_steps,
        convergence_tolerance=method.convergence_tolerance,
        linear_solver=method.linear_solver,
        reuse=method.reuse,
    )


_SMALL_UNKNOWN = 32
"""The largest unknown dimension treated as small, where a dense Jacobian is cheap.

Measured on the radiating rod and the latch ring, eight samples and six Newton
steps in double precision: the dense step is faster at 32 unknowns and the
sparse step at 64.
"""


def recommend_linear_solver(
    residual: Residual, domain: Domain, unknown_domain: Domain, *, seed: int = 0
) -> LinearSolver:
    """Pick how a Newton step solves its Jacobian system for this residual.

    A small unknown takes the dense solve. A larger one takes the sparse direct
    solve when the residual has a usable sparsity pattern, and the matrix-free
    solve otherwise, see `sparse_solve.residual_pattern`.
    """
    if unknown_domain.dimension <= _SMALL_UNKNOWN:
        return "dense"
    input_example = sampling.sample(domain, 1, seed=seed)[0]
    unknown_example = sampling.sample(unknown_domain, 1, seed=_START_SEED)[0]
    if sparse_solve.residual_pattern(residual, input_example, unknown_example) is None:
        return "matrix_free"
    return "sparse"


def recommend_method(
    residual: Residual,
    domain: Domain,
    unknown_domain: Domain,
    *,
    samples: int = 128,
    seed: int = 0,
) -> InverseMap | ExactSolve:
    """Pick an implicit method from the residual's structure.

    A large unknown goes to `ExactSolve`, which reduces the state. A small one
    goes to `InverseMap`, exact with no iteration when the residual is affine in
    the unknown, and a polynomial inverse with Newton steps otherwise.
    """
    affine = strategy.residual_is_affine_in_unknown(
        residual, domain, unknown_domain, samples=samples, seed=seed
    )
    if unknown_domain.dimension > _SMALL_UNKNOWN:
        linear_solver = recommend_linear_solver(residual, domain, unknown_domain, seed=seed)
        return ExactSolve(unknown_domain=unknown_domain, linear_solver=linear_solver)
    if affine:
        return InverseMap(unknown_domain=unknown_domain, newton_steps=1)
    return InverseMap(unknown_domain=unknown_domain, newton_steps=3)


BaseMethod = InverseMap | Objective | ExactSolve | ReducedSolve | AutoencoderSolve
"""The implicit methods that fit a surrogate on their own."""


class Completed(eqx.Module):
    """Fit through another method, then satisfy an equality constraint exactly.

    The inner `method` fits every output, and the result is wrapped so that the
    dependent outputs the constraint names are solved from it at call time, see
    `completion`. The constraint then holds to solver precision, while the full
    residual stays as accurate as the inner method leaves it.
    """

    method: BaseMethod
    constraint: Constraint


BaseSurrogate = ImplicitSurrogate | FittedSurrogate | DecodedSurrogate | CompletedSurrogate
"""What a method returns before any refinement."""


def recommend_implicit_form(
    residual: Residual,
    domain: Domain,
    method: BaseMethod | Completed,
    *,
    samples: int = 128,
    seed: int = 0,
) -> LinearForm:
    """Pick the form that maps the input to the unknowns or their latent.

    A residual that is affine in the unknown gives an `Affine` map. Any other
    residual takes a `ChebyshevPolynomial`, of degree 3 for an `InverseMap` and
    of degree 2 for the reduced methods. An `AutoencoderSolve` has learned latent
    coordinates with no structure to read from the residual, so it takes degree 3.
    An `Objective` is its own form.
    """
    if isinstance(method, Completed):
        return recommend_implicit_form(residual, domain, method.method, samples=samples, seed=seed)
    if isinstance(method, Objective):
        return method.form
    if isinstance(method, AutoencoderSolve):
        return ChebyshevPolynomial(degree=3)
    affine = strategy.residual_is_affine_in_unknown(
        residual, domain, method.unknown_domain, samples=samples, seed=seed
    )
    if affine:
        return Affine()
    return ChebyshevPolynomial(degree=3 if isinstance(method, InverseMap) else 2)


def _resolve_form(
    residual: Residual,
    domain: Domain,
    method: BaseMethod | Completed,
    form: LinearForm | None,
    seed: int,
) -> LinearForm:
    """Return `form`, or a recommended one when it is None."""
    inner = method.method if isinstance(method, Completed) else method
    if isinstance(inner, Objective) and form is not None:
        raise ValueError("An `Objective` is its own form, so it takes no `form` here.")
    if form is None:
        return recommend_implicit_form(residual, domain, method, seed=seed)
    return form


def _fit_method(
    residual: Residual,
    domain: Domain,
    train_inputs: Float[Array, "n d"],
    method: BaseMethod | Completed,
    form: LinearForm,
    holdout: float,
    ridge: float,
    seed: int,
) -> BaseSurrogate:
    """Fit a surrogate through one method on the training inputs."""
    match method:
        case InverseMap():
            return _fit_inverse_map(residual, train_inputs, method, form, holdout, ridge)
        case Objective():
            return _fit_objective(residual, domain, train_inputs, method)
        case ExactSolve():
            return _fit_exact_solve(residual, train_inputs, method, form, holdout, ridge)
        case ReducedSolve():
            return _fit_reduced_order(residual, train_inputs, method, form, holdout, ridge)
        case AutoencoderSolve():
            return _fit_autoencoder(
                residual, domain, train_inputs, method, form, holdout, ridge, seed
            )
        case Completed():
            inner = _fit_method(
                residual, domain, train_inputs, method.method, form, holdout, ridge, seed
            )
            return complete(inner, method.constraint, train_inputs, inner.report)
        case _:
            raise TypeError(f"unsupported implicit method: {type(method).__name__}")


def fit_implicit(
    residual: Residual,
    domain: Domain,
    method: BaseMethod | Completed | None = None,
    form: LinearForm | None = None,
    *,
    unknown_domain: Domain | None = None,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    refine_steps: int = 0,
    sampler: sampling.Sampler = "sobol",
    seed: int = 0,
) -> BaseSurrogate | RefinedSurrogate:
    """Fit a surrogate `f` with `residual(x, f(x))` near zero, through a method.

    Samples the domain, builds the surrogate through the strategy `method`, and
    reports the residual the surrogate leaves on held-out inputs. A positive
    `refine_steps` polishes each prediction with Newton steps on the residual at
    call time, which every method's result can use. Leaving `method` out picks
    one from the residual's structure, which needs `unknown_domain`.

    The `form` fits the map from the input to the solved unknowns, or to their
    latent. Leaving it out picks one, see `recommend_implicit_form`. An
    `Objective` is its own form and refuses a `form` here. The residual takes
    the input as a flat array, so the domain must be a `Box`. `sampler` picks
    the sequence the samples follow.
    """
    if not isinstance(domain, Box):
        raise TypeError("fit_implicit needs a Box domain")
    if method is None:
        if unknown_domain is None:
            raise ValueError("fit_implicit needs a method or an unknown_domain to recommend one")
        method = recommend_method(residual, domain, unknown_domain, seed=seed)

    inputs = sampling.sample(domain, samples, seed=seed, sampler=sampler)
    holdout_count = max(1, round(samples * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]

    form = _resolve_form(residual, domain, method, form, seed)
    base = _fit_method(residual, domain, train_inputs, method, form, holdout, ridge, seed)
    surrogate: BaseSurrogate | RefinedSurrogate = base
    if refine_steps > 0:
        surrogate = refine(base, residual, steps=refine_steps)

    predicted = jax.vmap(surrogate)(holdout_inputs)
    left = jnp.linalg.norm(jax.vmap(residual)(holdout_inputs, predicted), axis=1)
    report = _value_report(
        jnp.mean(left), jnp.max(left), inputs.shape[1], base.report.structure_cost
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)
