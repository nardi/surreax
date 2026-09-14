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

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optax
import optimistix as optx
from jaxtyping import Array, Bool, Float

from surreax import normalize, reduction, sampling, strategy
from surreax.domain import Domain, box_from_data
from surreax.fitted import FittedSurrogate
from surreax.fitting import LinearModel, fit_linear
from surreax.flatten import ravel_example
from surreax.forms import Affine, ChebyshevPolynomial
from surreax.forms.protocol import LinearForm
from surreax.report import ErrorReport

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
    )


def fit_data(
    inputs: Float[Array, "s d"],
    outputs: Float[Array, "s k"],
    form: LinearForm,
    *,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    reduce_outputs: reduction.OutputReducer | None = None,
) -> FittedSurrogate:
    """Fit a linear form to given input-output pairs, with no target callable.

    Normalizes from sample moments, since the pairs carry no domain box, and
    reports the held-out prediction error. This is the array counterpart of
    `fit`, used where the data is pairs rather than a callable. A `reduce_outputs`
    builder fits the map into a reduced output space, so a large output collapses
    to a handful of coefficients before the fit.
    """
    holdout_count = max(1, round(inputs.shape[0] * holdout))
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
    model = fit_linear(form, normalized_inputs, normalized_outputs, ridge=ridge)
    _, output_unravel = ravel_example(train_outputs[0])

    surrogate = FittedSurrogate(
        input_reduction=input_reduction,
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=output_normalizer,
        output_reduction=output_reduction,
        output_unravel=output_unravel,
        domain=box_from_data(inputs),
        report=_zero_report(inputs.shape[1]),
    )
    predicted = jax.vmap(surrogate)(holdout_inputs)
    residual = jnp.linalg.norm(predicted - holdout_outputs, axis=1)
    report = _value_report(
        jnp.mean(residual), jnp.max(residual), inputs.shape[1], model.weights.shape[1]
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

    form: LinearForm
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
    return _solve_independently_with(_newton_step, residual, inputs, starts, newton_steps)


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
    unknown. The solve is matrix-free through lineax by default, so it never forms
    the dense Jacobian, which is what lets it reach a large unknown.
    """

    form: LinearForm
    unknown_domain: Domain
    rank: int | None = eqx.field(static=True, default=None)
    variance: float = eqx.field(static=True, default=0.99)
    newton_steps: int = eqx.field(static=True, default=8)
    matrix_free: bool = eqx.field(static=True, default=True)


def _fit_exact_solve(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: ExactSolve,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Solve the residual for every sample, then fit the reduced output map."""
    starts = sampling.sample(method.unknown_domain, inputs.shape[0], seed=_START_SEED)
    step = _newton_step_matrix_free if method.matrix_free else _newton_step
    unknowns = _solve_independently_with(step, residual, inputs, starts, method.newton_steps)
    return fit_data(
        inputs,
        unknowns,
        method.form,
        holdout=holdout,
        ridge=ridge,
        reduce_outputs=reduction.pca_outputs(rank=method.rank, variance=method.variance),
    )


def _solve_independently_with(
    step: Callable[[Residual, Float[Array, " n"], Float[Array, " m"]], Float[Array, " m"]],
    residual: Residual,
    inputs: Float[Array, "n d"],
    starts: Float[Array, "n m"],
    newton_steps: int,
) -> Float[Array, "n m"]:
    """Iterate a Newton step over every sample from its own start, in parallel."""
    unknowns = starts
    for _ in range(newton_steps):
        unknowns = jax.vmap(lambda x, y: step(residual, x, y))(inputs, unknowns)
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

    See Benner, Gugercin, and Willcox on projection-based model-order reduction,
    https://doi.org/10.1137/130932715, and Hesthaven and Ubbiali on the distilled
    POD-NN, https://doi.org/10.1016/j.jcp.2018.02.037.
    """

    form: LinearForm
    unknown_domain: Domain
    rank: int = eqx.field(static=True)
    snapshots: int = eqx.field(static=True, default=64)
    newton_steps: int = eqx.field(static=True, default=8)
    reduced_steps: int = eqx.field(static=True, default=8)
    matrix_free: bool = eqx.field(static=True, default=True)


def _pod_basis(
    snapshots: Float[Array, "s m"], rank: int
) -> tuple[Float[Array, " m"], Float[Array, "r m"]]:
    """The mean and the leading `rank` principal directions of the snapshots."""
    center = jnp.mean(snapshots, axis=0)
    _, _, directions = jnp.linalg.svd(snapshots - center, full_matrices=False)
    return center, directions[:rank]


def _reduced_newton_step(
    residual: Residual,
    x: Float[Array, " n"],
    z: Float[Array, " r"],
    center: Float[Array, " m"],
    basis: Float[Array, "r m"],
) -> Float[Array, " r"]:
    """One Newton step on the Galerkin-projected residual in the latent.

    The reduced residual is `basis @ g(x, center + basis.T @ z)`, so its Jacobian
    is the `r` by `r` matrix `basis @ (dg/dy) @ basis.T`, small regardless of the
    unknown size.
    """

    def reduced_residual(latent: Float[Array, " r"]) -> Float[Array, " r"]:
        return basis @ residual(x, center + basis.T @ latent)

    jacobian = jax.jacobian(reduced_residual)(z)
    return z - jnp.linalg.solve(jacobian, reduced_residual(z))


def _fit_reduced_order(
    residual: Residual,
    inputs: Float[Array, "n d"],
    method: ReducedSolve,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Build a POD basis from snapshots, reduce-solve every sample, and distill."""
    snapshot_inputs = inputs[: method.snapshots]
    starts = sampling.sample(method.unknown_domain, snapshot_inputs.shape[0], seed=_START_SEED)
    step = _newton_step_matrix_free if method.matrix_free else _newton_step
    states = _solve_independently_with(
        step, residual, snapshot_inputs, starts, method.newton_steps
    )
    center, basis = _pod_basis(states, method.rank)

    def reduced_solve(x: Float[Array, " n"]) -> Float[Array, " r"]:
        latent = jnp.zeros(method.rank)
        for _ in range(method.reduced_steps):
            latent = _reduced_newton_step(residual, x, latent, center, basis)
        return latent

    latents = jax.vmap(reduced_solve)(inputs)
    projection = reduction.LinearReduction(
        encode_matrix=basis,
        encode_bias=-basis @ center,
        decode_matrix=basis.T,
        decode_bias=center,
    )
    return _distill_latent_map(inputs, latents, projection, method.form, holdout, ridge)


def _distill_latent_map(
    inputs: Float[Array, "n d"],
    latents: Float[Array, "n r"],
    projection: reduction.LinearReduction,
    form: LinearForm,
    holdout: float,
    ridge: float,
) -> FittedSurrogate:
    """Fit the input-to-latent map and decode through the POD projection."""
    holdout_count = max(1, round(inputs.shape[0] * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]
    train_latents, holdout_latents = latents[:-holdout_count], latents[-holdout_count:]

    input_normalizer = normalize.from_moments(train_inputs)
    latent_normalizer = normalize.from_moments(train_latents)
    normalized_inputs = jax.vmap(input_normalizer.forward)(train_inputs)
    normalized_latents = jax.vmap(latent_normalizer.forward)(train_latents)
    model = fit_linear(form, normalized_inputs, normalized_latents, ridge=ridge)
    _, output_unravel = ravel_example(projection.inverse(train_latents[0]))

    surrogate = FittedSurrogate(
        input_reduction=reduction.identity(inputs.shape[1]),
        input_normalizer=input_normalizer,
        model=model,
        output_normalizer=latent_normalizer,
        output_reduction=projection,
        output_unravel=output_unravel,
        domain=box_from_data(inputs),
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
    inverse_map = fit_data(
        augmented, on_branch_unknowns, method.form, holdout=holdout, ridge=ridge
    )
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
        residuals = jax.vmap(
            lambda x, z: residual(x, weights @ method.form.features(z))
        )(inputs, normalized)
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
    if unknown_domain.dimension > 32:
        form: LinearForm = Affine() if affine else ChebyshevPolynomial(degree=2)
        return ExactSolve(form=form, unknown_domain=unknown_domain)
    if affine:
        return InverseMap(form=Affine(), unknown_domain=unknown_domain, newton_steps=1)
    return InverseMap(
        form=ChebyshevPolynomial(degree=3), unknown_domain=unknown_domain, newton_steps=3
    )


def fit_implicit(
    residual: Residual,
    domain: Domain,
    method: InverseMap | Objective | ExactSolve | ReducedSolve | None = None,
    *,
    unknown_domain: Domain | None = None,
    samples: int = 512,
    holdout: float = 0.2,
    ridge: float = 1e-8,
    refine_steps: int = 0,
    seed: int = 0,
) -> ImplicitSurrogate | FittedSurrogate | RefinedSurrogate:
    """Fit a surrogate `f` with `residual(x, f(x))` near zero, through a method.

    Samples the domain, builds the surrogate through the strategy `method`, and
    reports the residual the surrogate leaves on held-out inputs. A positive
    `refine_steps` polishes each prediction with Newton steps on the residual at
    call time, which every method's result can use. Leaving `method` out picks
    one from the residual's structure, which needs `unknown_domain`.
    """
    if method is None:
        if unknown_domain is None:
            raise ValueError("fit_implicit needs a method or an unknown_domain to recommend one")
        method = recommend_method(residual, domain, unknown_domain, seed=seed)

    inputs = sampling.sample(domain, samples, seed=seed)
    holdout_count = max(1, round(samples * holdout))
    train_inputs, holdout_inputs = inputs[:-holdout_count], inputs[-holdout_count:]

    match method:
        case InverseMap():
            base: ImplicitSurrogate | FittedSurrogate = _fit_inverse_map(
                residual, train_inputs, method, holdout, ridge
            )
        case Objective():
            base = _fit_objective(residual, domain, train_inputs, method)
        case ExactSolve():
            base = _fit_exact_solve(residual, train_inputs, method, holdout, ridge)
        case ReducedSolve():
            base = _fit_reduced_order(residual, train_inputs, method, holdout, ridge)
        case _:
            raise TypeError(f"unsupported implicit method: {type(method).__name__}")

    surrogate: ImplicitSurrogate | FittedSurrogate | RefinedSurrogate = base
    if refine_steps > 0:
        surrogate = refine(base, residual, steps=refine_steps)

    predicted = jax.vmap(surrogate)(holdout_inputs)
    left = jnp.linalg.norm(jax.vmap(residual)(holdout_inputs, predicted), axis=1)
    report = _value_report(
        jnp.mean(left), jnp.max(left), inputs.shape[1], base.report.structure_cost
    )
    return eqx.tree_at(lambda s: s.report, surrogate, report)
