# Implicit targets

When you cannot write the target as an explicit function, you may still have a
residual `g(x, y)` and want `f` with `g(x, f(x))` near zero. The input `x` is a
vector, the unknown `y` is a vector, and the residual is a vector of the same
length as `y` that is zero when `y` solves the model for `x`. A discretized heat
equation is an example, with a few heater powers as the input and the
temperature at every point of a rod as the unknown. The test suite has several,
described on the [test systems](test-systems.md) page, and the [heated
rod](test-systems.md#heated-rod) is the simplest.

`fit_implicit` fits such an `f` without a closed form for it. You pass it a
`method` object that picks the strategy, and it returns a fitted surrogate. If
you leave out the method, `fit_implicit` picks one from the residual, which
needs an `unknown_domain` to draw unknowns from. The residual takes the input as
a flat array, so the input domain must be a `Box`. Several examples below are
small versions of those systems. The
[terminology](terminology.md#implicit-targets) page defines the terms of this
page, such as the exact solve and the reduced solve. It also explains what is
implicit about the target and what is not about the
[result](terminology.md#explicit-and-implicit-surrogates).

## The form of the fitted map

Every method but [`Objective`](terminology.md#objective-method) fits a linear
form to some map, and `fit_implicit` takes that form as its fourth argument. For
an [`InverseMap`](terminology.md#inverse-map) the map goes from the input and
the residual to the unknown. For an [`ExactSolve`](terminology.md#exact-solve)
it goes from the input to the reduced unknown, and for a
[`ReducedSolve`](terminology.md#reduced-solve) or an
[`AutoencoderSolve`](terminology.md#autoencoder-solve) it goes from the input to
the [latent](terminology.md#latent). If you leave out the form, `fit_implicit`
picks one with `recommend_implicit_form`. A residual that is affine in the
unknown gets an `Affine` form. Any other residual gets a `ChebyshevPolynomial`
of degree 3 for an `InverseMap` and of degree 2 for an `ExactSolve` or a
`ReducedSolve`. An `AutoencoderSolve` always gets degree 3, since its latent
coordinates are learned. An `Objective` contains the form that it trains, so
passing a form as well raises an error.

## The inverse map

The `InverseMap` method inverts the residual in its second argument. Writing
`r = g(x, y)`, the map `(x, y) -> (x, r)` is inverted to `y = h(x, r)`, and then
`f(x) = h(x, 0)`. The inverse is cheap to sample. Draw `(x, y)`, evaluate
`r = g(x, y)` forward, and the pair `((x, r), y)` is a labeled sample of `h`. So
`h` is fitted as an ordinary regression over the augmented input `(x, r)`, and
`f` evaluates it on the zero-residual slice without a root solve.

Newton steps first move the sampled unknowns toward the solution manifold, so
the residuals are concentrated near zero, where `h` is evaluated. Each step is
one linear solve per point, and the default is one step. `InverseMap` stores the
domain that the unknown is drawn from and the Newton step count.

## An affine residual

When `g` is affine in `y`, the inverse map is exactly linear in `r`, so a linear
form fits the zero-residual slice exactly. Here `g(x, y) = A y - x`, so
`f(x) = A^{-1} x`.

<!-- name: test_implicit_affine -->
```python
import jax.numpy as jnp
import surreax as sx

matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])


def residual(x, y):
    return matrix @ y - x


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
method = sx.InverseMap(unknown_domain=sx.box([-1.0, -1.0], [1.0, 1.0]))
surrogate = sx.fit_implicit(residual, domain, method, sx.Affine(), samples=256)

point = jnp.array([0.4, -0.2])
assert jnp.allclose(surrogate(point), jnp.linalg.solve(matrix, point), atol=1e-3)
```

## A nonlinear residual

For a nonlinear residual the inverse map is nonlinear, so a polynomial form
approximates it, and the Newton step matters more. Here `g(x, y) = y**3 + y - x`
is monotone in `y`, so the solution is unique.

<!-- name: test_implicit_nonlinear -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


domain = sx.box([-2.0], [2.0])
method = sx.InverseMap(unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2)
surrogate = sx.fit_implicit(
    residual, domain, method, sx.ChebyshevPolynomial(degree=4), samples=512
)

# The residual at the surrogate's predictions is small across the domain.
assert float(surrogate.report.value_mean) < 0.5
```

The mean residual at the surrogate's predictions is
`surrogate.report.value_mean`. A residual with several solutions needs the
branch handling of the next section.

## Branches and folds

A residual can have several solution branches, and it can fold, at a point where
its Jacobian in the unknown becomes singular and no function of the input
exists. The fit keeps one branch, chosen by the sign of `det(dg/dy)` at the
sample where its magnitude is largest, which is the sample furthest from a fold.
The two branches of a two-branch residual are then not averaged into a
meaningless middle. The `fold` report of the `InverseMap` result contains the
fraction of samples dropped as off-branch, and it flags a fold when the smallest
determinant is below `fold_tolerance`, 1e-3 by default.

Here `g(x, y) = y**2 - x` has two branches, `y = +sqrt(x)` and `y = -sqrt(x)`,
and folds at `x = 0`.

<!-- name: test_implicit_branches -->
```python
import jax
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**2 - x


domain = sx.box([0.25], [1.0])
method = sx.InverseMap(unknown_domain=sx.box([-1.1], [1.1]), newton_steps=6)
surrogate = sx.fit_implicit(
    residual, domain, method, sx.ChebyshevPolynomial(degree=4), samples=256
)

# The surrogate is on one branch, so every prediction has the same sign.
points = jnp.linspace(0.3, 0.95, 20)[:, None]
predictions = jax.vmap(surrogate)(points)[:, 0]
assert bool(jnp.all(predictions > 0)) or bool(jnp.all(predictions < 0))
assert float(surrogate.fold.off_branch) > 0.3
```

Turning `continuation` on orders the samples along the first input coordinate
and warm-starts each solve from the previous solution, which follows one branch
even across a fold. It is meant for a residual whose branches meet, where
independent solves from random starts would end up on both branches.

## Iterative refinement

The prediction of any surrogate can be improved with a few Newton steps on the
residual, so a cheap solve that starts from the surrogate's prediction brings
`g(x, f(x))` close to zero at call time. Pass `refine_steps` to `fit_implicit`
to enable it, and the refinement wraps the surrogate of any method.

<!-- name: test_implicit_refinement -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


domain = sx.box([-2.0], [2.0])
method = sx.InverseMap(unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2)
form = sx.ChebyshevPolynomial(degree=4)

base = sx.fit_implicit(residual, domain, method, form, samples=512)
refined = sx.fit_implicit(residual, domain, method, form, samples=512, refine_steps=3)
assert float(refined.report.value_mean) < float(base.report.value_mean)
```

## Refinement functions

Refinement is also available on its own, without a fit. `refine` wraps any
callable so that its output is improved by Newton steps at call time, and
`root_find` is the same wrapper, used to solve the residual from a starting
guess. Both are plain functions of a residual and a starting guess, so they work
on a fitted surrogate or on a crude analytic guess.

<!-- name: test_implicit_root_find -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


# A rough linear guess, which Newton steps move to the root at call time.
solver = sx.root_find(residual, lambda x: x / 2.0, steps=8)
point = jnp.array([2.0])
assert float(jnp.abs(residual(point, solver(point))[0])) < 1e-5
```

## Pairs instead of a callable

The inverse map fits its regression with `fit_data`, which fits a linear form to
input-output pairs and needs no callable. The [basic
workflow](getting-started.md#input-output-pairs) page describes it.

## Large sparse systems

When the input is small but the unknown is large, a physical model discretized
over many states, the [`ExactSolve`](terminology.md#exact-solve) method fits `f`
for unknowns larger than the inverse map can handle. That works because the
residual Jacobian in the unknown is sparse, so the Newton solve is cheap when
the sparsity is used, and because `f` maps a small input to a large unknown, so
the solved unknowns are on a manifold whose dimension is at most that of the
input. `ExactSolve` solves `g = 0` at each sample, projects the solved unknowns
onto their leading principal directions, and fits the map from the input to that
small reduced space.

Each Newton step factors the sparse Jacobian by default.
[asdex](https://github.com/adrhill/asdex) detects the sparsity pattern of the
Jacobian by tracing the residual and colors it, so computing the Jacobian takes
one Jacobian-vector product per color.
[splineax](https://github.com/nardi/splineax) then factors it with a sparse
direct solver. A [`StencilResidual`](terminology.md#stencil-residual), described
under [hyper-reduction](#hyper-reduction), has its pattern in its stencil table,
so no tracing is needed. The cost of the solve depends on the number of
nonzeros, not the square of the state size.

Here the unknown is a field of 64 states coupled by a nearest-neighbor
Laplacian, driven by three inputs, so the solution is on a rank-3 manifold. It
is a small version of the [heated rod](test-systems.md#heated-rod) of the test
systems, with three heater profiles.

<!-- name: test_implicit_exact_solve -->
```python
import jax
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 64)
shapes = jnp.stack([jnp.sin(jnp.pi * grid), jnp.sin(2 * jnp.pi * grid), grid * (1 - grid)])


def residual(x, y):
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ shapes


domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
method = sx.ExactSolve(unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, newton_steps=5)
surrogate = sx.fit_implicit(residual, domain, method, sx.Affine(), samples=128)

# The 64 unknowns are predicted through 3 reduced coefficients.
assert surrogate.output_reduction.encode_matrix.shape[0] == 3
assert float(surrogate.report.value_mean) < 1e-3
```

### Linear solvers

The `linear_solver` field on `ExactSolve`, `ReducedSolve` and `AutoencoderSolve`
sets how the Jacobian system of a [Newton step](terminology.md#newton-step) is
solved, as described in the [linear solver](terminology.md#linear-solver) entry.

- `"sparse"`, the default, factors the sparse Jacobian. When the residual has no
  usable pattern, because asdex cannot trace one of its operations or the
  pattern is close to dense, `"matrix_free"` is used instead. `residual_pattern`
  returns the pattern that `"sparse"` would use, or `None`.
- `"matrix_free"` finds the step by GMRES from Jacobian-vector products alone,
  through [lineax](https://arxiv.org/abs/2311.17283), and never forms the
  Jacobian. It is the option for a system whose sparse factorization does not
  fit in memory.
- `"dense"` forms the full Jacobian, which suits a small unknown.

`recommend_linear_solver` picks one from the residual, and `recommend_method`
sets it on the `ExactSolve` it returns.

### Precision

The direct solver depends on the JAX precision that you set, and `surreax` never
changes the JAX configuration. On CPU with x64 (double precision) enabled it is
KLU, a sparse direct solver from SuiteSparse that reuses the symbolic analysis
and the pivot order from one Newton step to the next. It is Pardiso instead when
the splineax `pardiso` extra is installed. Without x64 it is `Spsolve`, the
sparse solver of JAX, which solves in single precision and factors every step
from scratch. For a large implicit fit on CPU, enable x64 at startup with
`jax.config.update("jax_enable_x64", True)`. On a CUDA GPU with the splineax
`cudss` extra installed it is cuDSS, in either precision.

A `reuse` setting, a `splineax.HybridSettings`, keeps the factorization of an
earlier Newton step as the preconditioner of an iterative solve and only factors
again when that solve is slow. It is only faster where a factorization is
expensive compared to a solve, which is the case on a 3D grid. In timings on the
[radiating plate and block](test-systems.md#radiating-plate-and-block) systems,
the reuse took about five times as long as refactoring on 2D plates of 64, 128
and 256 points a side, but 0.85 times as long on a 16 by 16 by 16 block and 0.58
times on a 24 by 24 by 24 block, since the factorization of a 3D grid fills in
far more. Those timings are from one CPU, and timings on other hardware will
differ. The benchmark
[`test_benchmark_newton_solve_against_linear_solver`](https://github.com/nardi/surreax/blob/main/tests/benchmarks/test_hot_paths.py#L170)
repeats the comparison on a plate of 64 points a side and a block of 16, and the
[test systems](test-systems.md#radiating-plate-and-block) page describes both.
The setting is off by default.

### Derivative supervision

A positive `derivative_weight` on `ExactSolve` also fits the Jacobian of the
reduced coefficients in the input. This Jacobian is computed with the implicit
function theorem, from one transposed solve per coefficient at each solved
sample with the solver of the Newton steps, so the full Jacobian of the unknown
is never formed.

### Convergence of the Newton solves

Newton diverges from a start far from a root of a stiff residual, such as one
with exponentials. The basis and the fit then use states that are not roots, and
no other part of the fit detects this. `ExactSolve`, `ReducedSolve` and
`AutoencoderSolve` therefore check every full solve and raise an error if one
did not converge. A solve has converged when its residual norm is at most
`convergence_tolerance` times the residual norm at the zero state, which is 1e-3
by default. The check also runs under `jit`, through `equinox.error_if`.

The reference is the zero state and not the start of the solve. The residual at
a far start can be very large, so a ratio to it would pass a solve that is still
far from a root.

A smaller `unknown_domain` or more `newton_steps` fixes most failures. Near a
fold, Newton converges only linearly, and a deliberately ill-conditioned system
may need `convergence_tolerance=None` to skip the check.

## The reduced-order model

`ExactSolve` solves the full residual once per sample. When many samples are
needed, or the solve is to be reused online,
[`ReducedSolve`](terminology.md#reduced-solve) solves in a small latent space
instead. It builds a [proper orthogonal decomposition](terminology.md#pod-basis)
basis from a few solved [snapshots](terminology.md#snapshot), projects the
residual onto that basis by Galerkin projection, and solves the reduced residual
by [Newton](terminology.md#newton-step) in the latent space. Each reduced step
is an `r` by `r` solve, independent of the unknown size, and the snapshot solves
use the `linear_solver` of the previous section. The method then distills the
input-to-latent map into an explicit surrogate, so inference needs no solve.
This is the non-intrusive reduced-order model, POD-NN, of Hesthaven and Ubbiali,
[POD-NN](https://doi.org/10.1016/j.jcp.2018.02.037).

`reduced_solver` builds the solver without a surrogate, so it takes no form.

The basis is linear here, which suits a solution manifold whose [Kolmogorov
n-width](terminology.md#kolmogorov-n-width) decays fast. The n-width measures
how well `n` linear modes can approximate the set of all solutions, and it
decays fast when a few modes suffice. A strongly curved manifold needs the
learned manifold of a later section. The latent dimension is the `rank`, and the
residual decreases as the rank approaches the intrinsic dimension of the
manifold.

With `greedy` set, the method chooses the snapshots one at a time, each at the
sample where the residual of the current basis is largest, rather than taking
the first `snapshots` samples, so it reaches a given accuracy with fewer full
solves. A `snapshot_tolerance` above zero stops the selection once the largest
remaining residual is below it. Greedy selection helps most when the hard states
are in a small part of the input space, which has few samples under uniform
sampling.

<!-- name: test_implicit_reduced_order -->
```python
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 64)
shapes = jnp.stack([jnp.sin(jnp.pi * grid), jnp.sin(2 * jnp.pi * grid), grid * (1 - grid)])


def residual(x, y):
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ shapes


domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
# The basis is built from 16 full solves, and each sample is solved in the reduced space.
method = sx.ReducedSolve(unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16)
surrogate = sx.fit_implicit(residual, domain, method, sx.Affine(), samples=128)

assert surrogate.output_reduction.decode_matrix.shape == (64, 3)
assert float(surrogate.report.value_mean) < 1e-3
```

### The reduced solver on its own

`reduced_solver` returns the solver rather than the distilled map, so you have
the cheap latent solve and its exact input derivative without a fit. The
derivative is computed with the implicit function theorem, not by
differentiating the Newton steps, so it is exact and cheap. The solver's
`report` flags where the reduced problem is ill-conditioned or has several
latent roots, so an untrustworthy solve is flagged rather than returned in
silence.

<!-- name: test_implicit_reduced_solver -->
```python
import jax
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 64)
shapes = jnp.stack([jnp.sin(jnp.pi * grid), jnp.sin(2 * jnp.pi * grid), grid * (1 - grid)])


def residual(x, y):
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ shapes


domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
method = sx.ReducedSolve(unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16)
solver = sx.reduced_solver(residual, domain, method, samples=64)

point = jnp.array([0.4, -0.2, 0.6])
field = solver(point)
assert field.shape == (64,)
# The field derivative in the input is a full Jacobian from the latent solve.
assert jax.jacobian(solver)(point).shape == (64, 3)
# This linear system has one well-conditioned root.
assert not solver.report.multiple_roots
```

## A learned manifold without solved states

The reduced-order model above needs a few full solves to build its basis. When
even those are too expensive, `AutoencoderSolve` learns the manifold from the
residual alone. A decoder maps a small latent vector to the full state, and the
reduced solve picks the latent vector whose decoded state has the smallest
residual, by Gauss-Newton on `||g(x, D(z))||`. That is the least-squares
Petrov-Galerkin projection of Carlberg and colleagues,
[LSPG](https://arxiv.org/abs/2012.06073). Each training step samples inputs,
solves the reduced problem for them, and lowers the residual of the decoded
states, with the derivative through the inner solve computed by the implicit
function theorem. By default the training only evaluates `g` forward, so no full
state is solved.

An encoder is trained next to the decoder so that `E(D(z)) = z`, which lets you
map a known state back into the latent. The decoder is nonlinear, so it can
represent a curved manifold, in the manner of Lee and Carlberg,
[nonlinear-manifold ROMs](https://doi.org/10.1016/j.jcp.2019.108973). The result
is then no longer a linear program, and the training is slower than a
closed-form fit.

`autoencoder_solver` returns the trained solver on its own. `fit_implicit`
distills it, with its `form` argument, into a `DecodedSurrogate` that predicts
the latent and decodes it, so inference needs no solve. Like the network form,
it has an activation function as a field, so compile it with
`equinox.filter_jit`.

<!-- name: test_implicit_autoencoder -->
```python
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 32)
modes = jnp.stack([jnp.sin(2 * jnp.pi * grid), jnp.cos(2 * jnp.pi * grid)])


def residual(x, y):
    # A rod heated by two source modes that loses heat through a cubic term.
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y + 0.3 * y**3 - x @ modes


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
method = sx.AutoencoderSolve(unknown_dimension=32, rank=2, steps=400)
solver = sx.autoencoder_solver(residual, domain, method)

# The residual of the learned solve, relative to the residual at zero.
assert float(solver.training_residual) < 0.2
assert solver(jnp.array([0.4, -0.2])).shape == (32,)
```

### A moving front and the limit of the linear basis

Some manifolds are hard to find from the residual alone. Consider fluid in a
pipe that flows past a heater at an adjustable position along the pipe, which is
the [hot fluid pipe](test-systems.md#hot-fluid-pipe) of the test systems. The
fluid is cold before the heater and warm after it, so the solution is a sharp
step at the heater position. Two numbers, the heater position and the flow rate,
determine the state, but a linear basis needs about one mode per step position.
On 64 points, most of the residual remains with eight linear modes.

Training from a random start does not find that step. With `snapshots` set, the
method solves that many full states first. The autoencoder starts from their
linear basis and is trained to reconstruct them, and the training on the
residual then continues from there. With 16 snapshots, the residual of a rank-2
learned manifold is about a third of the residual with eight linear modes. A
sharp feature also causes local minima in the latent solve, so every solve
starts from the prediction of a small learned network, the `LatentGuess`, which
predicts the result of the solve.

`recommend_reduced_method` makes the choice for you. It solves the snapshots
that a `ReducedSolve` specifies and computes with `linear_energy` the fraction
of their variance that a linear basis of its rank explains. When that fraction
is almost one, it returns the linear model, which is simpler and exact.
Otherwise it returns an `AutoencoderSolve` at the same rank, pretrained on the
same snapshots.

<!-- name: test_implicit_recommend_reduced -->
```python
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 64)
shapes = jnp.stack([jnp.sin(jnp.pi * grid), jnp.sin(2 * jnp.pi * grid), grid * (1 - grid)])


def residual(x, y):
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y - x @ shapes


domain = sx.box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
method = sx.ReducedSolve(unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16)
# Three sources span this field exactly, so the method is returned unchanged.
assert sx.recommend_reduced_method(residual, domain, method) is method
```

## Hyper-reduction

A [reduced solve](terminology.md#reduced-solve) works in a small latent space,
but each Newton step still evaluates the whole residual, so its cost increases
with the state size. `hyper_reduce` removes that dependence. It takes residual
snapshots near the manifold, keeps a small residual basis, and picks a few
residual entries from which that basis can be recovered, by the discrete
empirical interpolation method of Chaturantabut and Sorensen,
[DEIM](https://doi.org/10.1137/090766498). The latent solve then minimizes the
residual reconstructed from those entries alone, which is the GNAT method of
Carlberg and colleagues, [GNAT](https://doi.org/10.1016/j.jcp.2013.02.028).

Reading a few entries only saves work if the residual can be evaluated entry by
entry. A `StencilResidual` has that structure. You write one entry function and
a table of neighbor indices, so entry `i` depends on the state only at the
indices `stencil[i]`. The hyper-reduced solve then decodes the state only at
those neighbors. The number of decoded state entries does not depend on the size
of the state, and hyper-reduction works with both the linear reduced solver and
the learned-manifold solver.

Here a rod radiates heat only at three bare patches, which is the [rod with hot
spots](test-systems.md#rod-with-hot-spots) of the test systems, so the nonlinear
part of its residual is nonzero only near a few points, and the chosen entries
are concentrated there.

<!-- name: test_implicit_hyper_reduction -->
```python
import jax.numpy as jnp
import surreax as sx

points = 256
position = jnp.arange(points) / (points - 1)
patches = sum(jnp.exp(-(((position - p) * points / 1.5) ** 2)) for p in (0.25, 0.5, 0.75))
heaters = jnp.stack([jnp.sin((k + 1) * jnp.pi * position) for k in range(4)])
index = jnp.arange(points)[:, None] + jnp.array([-1, 0, 1])
inside = ((index >= 0) & (index < points)).astype(float)


def entry(x, i, near):
    left, here, right = near * inside[i]
    radiation = 3.0 * patches[i] * here * jnp.abs(here) ** 3
    return -(left - 2.0 * here + right) + 0.5 * here + radiation - x @ heaters[:, i]


residual = sx.StencilResidual(stencil=jnp.clip(index, 0, points - 1), entry=entry)
domain = sx.box([0.0] * 4, [1.0] * 4)
method = sx.ReducedSolve(
    unknown_domain=sx.box([-2.0] * points, [2.0] * points), rank=6, snapshots=32
)
solver = sx.reduced_solver(residual, domain, method, samples=64)
hyper = sx.hyper_reduce(solver, residual, domain)

# The latent solve uses a few dozen of the 256 state entries.
assert hyper.rows.shape[0] < 64
assert hyper(jnp.full(4, 0.5)).shape == (points,)
```

`recommend_hyper_reduction` makes the choice for you. It returns the solver
unchanged when the residual is not a `StencilResidual`, or when the stencils of
the sampled entries would include more than half the state. Otherwise it
hyper-reduces the solver and checks the result on held-out inputs. It returns
the hyper-reduced solver only if its residual is at most `tolerance` times the
residual of the solver without hyper-reduction, 1.2 by default.

## Equations that hold exactly

With the methods above, every equation of the residual is satisfied
approximately. `sx.Completed(method, constraint)` wraps any of them so that a
chosen subset of the equations holds to solver precision, by solving for the
outputs that those equations determine at call time. The [equality
constraints](techniques.md#equality-constraints) section describes the
completion.

## The objective method

There is also an experimental method that never inverts anything. It fits `f`
directly by minimizing the residual at its predictions, `mean ||g(x, f(x))||^2`.
This suits a residual whose solve is expensive, but the objective is indifferent
between solution branches, so it can converge to a biased mixture where the
residual has several solutions. Prefer `InverseMap` unless the solve is the
bottleneck. Pass an `Objective` method to select it.

<!-- name: test_implicit_objective -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


domain = sx.box([-2.0], [2.0])
method = sx.Objective(form=sx.ChebyshevPolynomial(degree=4), unknown_dimension=1, steps=600)
surrogate = sx.fit_implicit(residual, domain, method, samples=256)
assert float(surrogate.report.value_mean) < 0.15
```
