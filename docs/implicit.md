# Implicit targets

Sometimes you cannot write the target down. You have a residual `g(x, y)` and
want `f` with `g(x, f(x))` near zero, but no closed form for `f`. `fit_implicit`
handles this without solving `g = 0` at every point. You pass it a `method`
object that picks the strategy, and it returns a fitted surrogate.

## The inverse map

The default method inverts the residual in its second argument. Writing
`r = g(x, y)`, the map `(x, y) -> (x, r)` is inverted to `y = h(x, r)`, and then
`f(x) = h(x, 0)`. The inverse is cheap to sample. Draw `(x, y)`, evaluate
`r = g(x, y)` forward, and the pair `((x, r), y)` is a labeled sample of `h`. So
`h` is fitted as an ordinary regression over the augmented input `(x, r)`, and
`f` reads it at the zero-residual slice. No root solve is involved.

A Newton step pulls the sampled unknowns near the solution manifold first, so
the residuals cluster near zero where the slice is read. It is one linear solve
per point, not an iterative solve. `InverseMap` carries the form, the domain the
unknown is drawn from, and the Newton step count.

## An affine residual

When `g` is affine in `y`, the inverse map is exactly linear in `r`, so a linear
form recovers the zero-residual slice exactly. Here `g(x, y) = A y - x`, so
`f(x) = A^{-1} x`.

<!-- name: test_implicit_affine -->
```python
import jax.numpy as jnp
import surreax as sx

matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])


def residual(x, y):
    return matrix @ y - x


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
method = sx.InverseMap(form=sx.Affine(), unknown_domain=sx.box([-1.0, -1.0], [1.0, 1.0]))
surrogate = sx.fit_implicit(residual, domain, method, samples=256)

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
method = sx.InverseMap(
    form=sx.ChebyshevPolynomial(degree=4), unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2
)
surrogate = sx.fit_implicit(residual, domain, method, samples=512)

# The surrogate leaves a small residual across the domain.
assert float(surrogate.report.value_mean) < 0.5
```

The residual the surrogate leaves is in `surrogate.report.value_mean`. A
multi-solution residual needs the branch handling planned for a later version,
so the current fit assumes a unique solution over the domain.

## Branches and folds

A residual can have several solution branches, and it can fold, at a point where
its Jacobian in the unknown turns singular and no function of the input exists.
The fit keeps one branch, chosen by the sign of `det(dg/dy)` at the most interior
sample, so a two-branch residual is not averaged into a meaningless middle. The
`fold` report says how much was dropped as off-branch, and flags a fold when the
smallest determinant falls below the tolerance.

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
method = sx.InverseMap(
    form=sx.ChebyshevPolynomial(degree=4), unknown_domain=sx.box([-1.1], [1.1]), newton_steps=6
)
surrogate = sx.fit_implicit(residual, domain, method, samples=256)

# The surrogate sits on one branch, so every prediction shares a sign.
points = jnp.linspace(0.3, 0.95, 20)[:, None]
predictions = jax.vmap(surrogate)(points)[:, 0]
assert bool(jnp.all(predictions > 0)) or bool(jnp.all(predictions < 0))
assert float(surrogate.fold.off_branch) > 0.3
```

Turning `continuation` on orders the samples along the first input coordinate and
warm-starts each solve from the previous solution, which follows one branch even
across a fold. It suits a residual whose branches meet, where independent solves
from random starts would scatter across both.

## Iterative refinement

Any surrogate can polish each prediction with a few Newton steps on the
residual, so a cheap solve from the surrogate's guess drives `g(x, f(x))` to zero
at call time. Set `refine_steps` on `fit_implicit` to turn it on, and it wraps
whatever the method produced.

<!-- name: test_implicit_refinement -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


domain = sx.box([-2.0], [2.0])
method = sx.InverseMap(
    form=sx.ChebyshevPolynomial(degree=4), unknown_domain=sx.box([-1.2], [1.2]), newton_steps=2
)

base = sx.fit_implicit(residual, domain, method, samples=512)
refined = sx.fit_implicit(residual, domain, method, samples=512, refine_steps=3)
assert float(refined.report.value_mean) < float(base.report.value_mean)
```

## The refinement primitives

Refinement is also available on its own, without a fit. `refine` wraps any
callable so its output is polished by Newton steps at call time, and `root_find`
is the same wrapper read as solving the residual from a warm start. Both are
plain functions of a residual and a starting guess, so they work on a fitted
surrogate or on a crude analytic guess.

<!-- name: test_implicit_root_find -->
```python
import jax.numpy as jnp
import surreax as sx


def residual(x, y):
    return y**3 + y - x


# A rough constant guess, driven to the root by Newton steps at call time.
solver = sx.root_find(residual, lambda x: x / 2.0, steps=8)
point = jnp.array([2.0])
assert float(jnp.abs(residual(point, solver(point))[0])) < 1e-5
```

## Fitting from pairs

When you already hold input-output pairs rather than a callable, `fit_data`
fits a linear form to them directly. It normalizes from the sample moments,
since the pairs carry no domain box, and reports the held-out prediction error.
The inverse map uses it internally, and it is useful on its own for a distilled
dataset.

<!-- name: test_implicit_fit_data -->
```python
import jax
import jax.numpy as jnp
import surreax as sx

key = jax.random.key(0)
inputs = jax.random.uniform(key, (200, 3), minval=-1.0, maxval=1.0)
weights = jnp.array([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]])
outputs = jax.vmap(lambda x: weights @ x + jnp.array([0.5, -1.0]))(inputs)

surrogate = sx.fit_data(inputs, outputs, sx.Affine())
assert float(surrogate.report.value_max) < 1e-3
```

## Large sparse systems

When the input is small but the unknown is large, a physical model discretized
over many states, the `ExactSolve` method fits `f` at a scale the inverse map
cannot reach. It rests on two facts. The residual Jacobian in the unknown is
sparse, so the Newton solve is cheap when the sparsity is used, and `f` maps a
small input to a large unknown, so the solved unknowns trace a manifold no wider
than the input. `ExactSolve` solves `g = 0` at each sample, collapses the
solved unknowns onto their leading principal directions, and fits the map into
that small reduced space.

Each Newton step factors the sparse Jacobian by default.
[asdex](https://github.com/adrhill/asdex) reads the sparsity pattern from the
residual and colors it, so the Jacobian costs one Jacobian-vector product per
color. [splineax](https://github.com/nardi/splineax) then factors it with a
sparse direct solver. A `StencilResidual` states its pattern in its stencil
table, so it skips the tracing. The solve cost follows the number of nonzeros,
not the square of the state size.

### Linear solvers

The `linear_solver` field on `ExactSolve`, `ReducedSolve` and `AutoencoderSolve`
sets how a Newton step solves its Jacobian system.

- `"sparse"`, the default, factors the sparse Jacobian. When the residual has no
  usable pattern, because asdex cannot read one of its operations or the pattern
  is close to dense, it falls back to `"matrix_free"`. `residual_pattern` returns
  the pattern it would use, or None.
- `"matrix_free"` finds the step by GMRES from Jacobian-vector products alone,
  through [lineax](https://arxiv.org/abs/2311.17283), and never forms the
  Jacobian. It is the route when even a sparse factorization does not fit in
  memory.
- `"dense"` forms the full Jacobian, which suits a small unknown.

`recommend_linear_solver` picks one from the residual, and `recommend_method` sets
it on the `ExactSolve` it returns.

### Precision

The direct solver follows the JAX precision you set, and surreax never changes
the JAX configuration. On CPU with x64 enabled it is KLU, which reuses the
symbolic analysis and the pivot order from one Newton step to the next. Without
x64 it is `Spsolve`, which solves in single precision and factors every step from
scratch. A large implicit fit on CPU should turn x64 on at startup with
`jax.config.update("jax_enable_x64", True)`. On a CUDA GPU with the splineax
`cudss` extra installed it is cuDSS, in either precision.

A `reuse` setting, a `splineax.HybridSettings`, keeps the factorization of an
earlier Newton step as the preconditioner of an iterative solve and only factors
again when that solve is slow. It only pays off where a factorization is
expensive next to a solve, which is a 3D grid. In the benchmarks it took about
five times as long as refactoring on 2D plates of 64, 128 and 256 points a side,
but 0.85 times as long on a 16 by 16 by 16 block and 0.58 times on a 24 by 24 by
24 block, since the factorization of a 3D grid fills in far more. It is off by
default.

### Derivative supervision

A positive `derivative_weight` on `ExactSolve` also fits the Jacobian of the
reduced coefficients in the input. The implicit function theorem gives it from
one transposed solve per coefficient at each solved sample, reusing the solver of
the Newton steps, so the full Jacobian of the unknown is never formed.

Here the unknown is a field of 64 states coupled by a nearest-neighbor
Laplacian, driven by three inputs, so the solution lives on a rank-3 manifold.

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
method = sx.ExactSolve(
    form=sx.Affine(), unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, newton_steps=5
)
surrogate = sx.fit_implicit(residual, domain, method, samples=128)

# The 64 unknowns are predicted through 3 reduced coefficients.
assert surrogate.output_reduction.encode_matrix.shape[0] == 3
assert float(surrogate.report.value_mean) < 1e-3
```

## The reduced-order model

`ExactSolve` solves the full residual once per sample. When many samples are
needed, or the solve is to be reused online, `ReducedSolve` solves in a small
latent space instead. It builds a proper orthogonal decomposition basis from a
few solved snapshots, projects the residual onto that basis in the manner of
Galerkin, and solves the reduced residual by Newton in the latent. Each reduced
step is an `r` by `r` solve, independent of the unknown size. The snapshot
solves use the `linear_solver` of the previous section. It then distills
the input-to-latent map into an explicit surrogate, so inference needs no solve.
This is the non-intrusive reduced-order model, POD-NN, of Hesthaven and Ubbiali,
[POD-NN](https://doi.org/10.1016/j.jcp.2018.02.037).

The basis is linear here, which suits a solution manifold whose Kolmogorov
n-width decays fast. A strongly nonlinear manifold needs a nonlinear autoencoder
basis, which is not built yet. The latent dimension is the `rank`, and the
residual falls as it reaches the intrinsic dimension of the manifold.

Setting `greedy` chooses the snapshots one at a time where the current basis
leaves the most residual, rather than taking the first `snapshots` samples, so it
reaches a given accuracy with fewer full solves. It helps most when the hard
states sit in a small part of the input space that uniform sampling underweights.

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
# Only 16 full solves build the basis, then every sample is reduce-solved.
method = sx.ReducedSolve(
    form=sx.Affine(), unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16
)
surrogate = sx.fit_implicit(residual, domain, method, samples=128)

assert surrogate.output_reduction.decode_matrix.shape == (64, 3)
assert float(surrogate.report.value_mean) < 1e-3
```

### The reduced solver on its own

`reduced_solver` returns the solver rather than the distilled map, so you have the
cheap latent solve and its exact input derivative without a fit. The derivative
comes through the implicit function theorem, not by differentiating the Newton
steps, so it is exact and cheap. The solver's `report` says where the reduced
problem is ill-conditioned or has several latent roots, so an untrustworthy solve
is flagged rather than returned in silence.

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
method = sx.ReducedSolve(
    form=sx.Affine(), unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16
)
solver = sx.reduced_solver(residual, domain, method, samples=64)

point = jnp.array([0.4, -0.2, 0.6])
field = solver(point)
assert field.shape == (64,)
# The field derivative in the input is a full Jacobian from the latent solve.
assert jax.jacobian(solver)(point).shape == (64, 3)
# One well-conditioned root over this linear system.
assert not solver.report.multiple_roots
```

## A learned manifold without solved states

The reduced-order model above needs a few full solves to build its basis. When
even those are too expensive, `AutoencoderSolve` learns the manifold from the
residual alone. A decoder maps a small latent to the full state, and the reduced
solve picks the latent whose decoded state leaves the least residual, by
Gauss-Newton on `||g(x, D(z))||`. That is the least-squares Petrov-Galerkin
projection of Carlberg and colleagues,
[LSPG](https://arxiv.org/abs/2012.06073). Training samples inputs,
reduce-solves them, and lowers the residual the decoded states leave, with the
derivative through the inner solve taken by the implicit function theorem. It
only ever evaluates `g` forward, so no full state is solved.

An encoder is trained next to the decoder so that `E(D(z)) = z`, which lets you
map a known state back into the latent. The decoder is nonlinear, so it can
follow a manifold that bends, in the manner of Lee and Carlberg,
[nonlinear-manifold ROMs](https://doi.org/10.1016/j.jcp.2019.108973). The price is
that the result is no longer a linear program, and the training is slower than a
closed-form fit.

`autoencoder_solver` returns the trained solver on its own, and `fit_implicit`
distills it into a `DecodedSurrogate` that predicts the latent and decodes it, so
inference needs no solve. Like the network form, it holds an activation function,
so jit it with `equinox.filter_jit`.

<!-- name: test_implicit_autoencoder -->
```python
import jax.numpy as jnp
import surreax as sx

grid = jnp.linspace(0.0, 1.0, 32)
modes = jnp.stack([jnp.sin(2 * jnp.pi * grid), jnp.cos(2 * jnp.pi * grid)])


def residual(x, y):
    # A rod heated by two source modes that sheds heat with a cubic term.
    laplacian = -2.0 * y + jnp.roll(y, 1) + jnp.roll(y, -1)
    return -laplacian + 0.5 * y + 0.3 * y**3 - x @ modes


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
method = sx.AutoencoderSolve(
    form=sx.ChebyshevPolynomial(degree=3), unknown_dimension=32, rank=2, steps=400
)
solver = sx.autoencoder_solver(residual, domain, method)

# The residual the learned solve leaves, relative to the residual at zero.
assert float(solver.training_residual) < 0.2
assert solver(jnp.array([0.4, -0.2])).shape == (32,)
```

### A moving front, and when to leave the linear basis

Some manifolds are hard to reach from the residual alone. Picture fluid in a pipe
flowing past a heater you can slide along it. The fluid is cold before the heater
and warm after it, so the solution is a sharp step whose position follows the
heater. Two numbers, the heater position and the flow rate, fix the state, but a
linear basis needs about one mode per step position. On 64 points, eight linear
modes still leave most of the residual.

Training from a random start does not find that step. Setting `snapshots` solves
that many full states first. The autoencoder starts from their linear basis and
is trained to reconstruct them, then the residual training refines it. With 16
snapshots, a rank-2 learned manifold leaves about a third of the residual that
eight linear modes leave. A sharp feature also gives the latent solve local
minima, so every solve starts from a small learned network, the `LatentGuess`,
that predicts where the solve ends up.

`recommend_reduced_method` makes the choice for you. It solves the snapshots a
`ReducedSolve` asks for and reads how much of their variation its rank captures,
with `linear_energy`. When the linear basis leaves out almost nothing, it keeps
the linear model, which is simpler and exact. Otherwise it returns an
`AutoencoderSolve` at the same rank, pretrained on the same snapshots.

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
method = sx.ReducedSolve(
    form=sx.Affine(), unknown_domain=sx.box([-3.0] * 64, [3.0] * 64), rank=3, snapshots=16
)
# Three sources span this field exactly, so the linear model stays.
assert sx.recommend_reduced_method(residual, domain, method) is method
```

## Hyper-reduction

A reduced solve works in a small latent, but each Newton step still evaluates the
whole residual, so its cost grows with the state. `hyper_reduce` removes that.
It takes residual snapshots near the manifold, keeps a small residual basis, and
picks a few residual entries from which that basis can be recovered, by the
discrete empirical interpolation method of Chaturantabut and Sorensen,
[DEIM](https://doi.org/10.1137/090766498). The latent solve then minimizes the
residual rebuilt from those entries alone, the GNAT method of Carlberg and
colleagues, [GNAT](https://doi.org/10.1016/j.jcp.2013.02.028).

Reading a few entries only saves work if the residual can be evaluated entry by
entry. A `StencilResidual` gives it that shape. You write one entry function and
a table of neighbor indices, so entry `i` reads the state only at `stencil[i]`.
The hyper-reduced solve then decodes the state on those neighbors and nothing
else. It reads the same number of state entries whatever the size of the state,
and it works under both the linear solver and the learned-manifold one.

Here a rod radiates heat only at three bare patches, so the nonlinear part of its
residual lives on a few points, and the chosen entries gather there.

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
    form=sx.Affine(),
    unknown_domain=sx.box([-2.0] * points, [2.0] * points),
    rank=6,
    snapshots=32,
)
solver = sx.reduced_solver(residual, domain, method, samples=64)
hyper = sx.hyper_reduce(solver, residual, domain)

# The latent solve reads a few dozen of the 256 state entries.
assert hyper.rows.shape[0] < 64
assert hyper(jnp.full(4, 0.5)).shape == (points,)
```

`recommend_hyper_reduction` makes the choice for you. It returns the solver
unchanged when the residual is not a `StencilResidual`, or when the sampled
stencils would read more than half the state. Otherwise it hyper-reduces and
checks the result on held-out inputs. It keeps the hyper-reduced solver only if
the residual it leaves is at most `tolerance` times that of the full solver,
1.2 by default.

## The objective route

There is also an experimental route that never inverts anything. It fits `f`
directly by minimizing the residual it leaves, `mean ||g(x, f(x))||^2`. This
suits a residual whose solve is expensive, but the objective is indifferent
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
