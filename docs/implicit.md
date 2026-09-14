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
