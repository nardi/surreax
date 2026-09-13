# Implicit targets

Sometimes you cannot write the target down. You have a residual `g(x, y)` and
want `f` with `g(x, f(x))` near zero, but no closed form for `f`. `fit_implicit`
handles this without solving `g = 0` at every point.

## The inverse map

The trick is to invert the residual in its second argument. Writing
`r = g(x, y)`, the map `(x, y) -> (x, r)` is inverted to `y = h(x, r)`, and then
`f(x) = h(x, 0)`. The inverse is cheap to sample: draw `(x, y)`, evaluate
`r = g(x, y)` forward, and the pair `((x, r), y)` is a labeled sample of `h`. So
`h` is fitted as an ordinary regression over the augmented input `(x, r)`, and
`f` reads it at the zero-residual slice. No root solve is involved.

A Newton step pulls the sampled unknowns near the solution manifold first, so
the residuals cluster near zero where the slice is read. It is one linear solve
per point, not an iterative solve.

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
unknown_domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit_implicit(residual, domain, unknown_domain, sx.Affine(), samples=256)

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
unknown_domain = sx.box([-1.2], [1.2])
form = sx.ChebyshevPolynomial(degree=4)
surrogate = sx.fit_implicit(residual, domain, unknown_domain, form, samples=512, newton_steps=2)

# The surrogate leaves a small residual across the domain.
assert float(surrogate.report.value_mean) < 0.5
```

The residual the surrogate leaves is in `surrogate.report.value_mean`. A
multi-solution residual needs the branch handling planned for a later version,
so the current fit assumes a unique solution over the domain.
