# Model forms

A model form is the shape the surrogate is allowed to take. `surreax` fits the
most accurate surrogate it can within that shape, so choosing the form sets the
trade-off between accuracy and how easy the result is to reason about.

## The form interface

A form supplies a feature map. For a form that is linear in its parameters, the
output is `weights @ features(x)`, one row of `weights` per output coordinate.
That single method is the whole interface, described by the `LinearForm`
protocol, so you can drop in your own form without subclassing anything.

<!-- name: test_forms_protocol -->
```python
import jax.numpy as jnp
import surreax as sx

assert isinstance(sx.Affine(), sx.LinearForm)
assert isinstance(sx.ChebyshevPolynomial(degree=3), sx.LinearForm)
```

## Affine

The affine form is the input with a constant prepended. It recovers an affine
target exactly, which makes it a baseline and a sanity check.

<!-- name: test_forms_affine -->
```python
import jax.numpy as jnp
import surreax as sx

coefficients = jnp.array([2.0, -1.0])


def linear_target(x):
    return (coefficients @ x + 3.0)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(linear_target, domain, sx.Affine(), samples=128)
assert float(surrogate.report.value_max) < 1e-3
```

## Polynomial in a Chebyshev basis

The polynomial form uses products of Chebyshev polynomials over every exponent
tuple whose total degree is at most `degree`. It is linear in its parameters,
so it fits in one solve, and it converges quickly for smooth targets. A target
that is itself a polynomial of low enough degree comes back exactly.

<!-- name: test_forms_polynomial -->
```python
import jax.numpy as jnp
import surreax as sx


def cubic(x):
    return (x[0] ** 2 - 2 * x[0] * x[1] + 0.5 * x[1] ** 3)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(cubic, domain, sx.ChebyshevPolynomial(degree=3), samples=256)
assert float(surrogate.report.value_max) < 1e-3
```

Raising the degree lowers the error on a smooth target, at the cost of more
parameters. The report is how you watch that trade-off.

## Max-affine

The max-affine form is the maximum of several affine functions. It is convex
and piecewise linear, so a downstream solver can minimize it as a linear
program, which is why it is the recommended structured form.

The maximum makes the form nonlinear in its parameters, so there is no
closed-form fit. `surreax` fits it by alternating minimization, which assigns
each sample to the piece that attains the maximum and refits each piece by
least squares, in the manner of k-means, from several random restarts. The
method is from Magnani and Boyd, [Convex piecewise-linear
fitting](https://web.stanford.edu/~boyd/papers/pdf/cvx_pwl_fit.pdf),
Optimization and Engineering 10(1), 2009. A gradient route, from [Max-Affine
Regression via First-Order Methods](https://arxiv.org/abs/2308.06276), is
available with `method="gradient"`.

You choose the number of pieces. The report gives the structure cost, the piece
count, which sets how expensive the surrogate is to reason about later.

<!-- name: test_forms_max_affine -->
```python
import jax.numpy as jnp
import surreax as sx


def convex(x):
    return jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(convex, domain, sx.MaxAffineForm(pieces=2), samples=256)
assert float(surrogate.report.value_max) < 1e-2
assert surrogate.report.structure_cost == 2
```

## Difference of max-affine

Subtracting one convex max-affine function from another gives a general
piecewise linear function, so this form fits a target that is not convex, which
a single max-affine cannot. The fit also carries the split into a convex and a
concave part, which a difference-of-convex optimization algorithm works on.

The difference is not convex and not linear in its parameters, so it fits by
gradient descent through optimistix from several restarts, each started from a
plain max-affine fit of the positive part. You choose the number of pieces per
part, and the structure cost is the total across both.

<!-- name: test_forms_difference -->
```python
import jax.numpy as jnp
import surreax as sx


def wave(x):
    # Neither convex nor concave, so a single max-affine cannot fit it.
    positive = jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])
    negative = jnp.maximum(-x[0], x[1])
    return (positive - negative)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
form = sx.DifferenceOfMaxAffineForm(pieces=2, restarts=6, steps=400)
surrogate = sx.fit(wave, domain, form, samples=400, worst_case_starts=0)
assert float(surrogate.report.value_max) < 1e-2
assert surrogate.report.structure_cost == 4
```

## Network

The network form is a feedforward multilayer perceptron with a smooth
activation. It is not a structured form, so a solver cannot reason about it, but
it is the accuracy reference that shows how much accuracy the structured forms
cost, and the fallback when none of them fits well. It is not linear in its
parameters, so it fits by gradient descent through optimistix. Its structure
cost is the parameter count.

Because the network holds an activation function as a leaf, jit it with
`equinox.filter_jit` or a closure rather than passing it as a raw `jax.jit`
argument.

<!-- name: test_forms_network -->
```python
import jax.numpy as jnp
import surreax as sx


def smooth(x):
    return (jnp.sin(2 * x[0]) + x[1] ** 2)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
form = sx.NetworkForm(width=32, depth=2, steps=800)
surrogate = sx.fit(smooth, domain, form, samples=256, worst_case_starts=0)
assert float(surrogate.report.value_mean) < 0.02
```

## Grid

The grid form places a regular grid over the input box and interpolates
multilinearly between the nodes. It is linear in the node values, so it fits by
one linear solve, and the error falls as the grid refines. The node count grows
as the resolution to the power of the input dimension, so it is best after a
reduction, when the reduced input is small.

<!-- name: test_forms_grid -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return (jnp.sin(2 * x[0]) + x[1] ** 2)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(target, domain, sx.Grid(resolution=16), samples=512, worst_case_starts=0)
assert float(surrogate.report.value_mean) < 0.02
```

`export` returns a grid surrogate as its node coordinates and values, so a solver
can encode the interpolation.

## Solver export

The point of a piecewise linear form is that a solver can reason about it. A
max-affine surrogate is a maximum of affine pieces, so it is a linear program,
and a difference of two is a pair of them. `export` returns the slopes and
intercepts as plain arrays, in the domain's flat coordinates, so you can write
the encoding in whatever optimization framework you use. For a box those are the
inputs themselves, and for a `LogRange` input it is its logarithm. The library exports arrays
and nothing else, to stay free of a solver dependency. For the mixed-integer and
linear-program encodings these arrays feed, see
[OMLT](https://arxiv.org/abs/2202.02414).

<!-- name: test_forms_export -->
```python
import jax
import jax.numpy as jnp
import surreax as sx


def target(x):
    return jnp.maximum(2 * x[0] - x[1], x[0] + 2 * x[1])[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(target, domain, sx.MaxAffineForm(pieces=2, restarts=6), worst_case_starts=0)
arrays = sx.export(surrogate)

# The exported pieces reproduce the surrogate on the original input.
point = jnp.array([0.3, -0.4])
by_hand = jnp.max(jnp.einsum("kmd,d->km", arrays.slopes, point) + arrays.intercepts, axis=1)
assert jnp.allclose(by_hand, surrogate(point), atol=1e-4)
```
