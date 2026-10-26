# Model forms

A model form is the shape the surrogate is allowed to take. `surreax` fits the
most accurate surrogate it can within that shape, so choosing the form sets the
trade-off between accuracy and how easy the result is to reason about. You pass
the form as the third argument of `fit`, and the [basic
workflow](getting-started.md) page describes the other arguments and the report
that every example below reads. The [terminology](terminology.md) page defines
the terms of this page, such as the linear form and the structure cost.

| Form | Fitted by | Structure of the result |
| --- | --- | --- |
| `Affine` | One linear solve | Affine |
| `ChebyshevPolynomial` | One linear solve | Polynomial of a chosen degree |
| `Grid` | One linear solve | Piecewise linear, multilinear on a grid |
| `MaxAffineForm` | Alternating minimization or gradient descent | Convex piecewise linear |
| `DifferenceOfMaxAffineForm` | Gradient descent | Piecewise linear |
| `ConvexNetworkForm` | Gradient descent | Convex piecewise linear |
| `ReluNetworkForm` | Gradient descent | Piecewise linear |
| `NetworkForm` | Gradient descent | Smooth, with no structure to exploit |

Every form takes the number of pieces, units or nodes that it may use, and the
report's `structure_cost` says how many it used. The piecewise linear forms
export as arrays, see [solver export](#solver-export).

## The form interface

A form supplies a feature map. For a form that is linear in its parameters, the
output is `weights @ features(x)`, one row of `weights` per output coordinate.
That single method is the whole interface, described by the `LinearForm`
protocol, so you can drop in your own form without subclassing anything. The
`Affine`, `ChebyshevPolynomial` and `Grid` forms satisfy it, and the other forms
are not linear in their parameters.

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
tuple whose total degree is at most `degree`. It is linear in its parameters, so
it fits in one solve, and it converges quickly for smooth targets. A target that
is itself a polynomial of low enough degree comes back exactly.

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

The max-affine form is the maximum of several affine functions. It is convex and
piecewise linear, so a downstream solver can minimize it as a linear program.
Automatic selection picks it with four pieces for a target that is piecewise
linear.

The maximum makes the form nonlinear in its parameters, so there is no
closed-form fit. `surreax` fits it by alternating minimization, which assigns
each sample to the piece that attains the maximum and refits each piece by least
squares, in the manner of k-means, from several random restarts. The method is
from Magnani and Boyd, [Convex piecewise-linear
fitting](https://web.stanford.edu/~boyd/papers/pdf/cvx_pwl_fit.pdf),
Optimization and Engineering 10(1), 2009. A gradient route, from [Max-Affine
Regression via First-Order Methods](https://arxiv.org/abs/2308.06276), is
available with `method="gradient"`.

You choose the number of pieces. The report gives the [structure
cost](terminology.md#structure-cost), the piece count, which sets how expensive
the surrogate is to reason about later.

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

## Convex network

`ConvexNetworkForm` is an input-convex neural network, from Amos, Xu and Kolter,
[Input Convex Neural Networks](https://arxiv.org/abs/1609.07400). Its hidden
layers use ReLU activations, its weights from one hidden layer to the next are
nonnegative, and every layer also reads the input directly. Each output is then
a convex piecewise linear function of the input, which is the same set of
functions as the max-affine form. A solver minimizes it as a linear program with
one auxiliary variable per hidden unit, so the program grows with the number of
units, not with the number of affine pieces.

It differs from a max-affine fit in size. The sum `|x0| + ... + |x5|` of six
inputs has 64 affine pieces as a maximum, and a network writes it with about a
dozen units. The nonnegative weights are the softplus of free parameters, and
the fit is gradient descent through optimistix from several random starts,
keeping the best. `depth` is the number of hidden layers.

<!-- name: test_forms_convex_network -->
```python
import jax.numpy as jnp
import surreax as sx


def spread(x):
    return jnp.sum(jnp.abs(x))[None]


domain = sx.box([-1.0] * 3, [1.0] * 3)
form = sx.ConvexNetworkForm(width=8, depth=2, steps=1200)
surrogate = sx.fit(spread, domain, form, samples=512, worst_case_starts=0)
assert float(surrogate.report.value_mean) < 0.05

# The exported layers act on the original input, with nonnegative hidden weights.
export = sx.export(surrogate)
assert all(bool(jnp.all(weights >= 0)) for weights in export.hidden_weights)
```

## ReLU network

`ReluNetworkForm` is a plain multilayer perceptron with ReLU activations. It
represents every continuous piecewise linear function, the same set as the
difference of max-affine form, so it fits targets that are neither convex nor
concave. A mixed-integer program encodes it with one binary variable per hidden
unit, each with a pair of big-M constraints, which switch an inequality on or
off with that binary variable through a bound on the expression. It trains the
same way as the convex network.

Each network form represents the same functions as its max-affine counterpart,
so the choice is about size and about how well the fit trains. The comparison
below comes from the [test systems](test-systems.md) and the form comparison
tests, with mean errors on held-out points. The three systems are described on
that page. The [radiating rod](test-systems.md#radiating-rod) is a rod with
heaters that also loses heat by radiation. The [thermostat
rod](test-systems.md#thermostat-rod) has thermostat-controlled heaters, which
makes its response neither convex nor concave. The [diode
network](test-systems.md#diode-network) is a mesh of resistors and diodes. The
target of a fit is the [latent](terminology.md#latent) coefficients of a
[reduced solver](test-systems.md#reduced-solvers-as-explicit-targets) of the
system, which are the few numbers that describe all of its temperatures or
voltages, as a function of two to four inputs.

- A target that is a sum of ridge functions in many inputs favors the networks.
  On `|x0 - x1| - |x2 + 0.3 x3| + 0.5 x4` in six inputs, a ReLU network with 253
  parameters reaches a mean error of 4e-4, and a difference of max-affine with
  24 and 24 pieces and 336 parameters reaches 0.33. On the sum of absolute
  values the convex network reaches 0.036, the ReLU network 1e-3, and 48
  max-affine pieces 0.16.
- A smooth response in few inputs favors the max-affine forms at a small size.
  For the latent coefficients of the radiating rod and the thermostat rod, a
  difference of max-affine with 4 and 4 pieces matched a ReLU network with about
  twice the parameters. On the four-input diode network, 8 and 8 pieces reached
  0.0044 against 0.0094 for a ReLU network with a similar parameter count. A
  wider ReLU network did overtake the pieces on the radiating rod, at 0.0096
  with 354 parameters against 0.017 with 96.
- A convex network cannot follow a response that is not convex. On the radiating
  rod and the thermostat rod it left more than three times the error of a ReLU
  network.
- The convex network trains less reliably than a plain one. On the mildly
  nonconvex diode network it reached 0.017, no better than max-affine with 16
  pieces at 0.012, and a wider network did worse at 0.025, so try a few widths.

<!-- name: test_forms_relu_network -->
```python
import jax.numpy as jnp
import surreax as sx


def difference(x):
    return (jnp.abs(x[0] - 0.2) - jnp.abs(x[1] + 0.3))[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
form = sx.ReluNetworkForm(width=8, depth=2, steps=1500)
surrogate = sx.fit(difference, domain, form, samples=512, worst_case_starts=0)
assert float(surrogate.report.value_mean) < 0.05

# The exported layers reproduce the surrogate on the original input.
export = sx.export(surrogate)
point = jnp.array([0.3, -0.6])
activation = point
for weights, biases in zip(export.weights[:-1], export.biases[:-1], strict=True):
    activation = jnp.maximum(weights @ activation + biases, 0.0)
by_hand = export.weights[-1] @ activation + export.biases[-1]
assert jnp.allclose(by_hand, surrogate(point), atol=1e-4)
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

`export` returns a grid surrogate as its node coordinates and values, so a
solver can encode the interpolation. The export needs an identity input
reduction, so a grid that is fitted after a reduction cannot be exported in the
original coordinates.

## Solver export

The point of a piecewise linear form is that a solver can reason about it. A
max-affine surrogate is a maximum of affine pieces, so it is a linear program,
and a difference of two is a pair of them. `export` returns the slopes and
intercepts as plain arrays, in the domain's flat coordinates, so you can write
the encoding in whatever optimization framework you use. The network forms
export their layer matrices the same way. For a box the flat coordinates are the
inputs themselves, and for a `LogRange` input the coordinate is its logarithm.
The [terminology](terminology.md#flat-coordinates) page defines the flat
coordinates. The export folds the input and output normalization into the
arrays, so they act on the original input and produce the original output. It
needs an identity output reduction, because a reduced output space mixes the
outputs, and it raises an error otherwise.

The library exports arrays and nothing else, to stay free of a solver
dependency. For the mixed-integer and linear-program encodings these arrays
feed, see [OMLT](https://arxiv.org/abs/2202.02414). The test suite also holds
two small encodings, which the [test
systems](test-systems.md#optimization-over-exported-surrogates) page describes.

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
