# Model forms

A model form is the shape the surrogate is allowed to take. `surreax` fits the
most accurate surrogate it can within that shape, so the form is your lever on
the trade-off between accuracy and how easy the result is to reason about.

## What a form is

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
