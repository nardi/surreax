<div align="center">

  <h2 align="center">
  <code>surreax</code> - Surrogate model training for JAX functions
  </h2>

</div>

## Introduction

`surreax` fits a surrogate to a function written in JAX. You give it the
function and a region where the surrogate should be accurate, and it returns a
second function that behaves approximately like the first one, but has a form
you chose in advance.

The form is the point. Some forms are easy to reason about after you fit them.
A piecewise linear surrogate can be handed to a mixed-integer solver. A convex
surrogate can be minimized reliably. A general neural network can do neither.
`surreax` treats the form as the thing you care about, and accuracy as the
thing to maximize once the form is fixed.

Install with pip:

```bash
pip install surreax
```

and [read the documentation here](https://nardi.github.io/surreax).

## Example

<!-- name: test_readme_example -->
```python
import jax
import jax.numpy as jnp
import surreax as sx


def target(x):
    return (x[0] ** 2 + x[1] ** 2)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=2), samples=128)

# The surrogate keeps the target's signature and is an ordinary JAX function.
prediction = surrogate(jnp.array([0.5, 0.5]))
assert abs(float(prediction[0]) - 0.5) < 1e-3

# It differentiates and vectorizes like anything else, because it is a pytree.
jacobian = jax.jacobian(surrogate)(jnp.array([0.5, 0.5]))
assert jacobian.shape == (1, 2)
```

## Why?

There are good surrogate libraries already. `surreax` aims at a narrower goal,
which leads to a few different choices:

1. It is built on JAX, so the target's exact Jacobian is available and cheap.
2. It cares about the form of the result, not only its accuracy, so a
   downstream solver can use the structure it keeps.
3. It fits in closed form wherever the form allows, with no learning rate and a
   deterministic answer.
4. It reports error by source, splitting value error from derivative error.

## Status

`surreax` fits explicit targets over a box, with the affine, Chebyshev
polynomial, convex max-affine, and difference-of-max-affine forms. It can train
against target derivatives, fit in a reduced space through PCA or an active
subspace, select a sparse basis by matching pursuit, and add samples where the
fit is worst. The report separates value, derivative, and projection error,
searches the worst case over the domain, and gives the structure cost. Implicit
targets and a neural form are planned.

## Development

`surreax` is formatted and linted with Ruff, type-checked with `ty`, and tested
with pytest, hypothesis, and pytest-benchmark at 100% branch coverage. Every
code example in the documentation is executed by markdown-pytest.
