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

You fix the form up front because some forms are easy to reason about after you
fit them. A piecewise linear surrogate can be handed to a mixed-integer solver. A convex
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

## Motivation

There are good surrogate libraries already. `surreax` aims at a narrower goal,
which leads to a few different choices:

1. It is built on JAX, so the target's exact Jacobian is available and cheap.
2. It cares about the form of the result, not only its accuracy, so a
   downstream solver can use the structure it keeps.
3. It fits in closed form wherever the form allows, with no learning rate and a
   deterministic answer.
4. It reports error by source, splitting value error from derivative error.

## Status

`surreax` fits explicit targets with the affine, Chebyshev polynomial, convex
max-affine, difference-of-max-affine, grid, and feedforward network forms. Two ReLU
network forms, an input-convex one and a general one, are piecewise linear too.
The piecewise linear forms export as arrays for a solver. A domain can mix boxes
with log-scaled, normal, integer, and categorical inputs, or be inferred from a
dataset, in which case the surrogate reports how far a query lies from the data.

A fit can train against target derivatives and work in a reduced space, found by
PCA, an active subspace, or exact pruning. It can select a sparse basis by
matching pursuit, split a separable target into blocks, add samples where it is
worst, and hold an equality constraint exactly. Leaving the form out picks the form, its polynomial
degree, the reductions, and the sparse budget from the target itself. The report
separates value, derivative, and projection error, searches the worst case over
the domain, and gives the structure cost.

It also fits implicit targets given as a residual `g(x, y)`. The methods are the
inverse map, a direct sparse Newton solve with output reduction, a linear
reduced-order model, and a manifold learned from the residual alone.
Hyper-reduction makes the online solve read only a few entries of the residual.

## Development

`surreax` is formatted and linted with Ruff, type-checked with `ty`, and tested
with pytest, hypothesis, and pytest-benchmark at 100% branch coverage. Every
code example in the documentation is executed by markdown-pytest.
