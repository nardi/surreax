<div align="center">

  <h2 align="center">
  <code>surreax</code>: surrogate model training for JAX functions
  </h2>

</div>

## Introduction

`surreax` fits a surrogate to a function written in JAX. You give it the
function and a region where the surrogate should be accurate, and it returns a
second function that behaves approximately like the first one, but has a form
you chose in advance.

You fix the form up front because some forms are easy to reason about after you
fit them. A piecewise linear surrogate can be passed to a mixed-integer solver.
A convex surrogate can be minimized reliably. A general neural network can do
neither. `surreax` treats the form as the thing you care about, and accuracy as
the thing to maximize once the form is fixed.

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

# The surrogate has the target's signature and is an ordinary JAX function.
prediction = surrogate(jnp.array([0.5, 0.5]))
assert abs(float(prediction[0]) - 0.5) < 1e-3

# It is a pytree, so it can be differentiated and vectorized like any JAX function.
jacobian = jax.jacobian(surrogate)(jnp.array([0.5, 0.5]))
assert jacobian.shape == (1, 2)
```

## Motivation

Good surrogate libraries exist already, so `surreax` has a narrower goal, which
leads to a few different choices:

1. It is built on JAX, so the target's exact Jacobian is available and cheap.
2. It cares about the form of the result, not only its accuracy, so a downstream
   solver can use its structure.
3. It fits in closed form wherever the form allows, with no learning rate and a
   deterministic answer.
4. It reports error by source, splitting value error from derivative error.

## Status

`surreax` fits explicit targets with the affine, Chebyshev polynomial, convex
max-affine, difference-of-max-affine, grid, and feedforward network forms. Two
ReLU network forms, an input-convex one and a general one, are piecewise linear
too. The piecewise linear forms can be exported as arrays for a solver. A domain
can mix boxes with log-scaled, normal, integer, and categorical inputs, or be
inferred from a dataset, in which case the surrogate reports how far a query
point is from the data.

A fit can train against target derivatives and work in a reduced space, found by
PCA, an active subspace, or exact pruning. It can select a sparse basis by
matching pursuit, split a separable target into blocks, add samples where its
error is largest, and satisfy an equality constraint exactly. If you leave out
the form, `fit` picks the form, its polynomial degree, the reductions, and the
sparse budget from the target itself. The error report contains the value,
derivative, and projection error as separate entries, the worst-case error from
a search over the domain, and the structure cost.

It also fits implicit targets given as a residual `g(x, y)`. The methods are the
inverse map, a direct sparse Newton solve with output reduction, a linear
reduced-order model, and a manifold learned from the residual alone. With
hyper-reduction, the online solve evaluates only a few entries of the residual.

## Development

`surreax` is formatted and linted with Ruff, type-checked with `ty`, and tested
with pytest, hypothesis, and pytest-benchmark at 100% branch coverage. Every
code example in the documentation is executed by markdown-pytest.

## Testing

As this is a model fitting library, tests may take a significant time to run.
The recommended way to validate new code is to start with the quickest tests and
work outward.

```bash
# While editing: the unit tests, without the slow ones.
uv run pytest tests/unit -m "not slow"

# For the area you changed: its unit and integration files.
uv run pytest tests/unit/test_implicit.py tests/integration/test_sparse_solve.py

# Before committing: every integration test, and the documentation examples.
uv run pytest tests/integration docs README.md
```

The tests are in `tests/unit`, `tests/integration` and `tests/benchmarks`, and
each directory has a matching marker. A test that takes about ten seconds or
more also has the `slow` marker. Each benchmark fits its system before the timed
part, so the benchmarks are slow, and `-m "not benchmark"` deselects them. The
[test systems](https://nardi.github.io/surreax/test-systems/) page describes the
physical systems that the integration tests fit.

The full run with branch coverage, `uv run pytest --cov --cov-branch`, took
about 50 minutes in one measured run. CI runs it and fails when branch coverage
is below 100%, so run it locally only when a change adds a branch that the
focused runs may not reach.
