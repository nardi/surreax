# Development

A few notes on how the library is put together, and how it compares with similar
tools.

## The pipeline

A fitted surrogate is a composition of stages, applied to a single input.

```
flatten -> encode -> normalize -> model -> denormalize -> decode -> unflatten
```

The flatten stage is the domain's `flatten`, which also maps a log-scaled or
categorical input to its flat coordinates. Every stage is stored in the fitted
object, so no stage is a separate preprocessing step that you have to run
yourself. The encode and decode stages are linear maps. The composition of a
linear map with a piecewise linear or convex model is again piecewise linear or
convex in the original coordinates, which is why the reduction is linear rather
than a general encoder. The `export` function relies on this when it includes
the stages in the exported arrays.

## Structural typing

The library uses protocols rather than base classes.

1. A model form is anything that satisfies the `LinearForm` protocol, so the
   affine form, the polynomial form, and a form you write yourself are
   interchangeable with no shared base class.
2. A domain is anything that satisfies the `Domain` protocol.
3. The one class that does use inheritance is the Equinox module, and only
   because Equinox needs it to register a pytree. Those modules are data
   containers, so the logic is in free functions and small protocols.

All library code is fully typed. The protocols define the interfaces for the
type checker, and jaxtyping with beartype checks array shapes at runtime during
tests.

## Code layout

- Each pipeline stage is its own module, `flatten`, `domain`, `sampling`,
  `normalize`, `reduction`, `fitting`, `report`, `fitted`.
- The model forms are in `forms`, with the protocol in `forms/protocol.py`. The
  two ReLU network forms are in `forms/relu_network.py`, and `export` converts
  the piecewise linear forms to arrays.
- `api` contains `fit`, `fit_adaptive` and `fit_constrained`, and `composite`
  contains `fit_separable`.
- `structure` computes the dependency structure from the Jacobian, `strategy`
  determines the function kind and recommends fit settings, and `degree`
  computes polynomial degrees from the jaxpr. `error_search` contains the
  worst-case search, and `completion` the equality constraints.
- `implicit` contains `fit_implicit` and its methods. `sparse_solve` contains
  the sparse direct Newton step, `autoencoder` the learned manifold,
  `hyperreduction` the sampled latent solve, and `stencil` the entry-wise
  residual.
- The common names are re-exported from the top-level `surreax` package.

## Style

- All code is formatted and linted with Ruff, and type-checked with `ty`.
- Tests use pytest, with hypothesis for numerical invariants and
  pytest-benchmark for the hot paths. Branch coverage must be 100%.
- Tests are in `tests/unit`, `tests/integration` and `tests/benchmarks`. A unit
  test checks one module on small inputs. An integration test fits a whole
  surrogate on a test system, or combines several modules. The tests in these
  directories have the markers `unit`, `integration` and `benchmark`, so
  `pytest -m unit` and `pytest tests/unit` select the same tests. The shared
  test systems are in `tests/`, and the [test systems](test-systems.md) page
  describes them.
- A test that takes about ten seconds or more has the `slow` marker. The command
  `pytest tests/unit -m "not slow"` is the quick check, and it deselects the
  unit tests with that marker.
- Each benchmark builds and fits its system before the timed part, so the
  directory takes several minutes to run. `pytest -m "not benchmark"` deselects
  it.
- Every code example on this site is executed by markdown-pytest in CI, so an
  example that breaks fails the build.

## Related tools

`surreax` overlaps with several existing libraries but has a narrower goal.

- [SMT](https://github.com/SMTorg/smt) fits general surrogates in Python with a
  derivative emphasis, but it is not built on JAX and does not fix the form of
  the result.
- [ALAMO](https://doi.org/10.1002/aic.14418) builds surrogates so they can be
  used in algebraic optimization, but it makes the surrogates tractable through
  sparsity rather than through a chosen structural form.
- [OMLT](https://arxiv.org/abs/2202.02414) converts trained models to
  optimization formulations, which is the step after a surrogate is fitted with
  this library.
- Physics-informed tools such as [DeepXDE](https://arxiv.org/abs/1907.04502)
  train a network with the residual as a penalty term in the loss.
  [`fit_implicit`](implicit.md) fits a surrogate of a chosen form to a residual
  instead, and [`fit_constrained`](techniques.md#equality-constraints) makes
  chosen equations hold exactly rather than by penalty.
