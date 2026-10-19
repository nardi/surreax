# Development

A few notes on how the library is put together, and where it sits next to
similar tools.

## The pipeline

A fitted surrogate is a composition, applied on a single example.

```
flatten -> encode -> normalize -> model -> denormalize -> decode -> unflatten
```

The flatten stage is the domain's `flatten`, which also maps a log-scaled or
categorical input to its flat coordinates. Every stage is stored with the fitted
object, not run as a separate preprocessing step you have to remember. The
encode and decode stages are linear maps. Composing a linear map with a piecewise linear or convex model
keeps that structure in the original coordinates, which is why the reduction is
linear rather than a general encoder.

## Structural typing

The library leans on protocols rather than base classes.

1. A model form is anything that satisfies the `LinearForm` protocol, so the
   affine form, the polynomial form, and a form you write yourself are
   interchangeable with no shared ancestor.
2. A domain is anything that satisfies the `Domain` protocol.
3. The one class that does use inheritance is the Equinox module, and only
   because Equinox needs it to register a pytree. Those modules are data
   containers, so behavior stays in free functions and small protocols.

The library relies on structural typing, so all library code is fully typed,
and jaxtyping with beartype checks array shapes at runtime during tests.

## Code layout

- Each pipeline stage is its own module, `flatten`, `domain`, `sampling`,
  `normalize`, `reduction`, `fitting`, `report`, `fitted`.
- The model forms live under `forms`, with the protocol in `forms/protocol.py`.
  The two ReLU network forms are in `forms/relu_network.py`, and `export` turns
  the piecewise linear forms into arrays.
- `api` holds `fit`, `fit_adaptive` and `fit_constrained`, and `composite` holds
  `fit_separable`.
- `structure` reads the dependency structure from the Jacobian, `strategy` reads
  the function kind and recommends fit settings, and `degree` reads polynomial
  degrees from the jaxpr. `error_search` holds the worst-case search, and
  `completion` the equality constraints.
- `implicit` holds `fit_implicit` and its methods. `sparse_solve` holds the
  sparse direct Newton step, `autoencoder` the learned manifold, `hyperreduction`
  the sampled latent solve, and `stencil` the entry-wise residual.
- The common names re-export from the top-level `surreax` package.

## Style

- All code is formatted and linted with Ruff, and type-checked with `ty`.
- Tests use pytest, with hypothesis for numerical invariants and
  pytest-benchmark for the hot paths. Branch coverage is kept at 100%.
- Tests live in `tests/unit`, `tests/integration` and `tests/benchmarks`. A unit
  test checks one module on small inputs. An integration test fits a whole
  surrogate on a test system, or combines several modules. The directories carry
  the markers `unit`, `integration` and `benchmark`, so `pytest -m unit` and
  `pytest tests/unit` select the same tests. The shared test systems stay in
  `tests/`.
- A test that takes about ten seconds or more carries the `slow` marker. The
  command `pytest tests/unit -m "not slow"` is the quick check, and it skips
  the 50 slowest unit tests.
- The benchmarks build and fit their systems before they time anything, so the
  directory takes several minutes. `pytest -m "not benchmark"` skips it.
- Every code example on this site is executed by markdown-pytest in CI, so an
  example that breaks fails the build.

## Related tools

`surreax` overlaps with several existing libraries but aims at a narrower goal.

- SMT fits general surrogates in Python with a derivative emphasis, but it is
  not built on JAX and does not fix the form of the result.
- ALAMO builds surrogates so they can be used in algebraic optimization, but it
  reaches tractability through sparsity rather than a chosen structural form.
- OMLT turns already-trained models into optimization formulations, so it
  starts where this library ends.
- Physics-informed tools such as DeepXDE train a network to drive a residual to
  zero by penalty. `fit_implicit` fits a surrogate of a chosen form to a residual
  instead, and `fit_constrained` makes chosen equations hold exactly rather than
  by penalty.
