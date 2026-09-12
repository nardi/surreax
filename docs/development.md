# Development

A few notes on how the library is put together, and where it sits next to
similar tools.

## The pipeline

A fitted surrogate is a composition, applied on a single example.

```
flatten -> encode -> normalize -> model -> denormalize -> decode -> unflatten
```

Every stage is stored with the fitted object, not run as a separate
preprocessing step you have to remember. The encode and decode stages are
linear maps. Composing a linear map with a piecewise linear or convex model
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

Because so much rides on structural typing, all library code is fully typed,
and jaxtyping with beartype checks array shapes at runtime during tests.

## Code layout

- Each pipeline stage is its own module, `flatten`, `domain`, `sampling`,
  `normalize`, `reduction`, `fitting`, `report`, `fitted`.
- The model forms live under `forms`, with the protocol in `forms/protocol.py`.
- `api` holds the `fit` entry point, and the common names re-export from the
  top-level `surreax` package.

## Style

- All code is formatted and linted with Ruff, and type-checked with `ty`.
- Tests use pytest, with hypothesis for numerical invariants and
  pytest-benchmark for the hot paths. Branch coverage is kept at 100%.
- Every code example on this site is executed by markdown-pytest in CI, so an
  example that breaks fails the build.

## Where this sits next to other tools

`surreax` overlaps with several existing libraries but aims at a narrower goal.

- SMT fits general surrogates in Python with a derivative emphasis, but it is
  not built on JAX and takes no interest in the form of the result.
- ALAMO builds surrogates so they can be used in algebraic optimization, but it
  reaches tractability through sparsity rather than a chosen structural form.
- OMLT turns already-trained models into optimization formulations, so it
  starts where this library ends.
