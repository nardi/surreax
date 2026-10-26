# Fitting techniques

Beyond the choice of form, `fit` takes a few options that change how the
surrogate is fitted. Each one is off by default, except the worst-case search.
The examples assume the target, domain and report of the [basic
workflow](getting-started.md) and the forms of [model forms](forms.md). The
[terminology](terminology.md) page defines the terms, with their mathematics.

## Derivative supervision

Because the target is written in JAX, its Jacobian is exact and cheap. Fitting
against the Jacobian as well as the values is Sobolev training, from Czarnecki
and colleagues, [Sobolev Training for Neural
Networks](https://arxiv.org/abs/1706.04859), NeurIPS 2017. It matters when the
surrogate goes into an optimizer, since the optimizer will consume its
derivatives.

For a form that is linear in its parameters, the Jacobian is also linear in the
weights, so the derivative equations join the value equations in one least
squares solve and the fit stays closed form. A positive `derivative_weight`
turns it on. The network forms add a Jacobian term to their loss in the same
way. The max-affine and difference of max-affine forms ignore the weight, and so
does a sparse fit, since `sparse_terms` takes the fit over.

<!-- name: test_techniques_derivative -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return (jnp.sin(x[0]) + jnp.cos(2 * x[1]))[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
form = sx.ChebyshevPolynomial(degree=3)
value_only = sx.fit(target, domain, form, samples=256)
with_derivatives = sx.fit(target, domain, form, samples=256, derivative_weight=1.0)
assert float(with_derivatives.report.derivative_max) < float(value_only.report.derivative_max)
```

## Dimension reduction

`fit` can fit the model in a smaller space. On the output side this is the
larger win, because a long correlated output vector is captured by a few
coefficients. `pca_outputs` keeps the leading principal directions of the
outputs.

<!-- name: test_techniques_reduction -->
```python
import jax.numpy as jnp
import surreax as sx


def spectrum(x):
    # A six-vector output that lives in a two-dimensional subspace.
    return jnp.array([x[0], x[1], x[0] + x[1], x[0] - x[1], 2 * x[0], 2 * x[1]])


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(
    spectrum,
    domain,
    sx.Affine(),
    samples=128,
    reduce_outputs=sx.pca_outputs(rank=2),
)
assert float(surrogate.report.value_max) < 1e-3
```

The report's projection error is the accuracy that the output reduction gives up
before any fitting, measured by reconstructing the held-out outputs.

On the input side, `pca_inputs` finds the directions in which the data varies,
while `active_subspace_inputs` finds the directions in which the function
varies, from the average outer product of the target gradient. The active
subspace is from Constantine and Gleich, [Computing active subspaces with Monte
Carlo](https://arxiv.org/abs/1408.0545). An input reduction has no separate
error in the report, since it shows up in the value error. Each reducer takes a
`rank`, or keeps the fewest directions that reach a `variance` fraction, 0.99 by
default.

## Sparse basis selection

A total-degree polynomial basis grows as the number of inputs and the degree
rise, and most of its coefficients should be zero. A sparse fit keeps only the
terms that matter, which limits that growth and lowers the structure cost. This
is the route ALAMO takes, from Cozad, Sahinidis, and Miller, [Learning surrogate
models for simulation-based optimization](https://doi.org/10.1002/aic.14418),
AIChE Journal 60, 2014.

`surreax` selects the terms by orthogonal matching pursuit, which adds the term
most correlated with the residual, refits, and repeats to a budget. A
`sparse_terms` budget turns it on, and the structure cost reports how many terms
the fit kept.

<!-- name: test_techniques_sparse -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    # Depends on two of its ten inputs.
    return (2.0 * x[3] - 3.0 * x[7])[None]


domain = sx.box([-1.0] * 10, [1.0] * 10)
surrogate = sx.fit(target, domain, sx.Affine(), samples=128, sparse_terms=3)
assert float(surrogate.report.value_max) < 1e-3
assert surrogate.report.structure_cost <= 3
```

## Worst-case error

The value and derivative errors in the report are measured on a held-out sample,
so they can miss a region where the fit is worst. Because the surrogate and the
target are both differentiable, `fit` also climbs the error over the whole
domain from several starts and reports the largest it finds, along with the
point where it occurs. The search starts from the held-out points with the
largest error and fills the remaining starts at random. This is a tighter check
than a sample statistic, and `worst_case_starts=0` skips it when speed matters.
The `worst_case_steps` and `worst_case_rate` arguments set the length and the
step size of each climb, and the defaults are 8 starts, 200 steps and 0.05.

<!-- name: test_techniques_worst_case -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return (jnp.sin(3 * x[0]) * jnp.cos(3 * x[1]))[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=2), samples=128)
report = surrogate.report
# The domain search cannot do better than the held-out sample it contains.
assert float(report.worst_case) >= float(report.value_max) - 1e-5
```

## Adaptive sampling

Because the target is callable, the sample is not a fixed resource, so the fit
can add points where it is worst instead of committing to one grid.
`fit_adaptive` fits on an initial sample, locates the highest-error points with
the domain search, adds them, and refits, stopping when the worst-case error
meets a tolerance or after a round budget. This is the residual-based adaptive
refinement of the surrogate-modeling literature, such as DeepXDE, from Lu and
colleagues, [DeepXDE](https://arxiv.org/abs/1907.04502), SIAM Review 63(1),
2021.

The `tolerance` is 1e-3 by default and `rounds` is 5. The stop test reads the
worst-case error, so keep `worst_case_starts` above zero, since a value of zero
reports a worst case of zero and stops the loop at once. Each refit starts from
the previous surrogate for the forms that accept a start, which are the
max-affine and network forms.

<!-- name: test_techniques_adaptive -->
```python
import jax.numpy as jnp
import surreax as sx


def bump(x):
    return (jnp.exp(-8 * (x[0] ** 2 + x[1] ** 2)))[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
form = sx.ChebyshevPolynomial(degree=6)
adaptive = sx.fit_adaptive(
    bump, domain, form, initial_samples=128, points_per_round=32, rounds=4, worst_case_steps=120
)
fixed = sx.fit(bump, domain, form, samples=256, worst_case_steps=120)
assert float(adaptive.report.worst_case) < float(fixed.report.worst_case)
```

## Structural pruning

The Jacobian says which inputs each output depends on. An input that no output
depends on can be dropped, which is exact rather than statistical.
`prune_inputs` is an input reducer that does this, and it fits into
`reduce_inputs` like the PCA and active-subspace reducers.

<!-- name: test_techniques_pruning -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    # Depends on two of its five inputs.
    return (2 * x[0] - x[3] ** 2)[None]


domain = sx.box([-1.0] * 5, [1.0] * 5)
surrogate = sx.fit(
    target,
    domain,
    sx.ChebyshevPolynomial(degree=2),
    samples=256,
    reduce_inputs=sx.prune_inputs(),
    worst_case_starts=0,
)
assert surrogate.input_reduction.encode_matrix.shape == (2, 5)
assert float(surrogate.report.value_max) < 1e-3
```

`analyze` returns the same reading as a report. It holds the coupling matrix and
the block labels that split outputs and inputs into independent groups, so a
separable target reads as separate blocks.

<!-- name: test_techniques_analyze -->
```python
import jax.numpy as jnp
import surreax as sx


def separable(x):
    return jnp.array([x[0] + x[1], x[2] * x[3]])


structure = sx.analyze(separable, sx.box([-1.0] * 4, [1.0] * 4), samples=128)
assert structure.input_block[0] != structure.input_block[2]
```

## Separable fitting

When a target splits into independent blocks, `fit_separable` fits each block on
its own inputs and assembles the pieces, which is faster and more accurate than
one fit over everything. It reads the blocks from the structural analysis. Pass
one form for every block, or a sequence of forms in sorted-block order, so a
linear block can take a closed-form fit and a nonlinear one a richer form.

Each block searches its own inputs for its worst-case error. The blocks read
separate inputs, so their worst cases can all happen at once, and the report's
worst case is the root sum of squares of the block worst cases, at the input
that joins the block points.

The domain must be a `Box`. The report's value errors are the largest over the
blocks. Its projection errors hold a first-order bound on the error from the
coupling that the separation drops, and its derivative errors are zero, since
`fit_separable` does not measure them.

<!-- name: test_techniques_separable -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    # Two independent blocks, a linear one and a nonlinear one.
    return jnp.array([2 * x[0] - 1.0, jnp.sin(3 * x[1])])


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit_separable(target, domain, [sx.Affine(), sx.ChebyshevPolynomial(degree=5)])

point = jnp.array([0.4, 0.5])
assert jnp.allclose(surrogate(point), target(point), atol=1e-2)
# The report bounds the error from any coupling dropped between blocks.
assert float(surrogate.report.projection_max) < 1e-5
# The per-block search finds a larger error than the held-out sample.
assert surrogate.report.value_max <= surrogate.report.worst_case < 5e-2
```

## Equality constraints

A surrogate breaks a known equation, such as a mass balance, by its fit error.
`fit_constrained` makes it hold exactly instead. A `Constraint` names an
equation `h(x, y) = 0` and the flat outputs it determines, as many as it has
equations. The surrogate predicts the other outputs, and the dependent ones are
solved from the equation at call time, by Newton from the fit's own prediction
of them. This is the completion of [DC3](https://arxiv.org/abs/2104.12225) by
Donti, Rolnick and Kolter. An equation that is affine in the dependent outputs
completes in one step, and any other equation takes eight unless the
`Constraint` sets `steps`. The derivative comes from the implicit function
theorem, so the completed surrogate differentiates exactly. A choice of
dependent outputs the equation does not determine is rejected at fit time.

<!-- name: test_techniques_constraint -->
```python
import jax
import jax.numpy as jnp
import surreax as sx


def flows(x):
    first = x[0] * jnp.sin(x[1]) ** 2 / 2
    second = x[0] * jnp.cos(x[1]) * x[1] / 3
    return jnp.stack([first, second, jnp.exp(x[0]) - first - second])


def balance(x, y):
    return (jnp.sum(y) - jnp.exp(x[0]))[None]


domain = sx.box([0.5, 0.0], [2.0, 1.5])
constraint = sx.Constraint(balance, dependent=(2,))
surrogate = sx.fit_constrained(flows, domain, constraint, sx.ChebyshevPolynomial(degree=3))
points = domain.sample_unit(jax.random.uniform(jax.random.key(0), (16, 2)))
# The balance holds to solver precision, not to the fit error.
assert float(jnp.max(sx.constraint_residuals(surrogate, constraint, points))) < 1e-5
```

In implicit mode, `sx.Completed(method, constraint)` wraps any other method the
same way, so a subset of the residual's equations holds exactly.

## Automatic strategy selection

The structural analysis can also choose the form for you. If you leave the form
out of `fit`, it reads the function kind of each output and picks a form to
match. The reading is numerical. It samples the Jacobian and the curvature of
the target, so it works on any JAX function. An affine output has a constant
Jacobian, a piecewise linear output has a varying Jacobian and no curvature, a
polynomial output has real curvature, and a discontinuous output has a jump in
value that the Jacobian does not show. The form follows from the kinds of all
the outputs.

| Kinds present | Form |
| --- | --- |
| Only affine | `Affine` |
| Polynomial, with no piecewise linear | `ChebyshevPolynomial`, of the degree read from the target, at least 2 |
| Piecewise linear, with no polynomial | `MaxAffineForm` with four pieces |
| Polynomial and piecewise linear | `NetworkForm` of width 32 and depth 2 |
| Any discontinuous output | The same `NetworkForm`, with a warning |

The choice is advisory, so a form you pass always wins.

<!-- name: test_techniques_auto_form -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return jnp.array([x[0] ** 2 - 2 * x[0] * x[1]])


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
# No form given, so a polynomial form is chosen from the detected kind.
surrogate = sx.fit(target, domain, samples=256)
assert float(surrogate.report.value_max) < 1e-3
```

For a polynomial form the choice goes further, through `recommend_fit`. The
degree comes from the target's jaxpr, which `polynomial_degrees` walks while it
propagates a degree per value, so a cubic reads as degree three and anything
with `sin` or a division by an input reads as not a polynomial. When a curved
output is not a polynomial, the degree is the lowest at which raising it no
longer halves the held-out error, up to degree 6. The other settings come from
three readings of the samples.

- An active-subspace reduction is added when the target has at least three
  continuous inputs and varies along fewer directions than that. The directions
  are the fewest that reach 99% of the Jacobian spectrum.
- A PCA output reduction is added when there are at least four outputs and the
  fewest directions that reach 99% of their variance are at most half of them.
- A sparse budget is added when the basis has more terms than half of the
  training samples. It is the best of three budgets by held-out error.

`recommend_fit` returns these as a `FitSettings`, and an explicit
`reduce_inputs`, `reduce_outputs` or `sparse_terms` passed to `fit` wins over
the recommended one.

<!-- name: test_techniques_recommend_fit -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return jnp.array([x[0] ** 3 - x[0] * x[1]])


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
assert sx.polynomial_degrees(target, domain) == (3,)
settings = sx.recommend_fit(target, domain, samples=128)
assert settings.form == sx.ChebyshevPolynomial(degree=3)
```

`fit_separable` chooses a form per block the same way when the form is omitted,
so a target that is linear in one block and nonlinear in another gets a linear
form on the first and a richer form on the second. `fit_implicit` chooses a
method from the residual when you pass an `unknown_domain` and no method. It
takes an `InverseMap` for an unknown of at most 32 entries and an `ExactSolve`,
which reduces the output, for a larger one. The [implicit targets](implicit.md)
page describes both methods. You can inspect the reading directly with
`function_kinds`, `recommend_form`, `recommend_block_forms`, `recommend_method`,
and `recommend_implicit_form`.
