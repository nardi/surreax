# Fitting techniques

Beyond the choice of form, `fit` takes a few options that change how the
surrogate is fitted. Each one is off by default.

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
turns it on.

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

On the input side, `pca_inputs` finds the directions in which the data varies,
while `active_subspace_inputs` finds the directions in which the function
varies, from the average outer product of the target gradient. The active
subspace is from Constantine and Gleich, [Computing active subspaces with Monte
Carlo](https://arxiv.org/abs/1408.0545). The report carries the projection
error, the accuracy the reduction gives up before any fitting.

## Sparse basis selection

A total-degree polynomial basis grows as the number of inputs and the degree
rise, and most of its coefficients should be zero. A sparse fit keeps only the
terms that matter, which tames that growth and lowers the structure cost. This
is the route ALAMO takes, from Cozad, Sahinidis, and Miller, [Learning
surrogate models for simulation-based
optimization](https://doi.org/10.1002/aic.14418), AIChE Journal 60, 2014.

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

The value and derivative errors in the report are measured on a held-out
sample, so they can miss a region where the fit is worst. Because the surrogate
and the target are both differentiable, `fit` also climbs the error over the
whole domain from several starts and reports the largest it finds, along with
the point where it occurs. This is a tighter check than a sample statistic, and
setting `worst_case_starts` to zero skips it when speed matters.

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
can add points where it is worst instead of committing to one grid. `fit_adaptive`
fits on an initial sample, locates the highest-error points with the domain
search, adds them, and refits, stopping when the worst-case error meets a
tolerance or after a round budget. This is the residual-based adaptive
refinement of the surrogate-modeling literature, such as DeepXDE, from Lu and
colleagues, [DeepXDE](https://arxiv.org/abs/1907.04502), SIAM Review 63(1),
2021.

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
depends on can be dropped, which is exact rather than statistical. `prune_inputs`
is an input reducer that does this, and it fits into `reduce_inputs` like the PCA
and active-subspace reducers.

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

`analyze` returns the same reading as a report: the coupling matrix and the
block labels that split outputs and inputs into independent groups, so a
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
