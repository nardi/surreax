# The basic workflow

This page walks through a full fit and the object it returns. A fit needs three
things. The target is the function to approximate, the domain is the region
where the approximation should be accurate, and the form is the shape the
approximation may take, such as a polynomial. The [terminology](terminology.md)
page defines these terms and the others that the documentation uses. The
recommended import is `import surreax as sx`.

## The target

A target is a pure JAX function from one input to a pytree of outputs. `surreax`
vectorizes it for you, so you write it for a single example. This one is the
horizontal range of a projectile as a function of its launch speed and angle.

<!-- name: test_start_target -->
```python
import jax.numpy as jnp
import surreax as sx


def projectile_range(state):
    speed, angle = state
    return (speed**2 * jnp.sin(2 * angle) / 9.81)[None]
```

## The domain

The domain says where the surrogate should be accurate. Without it the request
is not well posed, because reproducing a function has no meaning without a
region. A `Box` is the simplest domain, a lower and an upper bound per
coordinate.

<!-- name: test_start_target -->
```python
domain = sx.box([5.0, 0.2], [20.0, 1.3])
```

If you already have a dataset of inputs, `sx.box_from_data(points)` infers the
tightest box that contains them. A box around clustered or correlated data also
contains empty regions the data never reached, so `sx.domain_from_data(points)`
keeps a model of where the data lies as well. A surrogate fitted over it has
`surrogate.coverage(x)`, which is at most one near the data and grows away from
it. Use it to flag a prediction that is an extrapolation. The report's
`coverage_max` gives the largest coverage over the held-out points.

<!-- name: test_start_coverage -->
```python
import jax
import jax.numpy as jnp
import surreax as sx

along = jax.random.uniform(jax.random.key(0), (200,))
points = jnp.stack([along, along + 0.01 * jax.random.normal(jax.random.key(1), (200,))], axis=1)
outputs = jnp.sum(points**2, axis=1, keepdims=True)
surrogate = sx.fit_data(points, outputs, sx.ChebyshevPolynomial(degree=2))

# A point on the data's diagonal is covered, a corner of its bounding box is not.
assert float(surrogate.coverage(jnp.array([0.5, 0.5]))) <= 1.0
assert float(surrogate.coverage(jnp.array([0.95, 0.05]))) > 1.0
```

The example fits the data directly with `fit_data`, which the section on
[input-output pairs](#input-output-pairs) describes.

### Other input kinds

Not every input is a plain range. `sx.product` takes a pytree of leaf domains
that mirrors the target's input, and each leaf samples its own way.

- `sx.log_range(lower, upper)` samples uniformly in the logarithm, for an input
  that spans orders of magnitude. The model works on the logarithm too.
- `sx.normal(mean, std)` samples a normal distribution, cut at four standard
  deviations so the domain stays bounded.
- `sx.integer(lower, upper)` samples integers, both bounds included.
- `sx.categorical(choices)` picks one of a set of options, stacked along the
  first axis. The model sees one weight per option, which is one-hot at every
  sample.

Integer and categorical inputs are discrete. Derivative supervision and the
derivative error leave them out, an input reduction passes them through
unchanged, and the worst-case search only reports points with valid values.

<!-- name: test_start_mixed_domain -->
```python
import jax.numpy as jnp
import surreax as sx


def load(inputs):
    speed, density, panels = inputs["speed"][0], inputs["density"][0], inputs["panels"][0]
    return (inputs["shape"] * panels * speed**2 + jnp.log10(density))[None]


domain = sx.product(
    {
        "speed": sx.box([1.0], [10.0]),
        "density": sx.log_range([0.1], [10.0]),
        "panels": sx.integer([1], [4]),
        "shape": sx.categorical([0.5, 1.2]),
    }
)
surrogate = sx.fit(load, domain, sx.ChebyshevPolynomial(degree=4), samples=512)
point = {
    "speed": jnp.array([5.0]),
    "density": jnp.array([1.0]),
    "panels": jnp.array([2.0]),
    "shape": jnp.array(1.2),
}
assert jnp.allclose(surrogate(point), load(point), rtol=1e-3)
```

## The fit

`fit` samples the domain with a low-discrepancy sequence, evaluates the target,
and solves for the chosen form. A degree-five polynomial handles this target
well.

<!-- name: test_start_target -->
```python
surrogate = sx.fit(
    projectile_range,
    domain,
    sx.ChebyshevPolynomial(degree=5),
    samples=512,
)
```

The sequence is `"sobol"` by default. Pass `sampler="halton"` or
`sampler="latin_hypercube"` to change it. The `holdout` argument, 0.2 by
default, sets aside the last fraction of the samples for the report, so the fit
itself uses the rest. The `seed` argument makes the samples and any random
restarts reproducible.

The [model forms](forms.md) page describes the choices for the third argument.
If you leave it out, `fit` reads the target and picks a form, as described in
[automatic strategy selection](techniques.md#automatic-strategy-selection). The
[fitting techniques](techniques.md) page describes the other options of `fit`.

## The report

The fitted object carries an error report, measured on held-out samples that did
not take part in the fit. Value error and derivative error are reported apart.

<!-- name: test_start_target -->
```python
report = surrogate.report
assert float(report.value_mean) < 0.5
assert jnp.isfinite(report.derivative_mean)
```

Every error is the Euclidean norm of the difference between the surrogate and
the target at one held-out point, in the units of the target. The fields of the
report are these.

| Field | Meaning |
| --- | --- |
| `value_mean`, `value_max` | Mean and largest difference between the output of the surrogate and the output of the target. |
| `derivative_mean`, `derivative_max` | The same for the Jacobian, which a downstream optimizer consumes. |
| `projection_mean`, `projection_max` | Error that an output reduction gives up before any fitting, from reconstructing the held-out outputs. |
| `worst_case`, `worst_case_point` | Largest value error that the [worst-case search](techniques.md#worst-case-error) found over the whole domain, and the flat input where it occurs. |
| `structure_cost` | Number of parts in the model, such as basis terms or affine pieces. |
| `coverage_max` | Largest [coverage](#the-domain) over the held-out points. |

A surrogate that matches values well can still match derivatives badly, which is
why the two errors are kept apart. The worst case is zero when you pass
`worst_case_starts=0`.

## The surrogate as a function

The surrogate has the target's signature and evaluates in one step.

<!-- name: test_start_target -->
```python
true = projectile_range(jnp.array([12.0, 0.7]))
approx = surrogate(jnp.array([12.0, 0.7]))
assert abs(float(approx[0]) - float(true[0])) < 1.0
```

It is also a pytree of arrays, so it can be differentiated, vectorized with
`jax.vmap`, and compiled with `jax.jit`. A form that holds a Python function,
such as the activation of `NetworkForm`, needs `equinox.filter_jit` in place of
`jax.jit`.

## Input-output pairs

When you already hold input-output pairs and have no callable, `fit_data` fits a
linear form to them directly. Linear forms are `Affine`, `ChebyshevPolynomial`,
and `Grid`. It normalizes from the sample moments, since the pairs carry no
domain box, and reports the held-out prediction error. Its domain comes from the
data through `domain_from_data`, so the surrogate's `coverage` tells a query
near the data from one in an empty part of the box.

<!-- name: test_start_fit_data -->
```python
import jax
import jax.numpy as jnp
import surreax as sx

key = jax.random.key(0)
inputs = jax.random.uniform(key, (200, 3), minval=-1.0, maxval=1.0)
weights = jnp.array([[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]])
outputs = jax.vmap(lambda x: weights @ x + jnp.array([0.5, -1.0]))(inputs)

surrogate = sx.fit_data(inputs, outputs, sx.Affine())
assert float(surrogate.report.value_max) < 1e-3
```

The report from `fit_data` has no derivative, projection, or worst-case error,
because there is no target to compare against. The [implicit
targets](implicit.md) page uses `fit_data` internally, and it is useful on its
own for a distilled dataset.
