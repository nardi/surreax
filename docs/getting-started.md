# The basic workflow

This page walks through a full fit and the object it returns. The recommended
import is `import surreax as sx`.

## The target

A target is a pure JAX function from one input to a pytree of outputs.
`surreax` vectorizes it for you, so you write it for a single example. This one
is the horizontal range of a projectile as a function of its launch speed and
angle.

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

### Other input kinds

Not every input is a plain range. `sx.product` takes a pytree of leaf domains
that mirrors the target's input, and each leaf samples its own way.

- `sx.log_range(lower, upper)` samples uniformly in the logarithm, for an input
  that spans orders of magnitude. The model works on the logarithm too.
- `sx.normal(mean, std)` samples a normal distribution, cut at four standard
  deviations so the domain stays bounded.
- `sx.integer(lower, upper)` samples integers, both bounds included.
- `sx.categorical(choices)` picks one of a set of options, stacked along the first
  axis.

Integer and categorical inputs are discrete. Derivative supervision and the
derivative error leave them out, an input reduction passes them through
unchanged, and the worst-case search only reports points with valid values.
`fit` also takes a `sampler`, `"sobol"` by default, or `"halton"` or
`"latin_hypercube"`.

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

## The report

The fitted object carries an error report, measured on held-out samples that
did not take part in the fit. Value error and derivative error are reported
apart.

<!-- name: test_start_target -->
```python
report = surrogate.report
assert float(report.value_mean) < 0.5
assert jnp.isfinite(report.derivative_mean)
```

## The surrogate as a function

The surrogate has the target's signature and evaluates in one step.

<!-- name: test_start_target -->
```python
true = projectile_range(jnp.array([12.0, 0.7]))
approx = surrogate(jnp.array([12.0, 0.7]))
assert abs(float(approx[0]) - float(true[0])) < 1.0
```
