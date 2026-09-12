# Getting started

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
tightest box that contains them.

## Fitting

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

## Calling the surrogate

The surrogate has the target's signature and evaluates in one step.

<!-- name: test_start_target -->
```python
true = projectile_range(jnp.array([12.0, 0.7]))
approx = surrogate(jnp.array([12.0, 0.7]))
assert abs(float(approx[0]) - float(true[0])) < 1.0
```
