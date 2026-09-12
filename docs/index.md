# surreax

`surreax` fits a surrogate to a function written in JAX. You give it the
function and a region where the surrogate should be accurate, and it returns a
second function that behaves approximately like the first one but has a form
you chose in advance, such as affine, polynomial, or piecewise linear.

The point of fixing the form up front is that some forms are easy to reason
about later. A piecewise linear surrogate can be handed to a mixed-integer
solver. A convex surrogate can be minimized reliably. A general neural network
can do neither. `surreax` treats the form as the thing you care about, and
accuracy as the thing to maximize once the form is fixed.

## Install

```bash
pip install surreax
```

## A first fit

Here is a smooth target on a box, fitted with a low-degree polynomial. The
surrogate reproduces the target to a small tolerance.

<!-- name: test_index_first_fit -->
```python
import jax.numpy as jnp
import surreax as sx


def target(x):
    return (x[0] ** 2 + x[1] ** 2)[None]


domain = sx.box([-1.0, -1.0], [1.0, 1.0])
surrogate = sx.fit(target, domain, sx.ChebyshevPolynomial(degree=2), samples=128)

prediction = surrogate(jnp.array([0.5, 0.5]))
assert abs(float(prediction[0]) - 0.5) < 1e-3
```

The returned `surrogate` is an ordinary JAX function. It differentiates,
`vmap`s, and `jit`s like anything else, because it is registered as a pytree.

<!-- name: test_index_first_fit -->
```python
import jax

batch = jnp.array([[0.1, 0.2], [0.5, 0.5], [-0.3, 0.8]])
values = jax.vmap(surrogate)(batch)
assert values.shape == (3, 1)

jacobian = jax.jacobian(surrogate)(jnp.array([0.5, 0.5]))
assert jacobian.shape == (1, 2)
```

## Why another surrogate library

There are good surrogate libraries already. `surreax` aims at a narrower target
than most of them, which leads to a few different choices:

1. It is built on JAX, so the target's exact Jacobian is available and cheap.
   The error report uses it, and later versions train against it.
2. It cares about the *form* of the result, not only its accuracy. The fitted
   object keeps the structure you asked for, so a downstream solver can use it.
3. It fits in closed form wherever the form allows. The affine and polynomial
   forms are one linear solve, with no learning rate and a deterministic
   answer.
4. It reports error by source, splitting the value error from the derivative
   error, because a surrogate can match values well and derivatives badly.

For where this sits next to SMT, ALAMO, OMLT, and the physics-informed tools,
see the [development notes](development.md).
```
