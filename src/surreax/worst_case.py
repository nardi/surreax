"""Largest error over the whole domain.

The held-out error is a sample statistic, so it can miss a region where the fit
is worst. Both the surrogate and the target are differentiable, so the error can
be climbed with gradients instead. This maximizes the pointwise error over the
box from several random starts and keeps the largest value found.

The box constraint enters by a reparameterization: the optimizer works in an
unbounded coordinate that maps into the box through a sigmoid, so there is no
projection step. The search reuses the optimistix minimization backend, run on
the negative error.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from surreax.domain import Domain
from surreax.flatten import ravel


def worst_case(
    surrogate: Callable[[PyTree[Array]], PyTree[Array]],
    target: Callable[[PyTree[Array]], PyTree[Array]],
    domain: Domain,
    *,
    starts: int,
    steps: int,
    learning_rate: float,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], Float[Array, " d"]]:
    """Find the largest surrogate error over the domain and where it occurs.

    Returns the error and the flat input that attains it, searched from `starts`
    random points.
    """
    lower, upper = domain.bounds()

    def squared_error(point):
        predicted = ravel(surrogate(domain.unravel(point)))
        actual = ravel(target(domain.unravel(point)))
        # Optimize the squared error, which is smooth at zero, where the norm's
        # gradient is undefined and would poison the search.
        return jnp.sum((predicted - actual) ** 2)

    def point_of(unbounded):
        return lower + (upper - lower) * jax.nn.sigmoid(unbounded)

    def negative_error(unbounded, _):
        return -squared_error(point_of(unbounded))

    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=1e-6, atol=1e-6)

    def climb(start):
        found = optx.minimise(negative_error, solver, start, max_steps=steps, throw=False)
        point = point_of(found.value)
        return jnp.sqrt(squared_error(point)), point

    starts_unbounded = jax.random.normal(key, (starts, lower.shape[0]))
    errors, points = jax.vmap(climb)(starts_unbounded)
    best = jnp.argmax(errors)
    return errors[best], points[best]
