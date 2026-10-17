"""Searching the domain for large error.

The held-out error is a sample statistic, so it can miss a region where the fit
is worst. Both the surrogate and the target are differentiable, so the error can
be climbed with gradients instead, from several starts over the box. The largest
result is the worst-case error for the report, and the climbed points are where
adaptive sampling should add samples.

The box constraint enters by a reparameterization: the optimizer works in an
unbounded coordinate that maps into the box through a sigmoid, so the continuous
coordinates need no projection step. Integer and categorical coordinates stay
fixed at a valid value during each climb, and the result is projected onto the
domain, so the reported point is one the target could be called with. The search
reuses the optimistix minimization backend, run on the negative error.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from surreax.domain import Domain
from surreax.flatten import ravel


@eqx.filter_jit
def _search(
    surrogate: Callable[[PyTree[Array]], PyTree[Array]],
    target: Callable[[PyTree[Array]], PyTree[Array]],
    domain: Domain,
    *,
    starts: int,
    steps: int,
    learning_rate: float,
    key: PRNGKeyArray,
    seed_points: Float[Array, "p d"],
) -> tuple[Float[Array, " starts"], Float[Array, "starts d"]]:
    """Climb the error from every start and return each start's error and point.

    The search warm-starts from the highest-error points among `seed_points`, so
    it begins near a known-bad region, and fills any remaining starts at random.
    """
    lower, upper = domain.bounds()
    width = upper - lower

    def squared_error(point):
        predicted = ravel(surrogate(domain.unravel(point)))
        actual = ravel(target(domain.unravel(point)))
        # Optimize the squared error, which is smooth at zero, where the norm's
        # gradient is undefined and would poison the search.
        return jnp.sum((predicted - actual) ** 2)

    def point_of(unbounded):
        return lower + width * jax.nn.sigmoid(unbounded)

    def unbounded_of(point):
        # Invert the sigmoid, clipped off the ends so a boundary point is finite.
        fraction = jnp.clip((point - lower) / width, 1e-4, 1 - 1e-4)
        return jnp.log(fraction / (1 - fraction))

    def negative_error(unbounded, _):
        return -squared_error(point_of(unbounded))

    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=1e-6, atol=1e-6)

    # Discrete coordinates stay at the projected start of each climb, since the
    # target has no useful gradient in them and a step between categories leaves
    # the domain. Only the continuous coordinates move.
    discrete = jnp.asarray(domain.discrete_mask)

    def climb(start):
        fixed = domain.project(point_of(start))

        def negative_error_held(unbounded, args):
            held = unbounded_of(jnp.where(discrete, fixed, point_of(unbounded)))
            return negative_error(held, args)

        found = optx.minimise(negative_error_held, solver, start, max_steps=steps, throw=False)
        point = domain.project(jnp.where(discrete, fixed, point_of(found.value)))
        return jnp.sqrt(squared_error(point)), point

    seed_errors = jax.vmap(squared_error)(seed_points)
    warmest = jnp.argsort(-seed_errors)[: min(starts, seed_points.shape[0])]
    warm_starts = jax.vmap(unbounded_of)(seed_points[warmest])
    random_count = max(0, starts - warmest.shape[0])
    random_starts = jax.random.normal(key, (random_count, lower.shape[0]))
    starts_unbounded = jnp.concatenate([warm_starts, random_starts])

    return jax.vmap(climb)(starts_unbounded)


def worst_case(
    surrogate: Callable[[PyTree[Array]], PyTree[Array]],
    target: Callable[[PyTree[Array]], PyTree[Array]],
    domain: Domain,
    *,
    starts: int,
    steps: int,
    learning_rate: float,
    key: PRNGKeyArray,
    seed_points: Float[Array, "p d"],
) -> tuple[Float[Array, ""], Float[Array, " d"]]:
    """Find the largest surrogate error over the domain and where it occurs."""
    errors, points = _search(
        surrogate,
        target,
        domain,
        starts=starts,
        steps=steps,
        learning_rate=learning_rate,
        key=key,
        seed_points=seed_points,
    )
    best = jnp.argmax(errors)
    return errors[best], points[best]


def high_error_points(
    surrogate: Callable[[PyTree[Array]], PyTree[Array]],
    target: Callable[[PyTree[Array]], PyTree[Array]],
    domain: Domain,
    *,
    count: int,
    steps: int,
    learning_rate: float,
    key: PRNGKeyArray,
    seed_points: Float[Array, "p d"],
) -> Float[Array, "count d"]:
    """Points of large surrogate error, one climbed from each of `count` starts.

    These are where a refit should add samples, since the fit is worst there.
    """
    _, points = _search(
        surrogate,
        target,
        domain,
        starts=count,
        steps=steps,
        learning_rate=learning_rate,
        key=key,
        seed_points=seed_points,
    )
    return points
