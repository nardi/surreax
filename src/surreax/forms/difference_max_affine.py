"""The difference of two max-affine functions.

Subtracting one convex piecewise linear function from another gives a general
piecewise linear function, so this form is not limited to convex targets, and
it carries a convex-plus-concave decomposition that later algorithms can use.
There is a dedicated literature on fitting in this form.

The difference is neither convex nor linear in its parameters, so it fits by
minimizing the mean squared error through the optimistix backend, from several
restarts. Each restart starts from a plain max-affine fit of the positive part
with a small negative part, so the search begins near a convex fit and bends
from there. Every output coordinate is fitted on its own.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray

from surreax.forms.max_affine import _alternating, _augment


class DifferenceOfMaxAffine(eqx.Module):
    """A fitted difference `g(x) - h(x)` of two max-affine models per output."""

    positive_slopes: Float[Array, "k m d"]
    positive_intercepts: Float[Array, "k m"]
    negative_slopes: Float[Array, "k n d"]
    negative_intercepts: Float[Array, "k n"]

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        positive = jnp.max(
            jnp.einsum("kmd,d->km", self.positive_slopes, x) + self.positive_intercepts, axis=1
        )
        negative = jnp.max(
            jnp.einsum("knd,d->kn", self.negative_slopes, x) + self.negative_intercepts, axis=1
        )
        return positive - negative


class DifferenceOfMaxAffineForm(eqx.Module):
    """A request to fit a difference of two max-affine models, `pieces` each."""

    pieces: int = eqx.field(static=True)
    restarts: int = eqx.field(static=True, default=5)
    iterations: int = eqx.field(static=True, default=20)
    steps: int = eqx.field(static=True, default=300)
    learning_rate: float = eqx.field(static=True, default=0.05)


def _predict(
    positive: Float[Array, "m p"], negative: Float[Array, "m p"], augmented: Float[Array, "n p"]
) -> Float[Array, " n"]:
    return jnp.max(augmented @ positive.T, axis=1) - jnp.max(augmented @ negative.T, axis=1)


def _error(parts, augmented, target):
    positive, negative = parts
    return jnp.mean((_predict(positive, negative, augmented) - target) ** 2)


def _refine(augmented, target, start, learning_rate, steps):
    """Minimize the error over both plane sets from a starting pair."""
    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=1e-8, atol=1e-8)

    def loss(parts, _):
        return _error(parts, augmented, target)

    result = optx.minimise(loss, solver, start, max_steps=steps, throw=False)
    return result.value


def _fit_one_output(augmented, target, form, key):
    """Fit the two plane sets for one output, keeping the best of the restarts."""

    def restart(restart_key):
        positive_key, negative_key = jax.random.split(restart_key)
        positive = _alternating(augmented, target, form.pieces, form.iterations, positive_key)
        negative = 0.01 * jax.random.normal(negative_key, (form.pieces, augmented.shape[1]))
        return _refine(augmented, target, (positive, negative), form.learning_rate, form.steps)

    candidates = jax.vmap(restart)(jax.random.split(key, form.restarts))
    errors = jax.vmap(lambda parts: _error(parts, augmented, target))(candidates)
    best = jnp.argmin(errors)
    return candidates[0][best], candidates[1][best]


def fit_difference_max_affine(
    form: DifferenceOfMaxAffineForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
) -> DifferenceOfMaxAffine:
    """Fit a difference of two max-affine models to each output coordinate."""
    augmented = _augment(inputs)
    output_keys = jax.random.split(key, targets.shape[1])
    positive, negative = jax.vmap(
        lambda column, output_key: _fit_one_output(augmented, column, form, output_key),
        in_axes=(1, 0),
    )(targets, output_keys)
    return DifferenceOfMaxAffine(
        positive_slopes=positive[:, :, 1:],
        positive_intercepts=positive[:, :, 0],
        negative_slopes=negative[:, :, 1:],
        negative_intercepts=negative[:, :, 0],
    )
