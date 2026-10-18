"""The max-affine form, a maximum of several affine functions.

The form is convex and piecewise linear, so a downstream solver can minimize it
as a linear program. The maximum makes it nonlinear in its parameters, so there
is no closed-form fit. Two routes fit it, and the literature reports no single
best one, so both ship and several restarts run.

Alternating minimization, from Magnani and Boyd, *Convex piecewise-linear
fitting*, Optimization and Engineering 10(1), 2009, assigns each sample to the
piece that attains the maximum and refits each piece by weighted least squares,
in the manner of k-means. The gradient route minimizes the mean squared error
through optimistix, driving an optax optimizer, which is reported in
*Max-Affine Regression via First-Order Methods*, SIAM Journal on Mathematics of
Data Science, 2024.

Each output coordinate gets its own convex max-affine model, so a vector target
is fitted coordinate by coordinate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
from jaxtyping import Array, Float, Int, PRNGKeyArray


class MaxAffine(eqx.Module):
    """A fitted max-affine model, one convex piece set per output coordinate."""

    slopes: Float[Array, "k m d"]
    intercepts: Float[Array, "k m"]

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " k"]:
        activations = jnp.einsum("kmd,d->km", self.slopes, x) + self.intercepts
        return jnp.max(activations, axis=1)


class MaxAffineForm(eqx.Module):
    """A request to fit a max-affine model of a given number of pieces."""

    pieces: int = eqx.field(static=True)
    restarts: int = eqx.field(static=True, default=5)
    iterations: int = eqx.field(static=True, default=20)
    method: str = eqx.field(static=True, default="alternating")
    learning_rate: float = eqx.field(static=True, default=0.05)
    steps: int = eqx.field(static=True, default=200)


def _augment(inputs: Float[Array, "n d"]) -> Float[Array, "n p"]:
    """Prepend a constant column, so a plane is one matrix row of parameters."""
    ones = jnp.ones((inputs.shape[0], 1), dtype=inputs.dtype)
    return jnp.concatenate([ones, inputs], axis=1)


def _predict(planes: Float[Array, "m p"], augmented: Float[Array, "n p"]) -> Float[Array, " n"]:
    """The max-affine value at each augmented input."""
    return jnp.max(augmented @ planes.T, axis=1)


def _mean_squared_error(
    planes: Float[Array, "m p"], augmented: Float[Array, "n p"], target: Float[Array, " n"]
) -> Float[Array, ""]:
    return jnp.mean((_predict(planes, augmented) - target) ** 2)


def _alternating(
    augmented: Float[Array, "n p"],
    target: Float[Array, " n"],
    pieces: int,
    iterations: int,
    key: PRNGKeyArray,
) -> Float[Array, "m p"]:
    """One run of alternating minimization from a random assignment.

    Every refit is pulled toward the previous planes, starting from the plane that
    fits all the samples, see `_least_squares_pieces`.
    """
    initial = jax.random.randint(key, (augmented.shape[0],), 0, pieces)
    everything = jnp.ones((augmented.shape[0],), dtype=jnp.int32)
    prior = jnp.tile(_least_squares_pieces(augmented, target, everything, 1, None), (pieces, 1))

    def step(_, state):
        assignment, previous = state
        planes = _least_squares_pieces(augmented, target, assignment, pieces, previous)
        return jnp.argmax(augmented @ planes.T, axis=1), planes

    assignment, previous = jax.lax.fori_loop(0, iterations, step, (initial, prior))
    return _least_squares_pieces(augmented, target, assignment, pieces, previous)


def _least_squares_pieces(
    augmented: Float[Array, "n p"],
    target: Float[Array, " n"],
    assignment: Int[Array, " n"],
    pieces: int,
    prior: Float[Array, "m p"] | None,
) -> Float[Array, "m p"]:
    """Refit every piece by least squares on the samples assigned to it.

    A small ridge pulls each piece toward its plane in `prior`, or toward zero
    without one. A piece that lost all its samples then keeps its plane, and one
    left with fewer samples than parameters keeps the prior in the directions
    those samples do not fix. Pulling toward zero instead leaves an empty piece at
    the zero plane, which clips the maximum at zero, and shrinks a small piece
    toward zero, so the fit gets worse as pieces outnumber what the target needs.
    """
    parameters = augmented.shape[1]
    if prior is None:
        prior = jnp.zeros((pieces, parameters), dtype=augmented.dtype)

    def fit_piece(index, plane):
        weights = (assignment == index).astype(augmented.dtype)
        weighted = augmented * weights[:, None]
        ridge = float(np.sqrt(np.finfo(augmented.dtype).eps))
        gram = augmented.T @ weighted + ridge * jnp.eye(parameters, dtype=augmented.dtype)
        return jnp.linalg.solve(gram, weighted.T @ target + ridge * plane)

    return jax.vmap(fit_piece)(jnp.arange(pieces), prior)


def _gradient(
    augmented: Float[Array, "n p"],
    target: Float[Array, " n"],
    start: Float[Array, "m p"],
    learning_rate: float,
    steps: int,
) -> Float[Array, "m p"]:
    """Refine planes by minimizing the mean squared error through optimistix."""
    solver = optx.OptaxMinimiser(optax.adam(learning_rate), rtol=1e-8, atol=1e-8)

    def loss(planes, _):
        return _mean_squared_error(planes, augmented, target)

    result = optx.minimise(loss, solver, start, max_steps=steps, throw=False)
    return result.value


def _fit_one_output(augmented, target, form, key, initial):
    """Fit the planes for one output coordinate, keeping the best of the restarts.

    When `initial` planes are given, they join the restarts as an extra
    candidate, so a warm start can only improve the result, never worsen it.
    """
    keys = jax.random.split(key, form.restarts)
    candidates = jax.vmap(
        lambda k: _alternating(augmented, target, form.pieces, form.iterations, k)
    )(keys)
    if form.method == "gradient":
        candidates = jax.vmap(
            lambda start: _gradient(augmented, target, start, form.learning_rate, form.steps)
        )(candidates)
    if initial is not None:
        candidates = jnp.concatenate([candidates, initial[None]])
    errors = jax.vmap(lambda planes: _mean_squared_error(planes, augmented, target))(candidates)
    return candidates[jnp.argmin(errors)]


def _as_planes(model: "MaxAffine | None") -> Float[Array, "k m p"] | None:
    """Pack a fitted model's slopes and intercepts back into plane rows."""
    if model is None:
        return None
    return jnp.concatenate([model.intercepts[:, :, None], model.slopes], axis=-1)


def fit_max_affine(
    form: MaxAffineForm,
    inputs: Float[Array, "n d"],
    targets: Float[Array, "n k"],
    *,
    key: PRNGKeyArray,
    initial: "MaxAffine | None" = None,
) -> MaxAffine:
    """Fit a convex max-affine model to each output coordinate.

    An `initial` model seeds an extra candidate per output, so a refit reuses a
    previous fit.
    """
    augmented = _augment(inputs)
    output_keys = jax.random.split(key, targets.shape[1])
    initial_planes = _as_planes(initial)
    if initial_planes is None:
        planes = jax.vmap(
            lambda column, output_key: _fit_one_output(augmented, column, form, output_key, None),
            in_axes=(1, 0),
        )(targets, output_keys)
    else:
        planes = jax.vmap(
            lambda column, output_key, start: _fit_one_output(
                augmented, column, form, output_key, start
            ),
            in_axes=(1, 0, 0),
        )(targets, output_keys, initial_planes)
    return MaxAffine(slopes=planes[:, :, 1:], intercepts=planes[:, :, 0])
