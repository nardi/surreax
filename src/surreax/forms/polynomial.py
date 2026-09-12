"""The polynomial form in a Chebyshev basis.

The features are products of Chebyshev polynomials, one factor per input
coordinate, over every exponent tuple whose total degree is at most `degree`.
The basis is linear in its parameters, so it is fitted by one linear solve, and
it converges quickly for smooth targets. Inputs are expected in `[-1, 1]`,
where the input normalizer places them.
"""

import itertools

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _exponents(dimension: int, degree: int) -> list[tuple[int, ...]]:
    """Every exponent tuple over `dimension` axes with total degree <= degree."""
    grid = itertools.product(range(degree + 1), repeat=dimension)
    return [e for e in grid if sum(e) <= degree]


def _chebyshev_table(x: Float[Array, " d"], degree: int) -> Float[Array, "d degree_plus_one"]:
    """Chebyshev values `T_0 .. T_degree` for each coordinate of `x`.

    Uses the recurrence `T_{n+1} = 2 x T_n - T_{n-1}`.
    """
    columns = [jnp.ones_like(x)]
    if degree >= 1:
        columns.append(x)
    for _ in range(2, degree + 1):
        columns.append(2 * x * columns[-1] - columns[-2])
    return jnp.stack(columns, axis=1)


class ChebyshevPolynomial(eqx.Module):
    """A total-degree Chebyshev polynomial form."""

    degree: int = eqx.field(static=True)

    def features(self, x: Float[Array, " d"]) -> Float[Array, " f"]:
        table = _chebyshev_table(x, self.degree)
        exponents = _exponents(x.shape[0], self.degree)
        products = [
            jnp.prod(jnp.stack([table[axis, power] for axis, power in enumerate(e)]))
            for e in exponents
        ]
        return jnp.stack(products)
