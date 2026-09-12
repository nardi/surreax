"""Round-trip tests for the pytree flattening helpers."""

import jax.numpy as jnp
import jax.tree_util as jtu
from hypothesis import given
from hypothesis import strategies as st

from surreax.flatten import ravel, ravel_example


def test_ravel_array_is_flat_copy() -> None:
    values = jnp.array([1.0, 2.0, 3.0])
    assert jnp.array_equal(ravel(values), values)


def test_ravel_example_round_trips_a_nested_pytree() -> None:
    tree = {"a": jnp.array([1.0, 2.0]), "b": (jnp.array([[3.0, 4.0]]),)}
    flat, unravel = ravel_example(tree)
    assert flat.shape == (4,)
    restored = unravel(flat)
    assert jnp.array_equal(restored["a"], tree["a"])
    assert jnp.array_equal(restored["b"][0], tree["b"][0])


@given(st.lists(st.integers(1, 5), min_size=1, max_size=4))
def test_ravel_preserves_total_size(shapes: list[int]) -> None:
    leaves = [jnp.arange(float(n)) for n in shapes]
    tree = {str(i): leaf for i, leaf in enumerate(leaves)}
    flat, unravel = ravel_example(tree)
    assert flat.shape == (sum(shapes),)
    restored = unravel(flat)
    for original, back in zip(jtu.tree_leaves(tree), jtu.tree_leaves(restored), strict=True):
        assert jnp.array_equal(original, back)
