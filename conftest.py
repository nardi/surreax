"""Pytest configuration shared across the test and docs suites.

The import hook wraps every array annotation in `surreax` with a runtime
beartype check, so a shape or dtype that violates its type fails during tests.
"""

import jax
import pytest
from jaxtyping import install_import_hook

install_import_hook("surreax", "beartype.beartype")


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Free compiled executables after each test.

    The whole suite fits many surrogates of many shapes, and JAX caches a
    compiled program for each, so without this the accumulated executables
    exhaust memory over a full run.
    """
    yield
    jax.clear_caches()
