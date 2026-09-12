"""Pytest configuration shared across the test and docs suites.

The import hook wraps every array annotation in `surreax` with a runtime
beartype check, so a shape or dtype that violates its type fails during tests.
"""

from jaxtyping import install_import_hook

install_import_hook("surreax", "beartype.beartype")
