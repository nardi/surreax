"""Marks each test by its directory, so `-m unit` and `tests/unit` select the same tests."""

from pathlib import Path

import pytest

_CATEGORIES = ("unit", "integration", "benchmarks")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the `unit`, `integration` or `benchmark` marker for the directory of each test."""
    for item in items:
        for category in _CATEGORIES:
            if category in Path(item.path).parts:
                # The marker is singular, the directory plural.
                item.add_marker(getattr(pytest.mark, category.removesuffix("s")))
