"""Package-level smoke test.

Confirms the package imports cleanly under the beartype hook before the real
modules land.
"""

import surreax


def test_package_imports() -> None:
    assert surreax.__doc__ is not None
