"""Checks that the code links in the documentation still point at their definitions."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

LINK = re.compile(
    r"\[`(?P<symbol>\w+)`\]\(https://github\.com/nardi/surreax/blob/main/"
    r"(?P<path>tests/[\w./]+)#L(?P<line>\d+)\)"
)
"""A link to a line of a file under `tests`, with the name of its target as the text."""


def _links() -> list[tuple[str, str, int]]:
    pages = sorted((ROOT / "docs").glob("*.md"))
    return [
        (m["symbol"], m["path"], int(m["line"]))
        for page in pages
        for m in LINK.finditer(page.read_text())
    ]


def test_the_page_links_to_code() -> None:
    assert len(_links()) > 50


@pytest.mark.parametrize(("symbol", "path", "line"), _links())
def test_a_link_points_at_the_definition_of_its_name(symbol: str, path: str, line: int) -> None:
    source = (ROOT / path).read_text().split("\n")
    # A definition is a function, a class, or a module-level assignment.
    pattern = re.compile(rf"(def|class) {symbol}\b|{symbol}\b\s*[:=]")
    assert pattern.match(source[line - 1]), f"{path} line {line} does not define {symbol}"
