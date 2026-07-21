"""Tests for documentation build validation helpers."""

from types import SimpleNamespace

import pytest

from docs.hooks.validate_math import on_page_content


def test_unprocessed_display_math_fails_validation() -> None:
    """Plain-paragraph display delimiters fail with the source page name."""
    page = SimpleNamespace(file=SimpleNamespace(src_uri="guide.md"))

    with pytest.raises(RuntimeError, match=r"guide\.md.*indentation"):
        on_page_content("<p>$$\nx + y\n$$</p>", page)


def test_processed_display_math_passes_validation() -> None:
    """Arithmatex-wrapped display equations pass through unchanged."""
    html = '<div class="arithmatex">\\[x + y\\]</div>'
    page = SimpleNamespace(file=SimpleNamespace(src_uri="guide.md"))

    assert on_page_content(html, page) == html
