"""Reject display-math blocks that Markdown failed to process."""

import re
from typing import Any

UNPROCESSED_DISPLAY_MATH = re.compile(r"<p>\s*\$\$.*?\$\$\s*</p>", re.DOTALL)
FENCED_MATH = re.compile(r"^[ \t]*```math[ \t]*$", re.MULTILINE)
DOLLAR_DELIMITER = re.compile(r"^[ \t]*\$\$[ \t]*$", re.MULTILINE)
RENDERED_DISPLAY_MATH = re.compile(r'<div class="arithmatex">\s*\\\[')
EXPECTED_DISPLAY_MATH: dict[str, int] = {}


def on_page_markdown(markdown: str, page: Any, **kwargs: Any) -> str:
    """Record how many display equations the rendered page must contain."""
    dollar_delimiters = len(DOLLAR_DELIMITER.findall(markdown))
    if dollar_delimiters % 2:
        raise RuntimeError(
            f"Unmatched display-math delimiter in {page.file.src_uri}."
        )
    EXPECTED_DISPLAY_MATH[page.file.src_uri] = (
        len(FENCED_MATH.findall(markdown)) + dollar_delimiters // 2
    )
    return markdown


def on_page_content(html: str, page: Any, **kwargs: Any) -> str:
    """Fail when source display equations do not render as MathJax containers."""
    if UNPROCESSED_DISPLAY_MATH.search(html):
        raise RuntimeError(
            f"Unprocessed display-math delimiters in {page.file.src_uri}; "
            "check indentation around the $$ block."
        )
    expected = EXPECTED_DISPLAY_MATH.get(page.file.src_uri, 0)
    rendered = len(RENDERED_DISPLAY_MATH.findall(html))
    if rendered < expected:
        raise RuntimeError(
            f"Expected {expected} rendered display equations in "
            f"{page.file.src_uri}, found {rendered}; check math-fence "
            "configuration and delimiter indentation."
        )
    return html
