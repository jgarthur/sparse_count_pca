"""Reject display-math delimiters that Markdown failed to process."""

import re
from typing import Any

UNPROCESSED_DISPLAY_MATH = re.compile(r"<p>\s*\$\$.*?\$\$\s*</p>", re.DOTALL)


def on_page_content(html: str, page: Any, **kwargs: Any) -> str:
    """Fail the docs build when a display equation remains a plain paragraph."""
    if UNPROCESSED_DISPLAY_MATH.search(html):
        raise RuntimeError(
            f"Unprocessed display-math delimiters in {page.file.src_uri}; "
            "check indentation around the $$ block."
        )
    return html
