"""Tests for the repository's citation metadata."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_citation_version_matches_package_version() -> None:
    """CITATION.cff names the same version as pyproject.toml, so release archives
    and GitHub's citation export cite the version being released."""
    with (ROOT / "pyproject.toml").open("rb") as file:
        package_version = tomllib.load(file)["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)\s*$", citation, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == package_version
