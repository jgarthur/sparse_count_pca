"""Smoke tests for the executable example scripts."""

import runpy
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
EXAMPLES = sorted(EXAMPLE_DIR.glob("*.py"))


@pytest.mark.parametrize("example_path", EXAMPLES, ids=lambda path: path.stem)
def test_example_script_runs(example_path: Path) -> None:
    """Each canonical example executes successfully from start to finish."""
    runpy.run_path(example_path, run_name="__main__")
