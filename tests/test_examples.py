"""Smoke tests for the executable example scripts."""

import runpy
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
EXAMPLES = [
    "residual_pca_anndata.py",
    "shifted_clr_anndata.py",
    "transform_reuse.py",
    "correspondence_analysis.py",
]


@pytest.mark.parametrize("filename", EXAMPLES)
def test_example_script_runs(filename: str) -> None:
    """Each canonical example executes successfully from start to finish."""
    runpy.run_path(EXAMPLE_DIR / filename, run_name="__main__")
