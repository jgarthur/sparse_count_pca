"""Smoke tests for the executable example scripts."""

import runpy
from pathlib import Path

import matplotlib
import pytest

EXAMPLE_DIR = Path(__file__).parents[1] / "examples"
EXAMPLES = sorted(EXAMPLE_DIR.glob("*.py"))


@pytest.mark.parametrize("example_path", EXAMPLES, ids=lambda path: path.stem)
def test_example_script_runs(example_path: Path, monkeypatch) -> None:
    """Each canonical example runs to completion without opening plot windows."""
    matplotlib.use("Agg", force=True)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    runpy.run_path(example_path, run_name="__main__")
