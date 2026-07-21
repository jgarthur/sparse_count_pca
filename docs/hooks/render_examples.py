"""Execute Jupytext percent notebooks and render them as MkDocs pages."""

from pathlib import Path
from typing import Any

import jupytext
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "docs" / "examples"


def _write_if_changed(path: Path, content: str | bytes) -> None:
    """Write generated content only when its bytes have changed."""
    encoded = content.encode() if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _render_example(source: Path) -> tuple[str, dict[str, bytes]]:
    """Execute one percent notebook and export its cells and outputs to Markdown."""
    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    executor = ExecutePreprocessor(timeout=60, kernel_name="python3")
    executor.preprocess(
        notebook,
        resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
    )

    resources: dict[str, Any] = {
        "metadata": {"path": str(source.parent)},
        "output_files_dir": f"{source.stem}_files",
        "unique_key": source.stem,
    }
    markdown, resources = MarkdownExporter().from_notebook_node(
        notebook,
        resources=resources,
    )
    return markdown, resources.get("outputs", {})


def on_config(config: Any) -> Any:
    """Generate Markdown pages before MkDocs discovers documentation files."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for source in sorted(EXAMPLE_DIRECTORY.glob("*.py")):
        markdown, outputs = _render_example(source)
        _write_if_changed(OUTPUT_DIRECTORY / f"{source.stem}.md", markdown)
        for relative_path, content in outputs.items():
            _write_if_changed(OUTPUT_DIRECTORY / relative_path, content)
    return config
