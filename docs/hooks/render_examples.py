"""Execute the guide notebooks and render them as includable Markdown fragments.

Each ``examples/*.py`` percent notebook is executed and exported to
``docs/examples/<stem>.md``. Those fragments are excluded from the site's own
pages and are pulled into the matching guide with a ``pymdownx.snippets``
include, so every guide's complete example is code that ran during the build.

Rendered fragments are cached by their inputs so an unrelated Markdown edit
does not execute every notebook during ``mkdocs serve``. Set
``DOCS_FORCE_EXAMPLES=1`` to bypass the cache.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE_DIRECTORY = REPOSITORY_ROOT / "examples"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "docs" / "examples"
CACHE_PATH = OUTPUT_DIRECTORY / ".render-cache.json"
CACHE_VERSION = 1
LOGGER = logging.getLogger("mkdocs.plugins.render_examples")


def _digest_files(paths: list[Path]) -> str:
    """Return a stable digest of file names and contents."""
    digest = hashlib.sha256()
    digest.update(f"render-cache-v{CACHE_VERSION}\0".encode())
    for path in sorted(paths):
        relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _shared_dependency_digest() -> str:
    """Hash inputs that can affect every executed example."""
    paths = [
        Path(__file__),
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "uv.lock",
        *(
            path
            for path in (REPOSITORY_ROOT / "src").rglob("*.py")
            if path.is_file()
        ),
    ]
    return _digest_files(paths)


def _example_digest(source: Path, shared_digest: str) -> str:
    """Hash one example together with its shared dependencies."""
    digest = hashlib.sha256()
    digest.update(shared_digest.encode())
    digest.update(b"\0")
    digest.update(source.relative_to(REPOSITORY_ROOT).as_posix().encode())
    digest.update(b"\0")
    digest.update(source.read_bytes())
    return digest.hexdigest()


def _load_cache() -> dict[str, Any]:
    """Load the render cache, treating missing or invalid data as empty."""
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        return {}
    examples = cache.get("examples")
    return examples if isinstance(examples, dict) else {}


def _cache_hit(entry: Any, digest: str) -> bool:
    """Return whether a cache entry matches and all of its outputs exist."""
    if not isinstance(entry, dict) or entry.get("digest") != digest:
        return False
    outputs = entry.get("outputs")
    return isinstance(outputs, list) and all(
        isinstance(relative_path, str)
        and (OUTPUT_DIRECTORY / relative_path).is_file()
        for relative_path in outputs
    )


def _write_if_changed(path: Path, content: str | bytes) -> None:
    """Write generated content only when its bytes have changed."""
    encoded = content.encode() if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _prune_stale_outputs(cache: dict[str, dict[str, Any]]) -> int:
    """Delete generated files that no current cache entry accounts for."""
    accounted_for = {
        Path(relative_path)
        for entry in cache.values()
        for relative_path in entry["outputs"]
    }
    stale_paths = [
        path
        for path in OUTPUT_DIRECTORY.rglob("*")
        if path.is_file()
        and path != CACHE_PATH
        and path.relative_to(OUTPUT_DIRECTORY) not in accounted_for
    ]
    for path in stale_paths:
        path.unlink()

    directories = sorted(
        (path for path in OUTPUT_DIRECTORY.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            path.rmdir()
        except OSError:
            pass
    return len(stale_paths)


def _render_example(source: Path) -> tuple[str, dict[str, bytes]]:
    """Execute one percent notebook and export its cells and outputs to Markdown."""
    import jupytext
    from nbconvert import MarkdownExporter
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    executor = ExecutePreprocessor(
        timeout=60,
        kernel_name="python3",
    )
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
    """Generate include fragments before MkDocs discovers documentation files."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    previous_cache = _load_cache()
    next_cache: dict[str, dict[str, Any]] = {}
    shared_digest = _shared_dependency_digest()
    force = os.environ.get("DOCS_FORCE_EXAMPLES") == "1"
    rendered = 0

    for source in sorted(EXAMPLE_DIRECTORY.glob("*.py")):
        digest = _example_digest(source, shared_digest)
        cache_key = source.relative_to(REPOSITORY_ROOT).as_posix()
        previous_entry = previous_cache.get(cache_key)
        if not force and _cache_hit(previous_entry, digest):
            next_cache[cache_key] = previous_entry
            continue

        markdown, outputs = _render_example(source)
        relative_outputs = [f"{source.stem}.md"]
        _write_if_changed(OUTPUT_DIRECTORY / relative_outputs[0], markdown)
        for relative_path, content in outputs.items():
            normalized_path = Path(relative_path).as_posix()
            _write_if_changed(OUTPUT_DIRECTORY / normalized_path, content)
            relative_outputs.append(normalized_path)
        next_cache[cache_key] = {
            "digest": digest,
            "outputs": sorted(relative_outputs),
        }
        rendered += 1

    cache = {
        "version": CACHE_VERSION,
        "examples": next_cache,
    }
    _write_if_changed(CACHE_PATH, f"{json.dumps(cache, indent=2, sort_keys=True)}\n")
    pruned = _prune_stale_outputs(next_cache)
    LOGGER.info(
        "Notebook examples: rendered %d, reused %d, pruned %d",
        rendered,
        len(next_cache) - rendered,
        pruned,
    )
    return config
