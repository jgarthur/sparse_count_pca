"""Tests for documentation build validation helpers."""

from types import SimpleNamespace

import pytest

from docs.hooks import render_examples
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


def test_render_examples_prunes_only_unaccounted_outputs(tmp_path, monkeypatch) -> None:
    """The docs hook retains cached outputs and removes stale generated files."""
    example_directory = tmp_path / "examples"
    output_directory = tmp_path / "docs" / "examples"
    example_directory.mkdir()
    output_directory.mkdir(parents=True)
    source = example_directory / "current.py"
    source.write_text("# %%\n1 + 1\n")

    retained_markdown = output_directory / "current.md"
    retained_asset = output_directory / "current_files" / "plot.png"
    stale_markdown = output_directory / "renamed.md"
    stale_asset = output_directory / "renamed_files" / "plot.png"
    cache_path = output_directory / ".render-cache.json"
    for path in (retained_markdown, retained_asset, stale_markdown, stale_asset):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    cache_path.write_text("old cache")

    monkeypatch.setattr(render_examples, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(render_examples, "EXAMPLE_DIRECTORY", example_directory)
    monkeypatch.setattr(render_examples, "OUTPUT_DIRECTORY", output_directory)
    monkeypatch.setattr(render_examples, "CACHE_PATH", cache_path)
    monkeypatch.setattr(render_examples, "_shared_dependency_digest", lambda: "all")
    monkeypatch.setattr(
        render_examples,
        "_example_digest",
        lambda source, shared_digest: "current",
    )
    monkeypatch.setattr(
        render_examples,
        "_load_cache",
        lambda: {
            "examples/current.py": {
                "digest": "current",
                "outputs": ["current.md", "current_files/plot.png"],
            }
        },
    )

    config = object()
    assert render_examples.on_config(config) is config
    assert retained_markdown.is_file()
    assert retained_asset.is_file()
    assert cache_path.is_file()
    assert not stale_markdown.exists()
    assert not stale_asset.exists()
    assert not stale_asset.parent.exists()


@pytest.mark.parametrize(
    ("ipc_supported", "path_limit", "expect_ipc"),
    [(True, 96, True), (False, 96, False), (True, 0, False)],
    ids=["ipc", "windows", "long-temp-path"],
)
def test_render_examples_kernel_transport(
    monkeypatch, ipc_supported, path_limit, expect_ipc
) -> None:
    """Kernels use ipc sockets only on supporting platforms with short enough paths."""
    pytest.importorskip("traitlets")
    monkeypatch.setattr(render_examples, "IPC_SUPPORTED", ipc_supported)
    monkeypatch.setattr(render_examples, "IPC_BASE_PATH_LIMIT", path_limit)

    with render_examples._kernel_config() as config:
        uses_ipc = config.KernelManager.get("transport") == "ipc"

    assert uses_ipc is expect_ipc
