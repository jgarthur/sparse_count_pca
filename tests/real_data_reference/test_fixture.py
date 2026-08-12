"""Compact integrity tests for the committed PBMC3k count fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

FIXTURE_DIR = Path(__file__).resolve().parent
EXPECTED_ARTIFACTS = {"raw_counts", "equal_depth_counts"}


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Load the compact committed fixture manifest once for this module."""
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrices(manifest: dict[str, Any]) -> dict[str, sparse.csr_matrix]:
    """Load the two count artifacts named by the manifest."""
    return {
        label: sparse.load_npz(FIXTURE_DIR / record["filename"])
        for label, record in manifest["artifacts"].items()
    }


def test_count_artifacts_match_compact_integrity_records(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Manifest schema, checksums, sparse layout, and count totals are pinned."""
    assert manifest["schema_version"] == 2
    assert set(manifest["artifacts"]) == EXPECTED_ARTIFACTS

    for label, matrix in matrices.items():
        record = manifest["artifacts"][label]
        path = FIXTURE_DIR / record["filename"]
        row_totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
        column_totals = np.asarray(matrix.sum(axis=0, dtype=np.int64)).ravel()

        assert _sha256_file(path) == record["sha256"]
        assert isinstance(matrix, sparse.csr_matrix)
        assert list(matrix.shape) == record["shape"]
        assert str(matrix.dtype) == record["dtype"] == "int32"
        assert matrix.nnz == record["nnz"]
        assert int(row_totals.sum(dtype=np.int64)) == record["total_count"]
        assert matrix.has_canonical_format
        assert np.all(matrix.data > 0)
        assert np.all(row_totals > 0)
        assert np.all(column_totals > 0)


def test_equal_depth_counts_are_an_exact_cellwise_subsample(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Equal-depth counts have fixed totals and never exceed raw counts."""
    raw = matrices["raw_counts"]
    equal_depth = matrices["equal_depth_counts"]
    target = manifest["artifacts"]["equal_depth_counts"]["row_total"]
    equal_totals = np.asarray(equal_depth.sum(axis=1, dtype=np.int64)).ravel()

    np.testing.assert_array_equal(equal_totals, np.full(raw.shape[0], target))
    removed = raw - equal_depth
    assert np.all(removed.data >= 0)
