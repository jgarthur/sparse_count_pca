"""Integrity tests for the committed PBMC3k real-data count fixtures."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

FIXTURE_DIR = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_csr_sha256(matrix: sparse.csr_matrix) -> str:
    """Hash canonical CSR content using the manifest's portable encoding."""
    digest = hashlib.sha256()
    for values in (matrix.shape, matrix.indptr, matrix.indices, matrix.data):
        encoded = np.ascontiguousarray(values, dtype="<i8")
        digest.update(encoded.tobytes())
    return digest.hexdigest()


def _integer_summary(values: np.ndarray) -> dict[str, int | float]:
    """Compute the integer diagnostic summary recorded in the manifest."""
    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
    }


def _assert_summary_matches(
    actual: dict[str, int | float], expected: dict[str, int | float]
) -> None:
    """Compare exact endpoints and floating quantiles from a summary."""
    assert actual["min"] == expected["min"]
    assert actual["max"] == expected["max"]
    for key in ("q10", "median", "q90"):
        assert actual[key] == pytest.approx(expected[key], rel=0.0, abs=1e-12)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Load the committed fixture manifest once for this module."""
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrices(manifest: dict[str, Any]) -> dict[str, sparse.csr_matrix]:
    """Load both sparse artifacts in manifest order."""
    return {
        label: sparse.load_npz(FIXTURE_DIR / record["filename"])
        for label, record in manifest["artifacts"].items()
        if isinstance(record, dict)
    }


def test_artifact_files_match_manifest_checksums(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Committed files match their bytewise and canonical-CSR checksums."""
    for label, matrix in matrices.items():
        record = manifest["artifacts"][label]
        path = FIXTURE_DIR / record["filename"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256_file(path) == record["sha256"]
        assert _canonical_csr_sha256(matrix) == record["canonical_csr_sha256"]


def test_count_artifacts_are_canonical_nonnegative_int32_csr(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Both artifacts have the promised shape and strict canonical CSR form."""
    expected_shape = (
        manifest["selection"]["cells"]["selected_count"],
        manifest["selection"]["genes"]["selected_count"],
    )
    for matrix in matrices.values():
        assert isinstance(matrix, sparse.csr_matrix)
        assert matrix.shape == expected_shape
        assert matrix.dtype == np.dtype(np.int32)
        assert matrix.indices.dtype == np.dtype(np.int32)
        assert matrix.indptr.dtype == np.dtype(np.int32)
        assert matrix.has_sorted_indices
        assert matrix.has_canonical_format
        assert np.all(matrix.data > 0)
        assert np.all(np.diff(matrix.indptr) > 0)
        column_totals = np.asarray(matrix.sum(axis=0, dtype=np.int64)).ravel()
        assert np.all(column_totals > 0)


def test_equal_depth_counts_are_a_cellwise_subsample_of_raw_counts(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Equal-depth counts never exceed raw counts and have exact row totals."""
    raw = matrices["raw_counts"]
    equal_depth = matrices["equal_depth_counts"]
    target = manifest["downsampling"]["target_cell_total"]

    raw_totals = np.asarray(raw.sum(axis=1, dtype=np.int64)).ravel()
    equal_totals = np.asarray(equal_depth.sum(axis=1, dtype=np.int64)).ravel()
    np.testing.assert_array_equal(equal_totals, np.full(raw.shape[0], target))
    assert np.all(raw_totals >= equal_totals)

    removed = raw - equal_depth
    assert np.all(removed.data >= 0)


def test_manifest_identities_and_per_axis_counts_match_artifacts(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Manifest identities preserve source order and align with matrix axes."""
    raw = matrices["raw_counts"]
    equal_depth = matrices["equal_depth_counts"]
    cells = manifest["cells"]
    genes = manifest["genes"]
    source_genes, source_cells = manifest["source"]["source_shape_genes_by_cells"]

    assert len(cells) == raw.shape[0]
    assert len(genes) == raw.shape[1]

    cell_indices = [entry["source_index_zero_based"] for entry in cells]
    barcodes = [entry["barcode"] for entry in cells]
    assert cell_indices == sorted(cell_indices)
    assert len(set(cell_indices)) == len(cell_indices)
    assert len(set(barcodes)) == len(barcodes)
    assert all(0 <= index < source_cells for index in cell_indices)
    assert all(
        entry["full_total"]
        >= manifest["selection"]["cells"]["full_count_min_inclusive"]
        for entry in cells
    )
    raw_row_totals = np.asarray(raw.sum(axis=1, dtype=np.int64)).ravel()
    np.testing.assert_array_equal(
        raw_row_totals,
        np.asarray([entry["raw_restricted_total"] for entry in cells]),
    )

    gene_indices = [entry["source_index_zero_based"] for entry in genes]
    gene_ids = [entry["ensembl_id"] for entry in genes]
    gene_symbols = [entry["symbol"] for entry in genes]
    assert gene_indices == sorted(gene_indices)
    assert len(set(gene_indices)) == len(gene_indices)
    assert len(set(gene_ids)) == len(gene_ids)
    assert len(set(gene_symbols)) == len(gene_symbols)
    assert all(0 <= index < source_genes for index in gene_indices)

    selected_by = Counter(entry["selected_by"] for entry in genes)
    gene_selection = manifest["selection"]["genes"]
    assert selected_by == {
        "top_total": gene_selection["top_total_count"],
        "hash": gene_selection["hash_selected_count"],
    }
    top_totals = [
        entry["raw_total"] for entry in genes if entry["selected_by"] == "top_total"
    ]
    hash_totals = [
        entry["raw_total"] for entry in genes if entry["selected_by"] == "hash"
    ]
    assert min(top_totals) >= max(hash_totals)
    assert all(
        entry["raw_detection"]
        >= gene_selection["hash_candidate_min_detection_in_selected_cells"]
        for entry in genes
        if entry["selected_by"] == "hash"
    )

    for prefix, matrix in (("raw", raw), ("equal_depth", equal_depth)):
        totals = np.asarray(matrix.sum(axis=0, dtype=np.int64)).ravel()
        detections = np.asarray(matrix.getnnz(axis=0)).ravel()
        np.testing.assert_array_equal(
            totals, np.asarray([entry[f"{prefix}_total"] for entry in genes])
        )
        np.testing.assert_array_equal(
            detections,
            np.asarray([entry[f"{prefix}_detection"] for entry in genes]),
        )


def test_manifest_matrix_diagnostics_match_artifacts(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Recorded sparse, margin, and count diagnostics reproduce from artifacts."""
    for label, matrix in matrices.items():
        expected = manifest["diagnostics"][label]
        row_totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
        column_totals = np.asarray(matrix.sum(axis=0, dtype=np.int64)).ravel()
        detections = np.asarray(matrix.getnnz(axis=0)).ravel()

        assert expected["shape"] == list(matrix.shape)
        assert expected["dtype"] == str(matrix.dtype)
        assert expected["nnz"] == matrix.nnz
        assert expected["total_count"] == int(row_totals.sum(dtype=np.int64))
        assert expected["max_count"] == int(matrix.data.max())
        assert expected["empty_cells"] == int(np.count_nonzero(row_totals == 0))
        assert expected["empty_genes"] == int(np.count_nonzero(column_totals == 0))
        assert expected["density"] == pytest.approx(
            matrix.nnz / np.prod(matrix.shape), rel=0.0, abs=0.0
        )
        assert expected["stored_singleton_fraction"] == pytest.approx(
            np.mean(matrix.data == 1), rel=0.0, abs=0.0
        )
        _assert_summary_matches(_integer_summary(row_totals), expected["row_totals"])
        _assert_summary_matches(
            _integer_summary(column_totals), expected["gene_totals"]
        )
        _assert_summary_matches(
            _integer_summary(detections), expected["gene_detections"]
        )

    full_totals = np.asarray(
        [entry["full_total"] for entry in manifest["cells"]], dtype=np.int64
    )
    _assert_summary_matches(
        _integer_summary(full_totals),
        manifest["diagnostics"]["selected_full_cell_totals"],
    )


def test_equal_depth_clipping_diagnostics_reproduce(
    manifest: dict[str, Any], matrices: dict[str, sparse.csr_matrix]
) -> None:
    """Equal-depth residual diagnostics retain both tails and support growth."""
    matrix = matrices["equal_depth_counts"]
    expected = manifest["diagnostics"]["equal_depth_scaled_nb_clipping"]
    theta = expected["theta"]
    alpha = 1.0 / theta
    clip = float(np.sqrt(matrix.shape[0] / 30.0))
    dense = matrix.toarray().astype(np.float64)
    means = dense.mean(axis=0)
    residuals = (dense - means) / np.sqrt(means * (1.0 + alpha * means))
    below = residuals < -clip
    above = residuals > clip
    structural_zero_below = below & (dense == 0)
    added_support = int(np.count_nonzero(structural_zero_below))

    assert alpha == pytest.approx(expected["alpha"], rel=0.0, abs=0.0)
    assert clip == pytest.approx(expected["clip"], rel=0.0, abs=0.0)
    assert residuals.min() == pytest.approx(
        expected["residual_min"], rel=1e-14, abs=1e-14
    )
    assert residuals.max() == pytest.approx(
        expected["residual_max"], rel=1e-14, abs=1e-14
    )
    assert np.max(np.abs(residuals.mean(axis=0))) == pytest.approx(
        expected["max_abs_gene_mean"], rel=0.0, abs=1e-14
    )
    assert np.count_nonzero(below) == expected["entries_below_negative_clip"]
    assert np.count_nonzero(above) == expected["entries_above_positive_clip"]
    assert added_support == expected["structural_zeros_below_negative_clip"]
    assert (
        np.count_nonzero(residuals.min(axis=0) < -clip)
        == expected["genes_below_negative_clip"]
    )
    assert (
        np.count_nonzero(residuals.max(axis=0) > clip)
        == expected["genes_above_positive_clip"]
    )
    assert matrix.nnz + added_support == expected["symmetric_clipped_nnz"]
    assert (matrix.nnz + added_support) / matrix.nnz == pytest.approx(
        expected["symmetric_support_growth_ratio"], rel=0.0, abs=0.0
    )
    assert added_support > 0
    assert np.count_nonzero(above) > 0
