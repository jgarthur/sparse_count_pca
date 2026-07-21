"""Generate deterministic PBMC3k count fixtures for real-data oracle tests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from scipy.io import mmread

SOURCE_ARCHIVE_SHA256 = (
    "847d6ebd9a1ec9a768f2be7e40ca42cbfe75ebeb6d76a4c24167041699dc28b5"
)
SOURCE_DATASET_URL = (
    "https://www.10xgenomics.com/datasets/"
    "3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0"
)
SOURCE_ARCHIVE_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/"
    "pbmc3k_filtered_gene_bc_matrices.tar.gz"
)
SOURCE_PREFIX = "filtered_gene_bc_matrices/hg19/"
SOURCE_MEMBERS = {
    "matrix": f"{SOURCE_PREFIX}matrix.mtx",
    "genes": f"{SOURCE_PREFIX}genes.tsv",
    "barcodes": f"{SOURCE_PREFIX}barcodes.tsv",
}

CELL_HASH_NAMESPACE = "pbmc3k-cell-v1:"
GENE_HASH_NAMESPACE = "pbmc3k-gene-v1:"
MOLECULE_HASH_NAMESPACE = b"sparse-count-pca/pbmc3k/equal-depth/v1\0"
MIN_FULL_CELL_TOTAL = 2_500
N_CELLS = 256
N_GENES = 1_024
N_TOP_GENES = 128
MIN_GENE_DETECTION = 30
TARGET_CELL_TOTAL = 1_000

RAW_FILENAME = "pbmc3k_raw_counts.npz"
EQUAL_DEPTH_FILENAME = "pbmc3k_equal_depth_counts.npz"
MANIFEST_FILENAME = "manifest.json"


def _sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source_archive(
    archive_path: Path,
) -> tuple[sparse.csr_matrix, list[str], list[tuple[str, str]], dict[str, str]]:
    """Validate and read the three PBMC3k matrix members from the archive."""
    actual_sha256 = _sha256_file(archive_path)
    if actual_sha256 != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"Unexpected archive SHA-256 {actual_sha256}; "
            f"expected {SOURCE_ARCHIVE_SHA256}"
        )

    member_bytes: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for label, member_name in SOURCE_MEMBERS.items():
            extracted = archive.extractfile(member_name)
            if extracted is None:
                raise ValueError(f"Archive member is not a regular file: {member_name}")
            member_bytes[label] = extracted.read()

    matrix = sparse.csr_matrix(mmread(io.BytesIO(member_bytes["matrix"])).T)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape != (2_700, 32_738):
        raise ValueError(f"Unexpected source matrix shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Source matrix contains non-finite counts")
    if np.any(matrix.data < 0) or np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError("Source matrix contains non-integer or negative counts")
    if matrix.data.size and matrix.data.max() > np.iinfo(np.int32).max:
        raise ValueError("Source counts do not fit in int32")
    matrix = matrix.astype(np.int32)

    barcodes = member_bytes["barcodes"].decode("utf-8").splitlines()
    gene_rows = [
        tuple(line.split("\t"))
        for line in member_bytes["genes"].decode("utf-8").splitlines()
    ]
    if len(barcodes) != matrix.shape[0]:
        raise ValueError("Barcode count does not match the source matrix")
    if len(gene_rows) != matrix.shape[1]:
        raise ValueError("Gene count does not match the source matrix")
    if any(len(row) != 2 for row in gene_rows):
        raise ValueError("Every genes.tsv row must have an Ensembl ID and symbol")

    member_checksums = {
        SOURCE_MEMBERS[label]: _sha256_bytes(contents)
        for label, contents in member_bytes.items()
    }
    return matrix, barcodes, gene_rows, member_checksums


def _ranked_digest(namespace: str, identity: str) -> bytes:
    """Return the versioned selection digest for an identity."""
    return hashlib.sha256(f"{namespace}{identity}".encode()).digest()


def _select_cells(matrix: sparse.csr_matrix, barcodes: list[str]) -> np.ndarray:
    """Select sufficiently deep cells by versioned hash rank."""
    full_totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
    eligible = np.flatnonzero(full_totals >= MIN_FULL_CELL_TOTAL)
    if eligible.size < N_CELLS:
        raise ValueError(f"Only {eligible.size} cells pass the full-depth threshold")
    selected = sorted(
        eligible,
        key=lambda index: (
            _ranked_digest(CELL_HASH_NAMESPACE, barcodes[index]),
            barcodes[index],
            int(index),
        ),
    )[:N_CELLS]
    return np.sort(np.asarray(selected, dtype=np.int64))


def _select_genes(
    selected_cells: sparse.csr_matrix,
    gene_rows: list[tuple[str, str]],
) -> tuple[np.ndarray, set[int]]:
    """Select abundant and hash-ranked detected genes, then restore source order."""
    gene_ids = [row[0] for row in gene_rows]
    gene_totals = np.asarray(selected_cells.sum(axis=0, dtype=np.int64)).ravel()
    detections = np.asarray(selected_cells.getnnz(axis=0)).ravel()

    top_genes = sorted(
        range(selected_cells.shape[1]),
        key=lambda index: (-int(gene_totals[index]), gene_ids[index], index),
    )[:N_TOP_GENES]
    top_gene_set = set(top_genes)
    eligible_remainder = [
        index
        for index in range(selected_cells.shape[1])
        if detections[index] >= MIN_GENE_DETECTION and index not in top_gene_set
    ]
    needed = N_GENES - N_TOP_GENES
    if len(eligible_remainder) < needed:
        raise ValueError(
            f"Only {len(eligible_remainder)} non-top genes pass the detection threshold"
        )
    hashed_genes = sorted(
        eligible_remainder,
        key=lambda index: (
            _ranked_digest(GENE_HASH_NAMESPACE, gene_ids[index]),
            gene_ids[index],
            index,
        ),
    )[:needed]
    selected = np.sort(np.asarray(top_genes + hashed_genes, dtype=np.int64))
    return selected, top_gene_set


def _molecule_digest(barcode: str, gene_id: str, ordinal: int) -> bytes:
    """Return a stable priority digest for one molecule."""
    identity = (
        MOLECULE_HASH_NAMESPACE
        + barcode.encode()
        + b"\0"
        + gene_id.encode()
        + b"\0"
        + str(ordinal).encode("ascii")
    )
    return hashlib.sha256(identity).digest()


def _downsample_equal_depth(
    matrix: sparse.csr_matrix,
    barcodes: list[str],
    gene_ids: list[str],
) -> sparse.csr_matrix:
    """Retain the lowest-hash molecules to give every cell exactly 1,000 counts."""
    rows: list[sparse.csr_matrix] = []
    for row_index, barcode in enumerate(barcodes):
        row = matrix.getrow(row_index)
        total = int(row.sum())
        if total < TARGET_CELL_TOTAL:
            raise ValueError(
                f"Cell {barcode} has only {total} selected-gene counts; "
                f"cannot downsample to {TARGET_CELL_TOTAL}"
            )
        candidates: list[tuple[bytes, str, int, int]] = []
        for column, count in zip(row.indices, row.data, strict=True):
            gene_id = gene_ids[column]
            candidates.extend(
                (
                    _molecule_digest(barcode, gene_id, ordinal),
                    gene_id,
                    ordinal,
                    int(column),
                )
                for ordinal in range(int(count))
            )
        candidates.sort()
        retained_columns = [
            column for _, _, _, column in candidates[:TARGET_CELL_TOTAL]
        ]
        retained = np.bincount(retained_columns, minlength=matrix.shape[1]).astype(
            np.int32
        )
        nonzero = np.flatnonzero(retained)
        rows.append(
            sparse.csr_matrix(
                (
                    retained[nonzero],
                    nonzero.astype(np.int32),
                    np.array([0, nonzero.size], dtype=np.int32),
                ),
                shape=(1, matrix.shape[1]),
                dtype=np.int32,
            )
        )
    result = sparse.vstack(rows, format="csr", dtype=np.int32)
    row_totals = np.asarray(result.sum(axis=1, dtype=np.int64)).ravel()
    if not np.array_equal(row_totals, np.full(N_CELLS, TARGET_CELL_TOTAL)):
        raise RuntimeError("Equal-depth downsampling did not preserve exact row totals")
    if np.any(np.asarray(result.sum(axis=0, dtype=np.int64)).ravel() == 0):
        raise RuntimeError("Equal-depth downsampling produced an empty gene")
    return result


def _integer_summary(values: np.ndarray) -> dict[str, int | float]:
    """Return compact exact and linear-quantile diagnostics for integer values."""
    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.quantile(values, 0.5)),
        "q90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
    }


def _matrix_stats(matrix: sparse.csr_matrix) -> dict[str, Any]:
    """Return sparse count diagnostics used to audit the committed fixture."""
    row_totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
    column_totals = np.asarray(matrix.sum(axis=0, dtype=np.int64)).ravel()
    detections = np.asarray(matrix.getnnz(axis=0)).ravel()
    return {
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / np.prod(matrix.shape)),
        "total_count": int(row_totals.sum(dtype=np.int64)),
        "max_count": int(matrix.data.max()),
        "row_totals": _integer_summary(row_totals),
        "gene_totals": _integer_summary(column_totals),
        "gene_detections": _integer_summary(detections),
        "empty_cells": int(np.count_nonzero(row_totals == 0)),
        "empty_genes": int(np.count_nonzero(column_totals == 0)),
        "stored_singleton_fraction": float(np.mean(matrix.data == 1)),
    }


def _canonical_csr_sha256(matrix: sparse.csr_matrix) -> str:
    """Hash CSR shape and arrays after portable little-endian int64 encoding."""
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.indptr.astype("<i8", copy=False).tobytes())
    digest.update(matrix.indices.astype("<i8", copy=False).tobytes())
    digest.update(matrix.data.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _clipping_diagnostics(matrix: sparse.csr_matrix) -> dict[str, Any]:
    """Measure constant-dispersion SCTransform clipping coverage."""
    theta = 100.0
    alpha = 1.0 / theta
    clip = float(np.sqrt(matrix.shape[0] / 30.0))
    dense = matrix.toarray().astype(np.float64)
    means = dense.mean(axis=0)
    residuals = (dense - means) / np.sqrt(means * (1.0 + alpha * means))
    below = residuals < -clip
    above = residuals > clip
    structural_zero_below = below & (dense == 0)
    added_support = int(np.count_nonzero(structural_zero_below))
    return {
        "theta": theta,
        "alpha": alpha,
        "clip": clip,
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
        "max_abs_gene_mean": float(np.max(np.abs(residuals.mean(axis=0)))),
        "entries_below_negative_clip": int(np.count_nonzero(below)),
        "entries_above_positive_clip": int(np.count_nonzero(above)),
        "structural_zeros_below_negative_clip": added_support,
        "genes_below_negative_clip": int(
            np.count_nonzero(residuals.min(axis=0) < -clip)
        ),
        "genes_above_positive_clip": int(
            np.count_nonzero(residuals.max(axis=0) > clip)
        ),
        "symmetric_clipped_nnz": int(matrix.nnz + added_support),
        "symmetric_support_growth_ratio": float(
            (matrix.nnz + added_support) / matrix.nnz
        ),
    }


def _artifact_record(path: Path, matrix: sparse.csr_matrix) -> dict[str, Any]:
    """Return file and canonical-content checksums for a CSR artifact."""
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "canonical_csr_sha256": _canonical_csr_sha256(matrix),
        "bytes": path.stat().st_size,
    }


def _write_fixture(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Generate both matrices and their complete provenance manifest."""
    source, barcodes, gene_rows, member_checksums = _read_source_archive(archive_path)
    cell_indices = _select_cells(source, barcodes)
    selected_cells = source[cell_indices]
    gene_indices, top_gene_set = _select_genes(selected_cells, gene_rows)
    raw = selected_cells[:, gene_indices].tocsr().astype(np.int32)
    raw.sort_indices()

    selected_barcodes = [barcodes[index] for index in cell_indices]
    selected_gene_rows = [gene_rows[index] for index in gene_indices]
    selected_gene_ids = [row[0] for row in selected_gene_rows]
    equal_depth = _downsample_equal_depth(raw, selected_barcodes, selected_gene_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_FILENAME
    equal_depth_path = output_dir / EQUAL_DEPTH_FILENAME
    sparse.save_npz(raw_path, raw, compressed=True)
    sparse.save_npz(equal_depth_path, equal_depth, compressed=True)

    full_totals = np.asarray(source.sum(axis=1, dtype=np.int64)).ravel()
    raw_row_totals = np.asarray(raw.sum(axis=1, dtype=np.int64)).ravel()
    raw_gene_totals = np.asarray(raw.sum(axis=0, dtype=np.int64)).ravel()
    raw_detections = np.asarray(raw.getnnz(axis=0)).ravel()
    equal_gene_totals = np.asarray(equal_depth.sum(axis=0, dtype=np.int64)).ravel()
    equal_detections = np.asarray(equal_depth.getnnz(axis=0)).ravel()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "name": "3k PBMCs from a Healthy Donor",
            "producer": "10x Genomics",
            "reference_genome": "hg19",
            "dataset_url": SOURCE_DATASET_URL,
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "archive_member_sha256": member_checksums,
            "source_shape_genes_by_cells": [32_738, 2_700],
            "license": {
                "spdx": "CC-BY-4.0",
                "name": "Creative Commons Attribution 4.0 International",
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": "3k PBMCs from a Healthy Donor, 10x Genomics",
            },
        },
        "selection": {
            "ordering": "original source order after selection",
            "hash": "SHA-256 digest ranked lexicographically as bytes",
            "cells": {
                "full_count_min_inclusive": MIN_FULL_CELL_TOTAL,
                "eligible_count": int(
                    np.count_nonzero(full_totals >= MIN_FULL_CELL_TOTAL)
                ),
                "selected_count": N_CELLS,
                "hash_namespace": CELL_HASH_NAMESPACE,
            },
            "genes": {
                "selected_count": N_GENES,
                "top_total_count": N_TOP_GENES,
                "top_total_tie_break": "Ensembl ID, then source index",
                "hash_selected_count": N_GENES - N_TOP_GENES,
                "hash_candidate_min_detection_in_selected_cells": (MIN_GENE_DETECTION),
                "hash_namespace": GENE_HASH_NAMESPACE,
            },
        },
        "downsampling": {
            "target_cell_total": TARGET_CELL_TOTAL,
            "hash_namespace_utf8_with_trailing_nul": (
                MOLECULE_HASH_NAMESPACE.decode("utf-8")
            ),
            "molecule_identity": (
                "namespace + barcode + NUL + Ensembl ID + NUL + "
                "zero-based decimal ordinal within the cell-gene count"
            ),
            "selection": (
                "retain the target number of molecules ranked by "
                "(SHA-256 digest, Ensembl ID, ordinal)"
            ),
        },
        "artifacts": {
            "canonical_csr_checksum_encoding": (
                "SHA-256 over shape, indptr, indices, and data in that order, "
                "each encoded as contiguous little-endian signed int64"
            ),
            "raw_counts": _artifact_record(raw_path, raw),
            "equal_depth_counts": _artifact_record(equal_depth_path, equal_depth),
        },
        "diagnostics": {
            "selected_full_cell_totals": _integer_summary(full_totals[cell_indices]),
            "raw_counts": _matrix_stats(raw),
            "equal_depth_counts": _matrix_stats(equal_depth),
            "equal_depth_scaled_nb_clipping": _clipping_diagnostics(equal_depth),
        },
        "cells": [
            {
                "barcode": barcodes[source_index],
                "source_index_zero_based": int(source_index),
                "full_total": int(full_totals[source_index]),
                "raw_restricted_total": int(raw_row_totals[output_index]),
            }
            for output_index, source_index in enumerate(cell_indices)
        ],
        "genes": [
            {
                "ensembl_id": gene_rows[source_index][0],
                "symbol": gene_rows[source_index][1],
                "source_index_zero_based": int(source_index),
                "selected_by": (
                    "top_total" if int(source_index) in top_gene_set else "hash"
                ),
                "raw_total": int(raw_gene_totals[output_index]),
                "raw_detection": int(raw_detections[output_index]),
                "equal_depth_total": int(equal_gene_totals[output_index]),
                "equal_depth_detection": int(equal_detections[output_index]),
            }
            for output_index, source_index in enumerate(gene_indices)
        ],
        "generation_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="path to the official pbmc3k_filtered_gene_bc_matrices.tar.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="fixture output directory (default: directory containing this script)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate fixtures, fail on source mismatch, and print artifact checksums."""
    args = _parse_args()
    manifest = _write_fixture(args.archive, args.output_dir)
    print(f"verified source archive: {SOURCE_ARCHIVE_SHA256}")
    for artifact in manifest["artifacts"].values():
        if isinstance(artifact, dict):
            print(
                f"wrote {artifact['filename']}: sha256={artifact['sha256']} "
                f"canonical_csr_sha256={artifact['canonical_csr_sha256']}"
            )
    print(f"wrote {args.output_dir / MANIFEST_FILENAME}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
