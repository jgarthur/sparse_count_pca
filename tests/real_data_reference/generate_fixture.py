"""Generate deterministic PBMC3k count fixtures for real-data oracle tests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
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


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source_archive(
    archive_path: Path,
) -> tuple[sparse.csr_matrix, list[str], list[tuple[str, str]]]:
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

    return matrix, barcodes, gene_rows


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
) -> np.ndarray:
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
    return selected


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


def _artifact_record(path: Path, matrix: sparse.csr_matrix) -> dict[str, Any]:
    """Return the compact integrity contract for a count artifact."""
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "nnz": int(matrix.nnz),
        "total_count": int(matrix.sum(dtype=np.int64)),
    }


def _write_fixture(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Generate both matrices and their compact provenance manifest."""
    source, barcodes, gene_rows = _read_source_archive(archive_path)
    cell_indices = _select_cells(source, barcodes)
    selected_cells = source[cell_indices]
    gene_indices = _select_genes(selected_cells, gene_rows)
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

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "source": {
            "name": "3k PBMCs from a Healthy Donor",
            "dataset_url": SOURCE_DATASET_URL,
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "license": "CC-BY-4.0",
        },
        "artifacts": {
            "raw_counts": _artifact_record(raw_path, raw),
            "equal_depth_counts": {
                **_artifact_record(equal_depth_path, equal_depth),
                "row_total": TARGET_CELL_TOTAL,
            },
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
                f"nnz={artifact['nnz']} total_count={artifact['total_count']}"
            )
    print(f"wrote {args.output_dir / MANIFEST_FILENAME}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
