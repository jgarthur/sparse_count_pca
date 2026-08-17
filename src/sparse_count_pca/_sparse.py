"""Shared sparse-matrix traversal helpers."""

from __future__ import annotations

from collections.abc import Iterator

from scipy import sparse


def _support_row_blocks(
    matrix: sparse.csr_matrix,
    target_nnz: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield whole-row blocks containing about ``target_nnz`` stored values."""
    if target_nnz <= 0:
        raise ValueError("target_nnz must be positive")

    row_start = 0
    while row_start < matrix.shape[0]:
        target = min(int(matrix.indptr[row_start]) + target_nnz, matrix.nnz)
        row_stop = int(matrix.indptr.searchsorted(target, side="right")) - 1
        row_stop = min(matrix.shape[0], max(row_start + 1, row_stop))
        value_start = int(matrix.indptr[row_start])
        value_stop = int(matrix.indptr[row_stop])
        yield row_start, row_stop, value_start, value_stop
        row_start = row_stop
