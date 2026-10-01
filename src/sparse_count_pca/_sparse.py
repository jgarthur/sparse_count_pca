"""Shared sparse-matrix traversal helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


def _csr_with_data(matrix: sparse.csr_matrix, data: NDArray[Any]) -> sparse.csr_matrix:
    """Build CSR with supplied data and one owned copy of its support arrays."""
    index_dtype = (
        np.int64
        if max(matrix.shape + (matrix.nnz,)) > np.iinfo(np.int32).max
        else np.int32
    )
    return sparse.csr_matrix(
        (
            data,
            matrix.indices.astype(index_dtype, copy=True),
            matrix.indptr.astype(index_dtype, copy=True),
        ),
        shape=matrix.shape,
        copy=False,
    )


def _support_row_blocks(
    matrix: sparse.csr_matrix,
    target_nnz: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield whole-row blocks containing about ``target_nnz`` stored values.

    Raises:
        ValueError: If ``target_nnz`` is not positive.
    """
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
