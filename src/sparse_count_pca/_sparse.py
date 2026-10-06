"""Shared sparse-matrix traversal helpers."""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

_SUPPORT_CHECKSUM_CHUNK_SIZE = 1_000_000


def _csr_with_data(
    matrix: sparse.csr_matrix,
    data: NDArray[Any],
    *,
    share_structure: bool = False,
) -> sparse.csr_matrix:
    """Build CSR with supplied data, borrowing support only when requested.

    Owned support uses the narrowest index width the shape and stored-entry
    count allow. Borrowed support keeps wider int64 input indices instead of
    narrowing them; narrower input that the shape requires to be int64 is
    still copied. Borrowed support must not be mutated while the result lives.
    """
    index_dtype = np.dtype(
        np.int64
        if max(matrix.shape + (matrix.nnz,)) > np.iinfo(np.int32).max
        else np.int32
    )
    if share_structure:
        index_dtype = np.promote_types(index_dtype, matrix.indices.dtype)
    # The csr_matrix constructor narrows int64 support that fits int32 into a
    # copy; csr_array keeps it, and wrapping it in csr_matrix does not copy.
    return sparse.csr_matrix(
        sparse.csr_array(
            (
                data,
                matrix.indices.astype(index_dtype, copy=not share_structure),
                matrix.indptr.astype(index_dtype, copy=not share_structure),
            ),
            shape=matrix.shape,
            copy=False,
        ),
        copy=False,
    )


def _support_checksum(matrix: sparse.csr_matrix) -> int:
    """Return a CRC-32 of CSR indptr and indices for detecting in-place edits.

    CRC-32 accumulates across chunks, so chunking does not change the result.
    Contiguous chunks are hashed as views; strided borrowed support is copied
    one chunk at a time rather than as a whole array.
    """
    checksum = 0
    for values in (matrix.indptr, matrix.indices):
        for start in range(0, values.size, _SUPPORT_CHECKSUM_CHUNK_SIZE):
            chunk = values[start : start + _SUPPORT_CHECKSUM_CHUNK_SIZE]
            checksum = zlib.crc32(np.ascontiguousarray(chunk), checksum)
    return checksum


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
