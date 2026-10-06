"""Count-matrix canonicalization and variable-mask validation."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
from anndata.abc import CSCDataset, CSRDataset
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from ._sparse import _support_row_blocks

CountMatrix: TypeAlias = (
    ArrayLike | sparse.spmatrix | sparse.sparray | CSRDataset | CSCDataset
)
CSRMatrix: TypeAlias = sparse.csr_matrix | sparse.csr_array
BoolArray: TypeAlias = NDArray[np.bool_]
Float64Array: TypeAlias = NDArray[np.float64]
COUNT_INTEGER_ATOL = 1e-8
_MAX_EXACT_CONSECUTIVE_FLOAT64_INTEGER = 2**53
_COUNT_BLOCK_SIZE = 100_000


def _count_data_blocks(values: NDArray[np.generic]) -> Iterator[NDArray[np.generic]]:
    """Yield fixed-size views for count validation with bounded scratch storage."""
    for start in range(0, values.size, _COUNT_BLOCK_SIZE):
        yield values[start : start + _COUNT_BLOCK_SIZE]


def _validate_boolean_mask(mask: ArrayLike, n_vars: int, *, name: str) -> BoolArray:
    """Require a one-dimensional, genuinely boolean variable mask."""
    values = np.asarray(mask)
    if values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be genuinely boolean")
    if values.ndim != 1 or values.shape[0] != n_vars:
        raise ValueError(
            f"{name} must be a boolean vector with length {n_vars}; "
            f"got shape {values.shape}"
        )
    return values


def _round_trips_through_float64(values: NDArray[np.generic]) -> bool:
    """Check exact float64 conversion with bounded temporary memory."""
    for block in _count_data_blocks(values):
        with np.errstate(over="ignore", invalid="ignore"):
            converted = block.astype(np.float64)
            restored = converted.astype(values.dtype)
        if not np.array_equal(restored, block):
            return False
    return True


def _sum_counts(X: CSRMatrix, *, axis: int) -> Float64Array:
    """Accumulate count margins in float64 using bounded stored-data blocks.

    Row margins use complete CSR rows, so one longer row can exceed the block
    target. Column margins use fixed-size data views. Empty rows and columns
    have zero margins, and input arrays are not modified.
    """
    if axis not in (-2, -1, 0, 1):
        raise ValueError("axis must be 0 or 1")
    axis %= 2
    totals = np.zeros(X.shape[1 - axis], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if axis == 0:
            for start in range(0, X.nnz, _COUNT_BLOCK_SIZE):
                stop = start + _COUNT_BLOCK_SIZE
                # Unbuffered accumulation keeps per-block work independent of
                # the number of columns; it is fast only for float64 values.
                np.add.at(
                    totals,
                    X.indices[start:stop],
                    X.data[start:stop].astype(np.float64, copy=False),
                )
        else:
            for row_start, row_stop, start, stop in _support_row_blocks(
                X, _COUNT_BLOCK_SIZE
            ):
                indptr = X.indptr[row_start : row_stop + 1] - start
                nonempty = np.diff(indptr) > 0
                values = X.data[start:stop].astype(np.float64, copy=False)
                totals[row_start:row_stop][nonempty] = np.add.reduceat(
                    values, indptr[:-1][nonempty]
                )
    return totals


def _empty_columns(X: CSRMatrix) -> BoolArray:
    """Flag variables whose counts are zero in every observation."""
    return _sum_counts(X, axis=0) == 0


def _canonicalize_counts(X: CountMatrix, *, check_values: bool) -> CSRMatrix:
    """Convert a count matrix to canonical CSR form and validate its data."""
    if isinstance(X, (CSRDataset, CSCDataset)):
        # Downstream transforms currently require an in-memory SciPy CSR matrix.
        X = X.to_memory()
        # The materialized matrix is private to this call.
        X = X if X.format == "csr" else X.tocsr()
    elif sparse.issparse(X):
        if X.format != "csr":
            # Converting another sparse format necessarily creates a private CSR.
            X = X.tocsr()
        else:
            has_stored_zeros = any(
                (block == 0).any() for block in _count_data_blocks(X.data)
            )
            if not X.has_canonical_format or has_stored_zeros:
                reasons = []
                remedies = []
                if not X.has_canonical_format:
                    reasons.append("duplicate or unsorted column indices")
                    remedies.append("sum_duplicates() and sort_indices()")
                if has_stored_zeros:
                    reasons.append("explicitly stored zeros")
                    remedies.append("eliminate_zeros()")
                warnings.warn(
                    "CSR input was copied for canonicalization ("
                    + " and ".join(reasons)
                    + "), leading to increased memory usage. Run "
                    + " and ".join(remedies)
                    + " on the input to avoid this.",
                    UserWarning,
                    stacklevel=3,
                )
                # Canonicalization mutates CSR, so copy only inputs that need it.
                X = X.copy()
    else:
        X = np.asarray(X)
        if not (
            np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)
        ):
            raise ValueError("Input counts must have a real numeric dtype")
        warnings.warn(
            "Dense input was converted to CSR. This may require substantial memory.",
            UserWarning,
            stacklevel=3,
        )
        X = sparse.csr_matrix(X)

    if not (np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)):
        raise ValueError("Input counts must have a real numeric dtype")
    if not X.has_canonical_format:
        X.sum_duplicates()
        # Canonical CSR keeps each row's column indices in ascending order.
        X.sort_indices()
    if any(not np.isfinite(block).all() for block in _count_data_blocks(X.data)):
        raise ValueError("Input contains NaN or inf")
    if any((block < 0).any() for block in _count_data_blocks(X.data)):
        raise ValueError("Input contains negative values")
    if np.issubdtype(X.dtype, np.integer) and X.dtype.itemsize > 4:
        maximum = int(np.max(X.data)) if X.data.size else 0
        if (
            maximum > _MAX_EXACT_CONSECUTIVE_FLOAT64_INTEGER
            and not _round_trips_through_float64(X.data)
        ):
            raise ValueError("Input counts cannot be represented exactly as float64")
    elif (
        np.issubdtype(X.dtype, np.floating)
        and X.dtype.itemsize > np.dtype(np.float64).itemsize
    ):
        if not _round_trips_through_float64(X.data):
            raise ValueError("Input counts cannot be represented exactly as float64")
    if (
        check_values
        and np.issubdtype(X.dtype, np.floating)
        and any(
            not np.allclose(block, np.rint(block), rtol=0.0, atol=COUNT_INTEGER_ATOL)
            for block in _count_data_blocks(X.data)
        )
    ):
        raise ValueError(
            "Input contains non-integer values; pass check_values=False to skip"
        )
    if any((block == 0).any() for block in _count_data_blocks(X.data)):
        X.eliminate_zeros()
    # Counts are nonnegative and explicit zeros are gone, so an empty CSR row
    # is exactly a cell with zero total counts. Inspecting indptr avoids another
    # O(nnz) floating-point margin pass for transforms that do not need totals.
    if (np.diff(X.indptr) == 0).any():
        raise ValueError(
            "Cells with zero total counts are not supported; filter empty rows "
            "out of the count matrix first"
        )
    return X
