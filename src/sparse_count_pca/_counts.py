"""Count-matrix canonicalization and variable-mask validation."""

from __future__ import annotations

import warnings
from typing import TypeAlias

import numpy as np
from anndata.abc import CSCDataset, CSRDataset
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

CountMatrix: TypeAlias = (
    ArrayLike | sparse.spmatrix | sparse.sparray | CSRDataset | CSCDataset
)
CSRMatrix: TypeAlias = sparse.csr_matrix | sparse.csr_array
BoolArray: TypeAlias = NDArray[np.bool_]
Float64Array: TypeAlias = NDArray[np.float64]
COUNT_INTEGER_ATOL = 1e-8
_MAX_EXACT_CONSECUTIVE_FLOAT64_INTEGER = 2**53
_ROUND_TRIP_BLOCK_SIZE = 1_000_000


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
    for start in range(0, values.size, _ROUND_TRIP_BLOCK_SIZE):
        block = values[start : start + _ROUND_TRIP_BLOCK_SIZE]
        with np.errstate(over="ignore", invalid="ignore"):
            converted = block.astype(np.float64)
            restored = converted.astype(values.dtype)
        if not np.array_equal(restored, block):
            return False
    return True


def _sum_counts(X: CSRMatrix, *, axis: int) -> Float64Array:
    """Sum sparse counts in float64 without casting the whole matrix."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(X.sum(axis=axis, dtype=np.float64)).ravel()


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
        elif not X.has_canonical_format or (X.data == 0).any():
            reasons = []
            if not X.has_canonical_format:
                reasons.append("duplicate or unsorted column indices")
            if (X.data == 0).any():
                reasons.append("explicitly stored zeros")
            warnings.warn(
                "CSR input was copied for canonicalization: " + " and ".join(reasons),
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
    if not np.isfinite(X.data).all():
        raise ValueError("Input contains NaN or inf")
    if (X.data < 0).any():
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
        and not np.allclose(X.data, np.rint(X.data), rtol=0.0, atol=COUNT_INTEGER_ATOL)
    ):
        raise ValueError(
            "Input contains non-integer values; pass check_values=False to skip"
        )
    if (X.data == 0).any():
        X.eliminate_zeros()
    # Counts are nonnegative and explicit zeros are gone, so an empty CSR row
    # is exactly a cell with zero total counts. Inspecting indptr avoids another
    # O(nnz) floating-point margin pass for transforms that do not need totals.
    if (np.diff(X.indptr) == 0).any():
        raise ValueError("Cells with zero total counts are not supported")
    return X
