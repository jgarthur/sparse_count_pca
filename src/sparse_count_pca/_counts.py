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
BoolArray: TypeAlias = NDArray[np.bool_]
COUNT_INTEGER_ATOL = 1e-8


def _validate_boolean_mask(mask: ArrayLike, n_vars: int, *, name: str) -> BoolArray:
    """Require a one-dimensional, genuinely boolean variable mask."""
    values = np.asarray(mask)
    if values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be genuinely boolean")
    if values.ndim != 1 or values.shape[0] != n_vars:
        raise ValueError(f"{name} must be a boolean vector with length n_vars")
    return values


def _canonicalize_counts(X: CountMatrix, *, check_values: bool) -> sparse.csr_matrix:
    """Convert a count matrix to canonical CSR form and validate its data."""
    if isinstance(X, (CSRDataset, CSCDataset)):
        X = X.to_memory()

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
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
    X.sum_duplicates()
    X.sort_indices()
    if not np.isfinite(X.data).all():
        raise ValueError("Input contains NaN or inf")
    if (X.data < 0).any():
        raise ValueError("Input contains negative values")
    data_float64 = X.data.astype(np.float64)
    if X.dtype.itemsize > np.dtype(np.float64).itemsize or np.issubdtype(
        X.dtype, np.integer
    ):
        if not np.array_equal(data_float64.astype(X.dtype), X.data):
            raise ValueError("Input counts cannot be represented exactly as float64")
    if (
        check_values
        and np.issubdtype(X.dtype, np.floating)
        and not np.allclose(X.data, np.rint(X.data), rtol=0.0, atol=COUNT_INTEGER_ATOL)
    ):
        raise ValueError(
            "Input contains non-integer values; pass check_values=False to skip"
        )
    X.eliminate_zeros()
    return X
