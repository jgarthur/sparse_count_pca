"""Sparse-plus-low-rank builders for shifted logarithmic transforms."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse

from ._counts import _sum_counts
from ._representation import SparseLowRankMatrix
from ._sparse import _support_row_blocks

PriorProportions: TypeAlias = ArrayLike | None
Float64Array = NDArray[np.float64]
PRIOR_SUM_RTOL = 1e-8
PRIOR_SUM_ATOL = 1e-12
_LOG_TRANSFORM_BLOCK_NNZ = 100_000


def validate_positive_scalar(value: float, *, name: str) -> float:
    """Return a finite positive scalar parameter."""
    values = np.asarray(value)
    if values.ndim != 0 or values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be finite and positive")
    result = float(values)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def validate_dirichlet_prior(
    concentration: float,
    prior_proportions: PriorProportions,
    n_vars: int,
) -> tuple[float, Float64Array]:
    """Validate a total prior concentration and prior composition."""
    concentration = validate_positive_scalar(concentration, name="concentration")
    if prior_proportions is None:
        proportions = np.full(n_vars, 1.0 / n_vars, dtype=np.float64)
    else:
        proportions = np.asarray(prior_proportions, dtype=np.float64)
        if proportions.ndim != 1 or proportions.shape[0] != n_vars:
            raise ValueError("prior_proportions must have shape (n_vars,)")
        if not np.isfinite(proportions).all():
            raise ValueError("prior_proportions must be finite")
        if (proportions <= 0).any():
            raise ValueError("prior_proportions must be strictly positive")
        total = float(np.sum(proportions))
        if not np.isclose(
            total,
            1.0,
            rtol=PRIOR_SUM_RTOL,
            atol=PRIOR_SUM_ATOL,
        ):
            raise ValueError("prior_proportions must sum to one")
        proportions = proportions / total
    return concentration, proportions


def validate_size_factors(
    size_factors: ArrayLike,
    n_obs: int,
) -> Float64Array:
    """Return finite positive per-observation divisors as float64."""
    values = np.asarray(size_factors)
    if values.ndim != 1 or values.shape[0] != n_obs:
        raise ValueError("size_factors must have shape (n_obs,)")
    if values.dtype.kind not in "fiu":
        raise ValueError("size_factors must be finite and strictly positive")
    result = values.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("size_factors must be finite and strictly positive")
    if (result <= 0).any():
        raise ValueError("size_factors must be finite and strictly positive")
    return result


def _log1p_sparse_correction(
    X: sparse.csr_matrix,
    divisors: Float64Array | float,
    *,
    by_row: bool,
    dtype: DTypeLike,
    overflow_message: str,
    row_sums: Float64Array | None,
) -> sparse.csr_matrix:
    """Write owned sparse log corrections from bounded float64 row blocks.

    A single CSR row may exceed the stored-entry target. Optional row sums
    use float64 values before they are cast to the output storage dtype.
    """
    data = np.empty(X.nnz, dtype=dtype)
    for row_start, row_stop, start, stop in _support_row_blocks(
        X, _LOG_TRANSFORM_BLOCK_NNZ
    ):
        values = X.data[start:stop].astype(np.float64, copy=True)
        if by_row:
            denominator = np.repeat(
                divisors[row_start:row_stop],
                np.diff(X.indptr[row_start : row_stop + 1]),
            )
        elif np.ndim(divisors) == 0:
            denominator = divisors
        else:
            denominator = np.asarray(divisors)[X.indices[start:stop]]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(values, denominator, out=values)
        if not np.isfinite(values).all():
            raise ValueError(overflow_message)
        np.log1p(values, out=values)
        if row_sums is not None:
            indptr = X.indptr[row_start : row_stop + 1] - start
            nonempty = np.diff(indptr) > 0
            row_sums[row_start:row_stop][nonempty] = np.add.reduceat(
                values, indptr[:-1][nonempty]
            )
        data[start:stop] = values
    index_dtype = (
        np.int64 if max(X.shape + (X.nnz,)) > np.iinfo(np.int32).max else np.int32
    )
    return sparse.csr_matrix(
        (
            data,
            X.indices.astype(index_dtype, copy=True),
            X.indptr.astype(index_dtype, copy=True),
        ),
        shape=X.shape,
        copy=False,
    )


def log1p_count_correction(
    X: sparse.csr_matrix,
    prior_counts: Float64Array | float,
    *,
    dtype: DTypeLike = "float64",
    row_sums: Float64Array | None = None,
) -> sparse.csr_matrix:
    """Return sparse ``log1p(x_ij / prior_count_j)`` corrections."""
    return _log1p_sparse_correction(
        X,
        prior_counts,
        by_row=False,
        dtype=dtype,
        overflow_message="Count-to-prior ratios overflow float64",
        row_sums=row_sums,
    )


def log1p_row_scaled_counts(
    X: sparse.csr_matrix,
    row_divisors: Float64Array,
    *,
    overflow_message: str,
    dtype: DTypeLike = "float64",
    row_sums: Float64Array | None = None,
) -> sparse.csr_matrix:
    """Return sparse ``log1p(x_ij / row_divisor_i)`` values."""
    if not np.isfinite(row_divisors).all() or (row_divisors <= 0).any():
        raise ValueError(overflow_message)
    return _log1p_sparse_correction(
        X,
        row_divisors,
        by_row=True,
        dtype=dtype,
        overflow_message=overflow_message,
        row_sums=row_sums,
    )


def build_log1p_norm_representation(
    X: sparse.csr_matrix,
    *,
    size_factors: ArrayLike,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Represent size-factor-normalized ``log1p`` values as rank-zero sparse."""
    divisors = validate_size_factors(size_factors, X.shape[0])
    sparse_part = log1p_row_scaled_counts(
        X,
        divisors,
        overflow_message="Count-to-size-factor ratios overflow float64",
        dtype=dtype,
    )
    return SparseLowRankMatrix(
        sparse_part,
        np.zeros((X.shape[0], 0), dtype=np.float64),
        np.zeros((X.shape[1], 0), dtype=np.float64),
    )


def build_shifted_clr_representation(
    X: sparse.csr_matrix,
    *,
    count_shift: float,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Represent count-scale shifted CLR as sparse plus rank one."""
    count_shift = validate_positive_scalar(count_shift, name="count_shift")
    row_sums = np.zeros(X.shape[0], dtype=np.float64)
    sparse_part = log1p_count_correction(X, count_shift, dtype=dtype, row_sums=row_sums)
    row_mean = row_sums / X.shape[1]
    return SparseLowRankMatrix(
        sparse_part,
        -row_mean,
        np.ones(X.shape[1], dtype=np.float64),
    )


def build_proportion_shifted_clr_representation(
    X: sparse.csr_matrix,
    *,
    composition_shift: float,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Represent composition-scale shifted CLR as sparse plus rank one."""
    composition_shift = validate_positive_scalar(
        composition_shift, name="composition_shift"
    )
    row_totals = _sum_counts(X, axis=1)
    if not np.isfinite(row_totals).all():
        raise ValueError("Cell totals overflow float64")
    if (row_totals == 0).any():
        raise ValueError(
            "Cells with zero total counts are not supported; filter empty rows "
            "out of the count matrix first"
        )
    with np.errstate(over="ignore", under="ignore"):
        row_divisors = row_totals * composition_shift
    row_sums = np.zeros(X.shape[0], dtype=np.float64)
    sparse_part = log1p_row_scaled_counts(
        X,
        row_divisors,
        overflow_message="Count-to-composition ratios overflow float64",
        dtype=dtype,
        row_sums=row_sums,
    )
    row_mean = row_sums / X.shape[1]
    return SparseLowRankMatrix(
        sparse_part,
        -row_mean,
        np.ones(X.shape[1], dtype=np.float64),
    )


def build_dirichlet_log_representation(
    X: sparse.csr_matrix,
    *,
    concentration: float,
    prior_proportions: PriorProportions = None,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Represent log Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    if not np.isfinite(prior_counts).all() or (prior_counts == 0).any():
        raise ValueError("Dirichlet prior counts must be representable in float64")
    sparse_part = log1p_count_correction(X, prior_counts, dtype=dtype)
    row_totals = _sum_counts(X, axis=1)
    if not np.isfinite(row_totals).all():
        raise ValueError("Cell totals overflow float64")
    with np.errstate(over="ignore"):
        posterior_totals = row_totals + concentration
    if not np.isfinite(posterior_totals).all():
        raise ValueError("Dirichlet posterior totals overflow float64")
    left = np.column_stack(
        (
            np.ones(X.shape[0], dtype=np.float64),
            -np.log(posterior_totals),
        )
    )
    right = np.column_stack(
        (
            np.log(prior_counts),
            np.ones(X.shape[1], dtype=np.float64),
        )
    )
    return SparseLowRankMatrix(sparse_part, left, right)


def build_dirichlet_clr_representation(
    X: sparse.csr_matrix,
    *,
    concentration: float,
    prior_proportions: PriorProportions = None,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Represent CLR of Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    if not np.isfinite(prior_counts).all() or (prior_counts == 0).any():
        raise ValueError("Dirichlet prior counts must be representable in float64")
    row_sums = np.zeros(X.shape[0], dtype=np.float64)
    sparse_part = log1p_count_correction(
        X, prior_counts, dtype=dtype, row_sums=row_sums
    )
    sparse_row_mean = row_sums / X.shape[1]
    log_prior_counts = np.log(prior_counts)
    centered_log_prior = log_prior_counts - np.mean(log_prior_counts)
    left = np.column_stack(
        (
            -sparse_row_mean,
            np.ones(X.shape[0], dtype=np.float64),
        )
    )
    right = np.column_stack(
        (
            np.ones(X.shape[1], dtype=np.float64),
            centered_log_prior,
        )
    )
    return SparseLowRankMatrix(sparse_part, left, right)
