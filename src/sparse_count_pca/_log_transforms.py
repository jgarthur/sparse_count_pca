"""Sparse-plus-low-rank builders for shifted logarithmic transforms."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from ._counts import _sum_counts
from ._representation import SparseLowRankMatrix

PriorProportions: TypeAlias = ArrayLike | None
Float64Array = NDArray[np.float64]
PRIOR_SUM_RTOL = 1e-8
PRIOR_SUM_ATOL = 1e-12


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


def log1p_count_correction(
    X: sparse.csr_matrix,
    prior_counts: Float64Array | float,
) -> sparse.csr_matrix:
    """Return sparse ``log1p(x_ij / prior_count_j)`` corrections."""
    data = X.data.astype(np.float64, copy=True)
    denominator = (
        prior_counts
        if np.ndim(prior_counts) == 0
        else np.asarray(prior_counts)[X.indices]
    )
    with np.errstate(over="ignore"):
        np.divide(data, denominator, out=data)
    if not np.isfinite(data).all():
        raise ValueError("Count-to-prior ratios overflow float64")
    np.log1p(data, out=data)
    return sparse.csr_matrix(
        (data, X.indices.copy(), X.indptr.copy()), shape=X.shape, copy=False
    )


def _rank_zero(
    sparse_part: sparse.csr_matrix,
) -> SparseLowRankMatrix:
    return SparseLowRankMatrix(
        sparse_part,
        np.empty((sparse_part.shape[0], 0), dtype=np.float64),
        np.empty((sparse_part.shape[1], 0), dtype=np.float64),
    )


def build_shifted_log_representation(
    X: sparse.csr_matrix,
    *,
    count_shift: float,
) -> SparseLowRankMatrix:
    """Represent ``log1p(X / count_shift)`` as an exactly sparse matrix."""
    count_shift = validate_positive_scalar(count_shift, name="count_shift")
    return _rank_zero(log1p_count_correction(X, count_shift))


def build_shifted_clr_representation(
    X: sparse.csr_matrix,
    *,
    count_shift: float,
) -> SparseLowRankMatrix:
    """Represent count-scale shifted CLR as sparse plus rank one."""
    count_shift = validate_positive_scalar(count_shift, name="count_shift")
    sparse_part = log1p_count_correction(X, count_shift)
    row_mean = np.asarray(sparse_part.sum(axis=1)).ravel() / X.shape[1]
    return SparseLowRankMatrix(
        sparse_part,
        -row_mean,
        np.ones(X.shape[1], dtype=np.float64),
    )


def build_proportion_shifted_clr_representation(
    X: sparse.csr_matrix,
    *,
    composition_shift: float,
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
    rows = np.repeat(np.arange(X.shape[0], dtype=np.intp), np.diff(X.indptr))
    data = X.data.astype(np.float64, copy=True)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(
            data,
            row_totals[rows] * composition_shift,
            out=data,
        )
    if not np.isfinite(data).all():
        raise ValueError("Count-to-composition ratios overflow float64")
    np.log1p(data, out=data)
    sparse_part = sparse.csr_matrix(
        (data, X.indices.copy(), X.indptr.copy()), shape=X.shape, copy=False
    )
    row_mean = np.asarray(sparse_part.sum(axis=1)).ravel() / X.shape[1]
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
) -> SparseLowRankMatrix:
    """Represent log Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    if not np.isfinite(prior_counts).all() or (prior_counts == 0).any():
        raise ValueError("Dirichlet prior counts must be representable in float64")
    sparse_part = log1p_count_correction(X, prior_counts)
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
) -> SparseLowRankMatrix:
    """Represent CLR of Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    if not np.isfinite(prior_counts).all() or (prior_counts == 0).any():
        raise ValueError("Dirichlet prior counts must be representable in float64")
    sparse_part = log1p_count_correction(X, prior_counts)
    sparse_row_mean = np.asarray(sparse_part.sum(axis=1)).ravel() / X.shape[1]
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
