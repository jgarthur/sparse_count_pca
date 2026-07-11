from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

ClipMode: TypeAlias = Literal["symmetric", "upper"]
Float64Array: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]

CLIP_MODES: set[ClipMode] = {"symmetric", "upper"}


def validate_clip(
    clip: float | None,
    clip_mode: ClipMode,
    clip_max_nnz_ratio: float | None,
) -> None:
    """Validate clipping options.

    Args:
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.

    Raises:
        ValueError: If a clipping option is outside its supported range.
        NotImplementedError: If ``clip`` is a string alias.
    """
    if isinstance(clip, str):
        raise NotImplementedError(
            "String clip aliases are not supported in v1; pass a finite positive float."
        )
    if clip is not None and (not np.isfinite(clip) or clip <= 0):
        raise ValueError("clip must be finite and positive or None")
    if clip_mode not in CLIP_MODES:
        allowed = ", ".join(sorted(CLIP_MODES))
        raise ValueError(f"clip_mode must be one of: {allowed}")
    if clip_max_nnz_ratio is not None and (
        not np.isfinite(clip_max_nnz_ratio) or clip_max_nnz_ratio < 1
    ):
        raise ValueError("clip_max_nnz_ratio must be finite and at least 1 or None")


def _clipped_zero_locations(
    X: sparse.csr_matrix,
    u: Float64Array,
    v: Float64Array,
    threshold: float,
    *,
    max_count: int | None = None,
    support_rows: IndexArray | None = None,
) -> tuple[IndexArray, IndexArray] | None:
    """Find structural zeros (i, j) of X where ``u[i] * v[j] < -threshold``.

    Args:
        X: CSR count matrix defining the nonzero support.
        u: Rank-one row factor.
        v: Rank-one column factor.
        threshold: Positive clipping threshold.
        max_count: If given and the number of locations provably exceeds it,
            return ``None`` without materializing the index arrays.
        support_rows: Precomputed row index aligned with ``X.data``.

    Returns:
        Flat row and column index arrays in row-major order, or ``None`` if
        the ``max_count`` bound was exceeded.
    """
    m, n = X.shape
    u64 = np.asarray(u, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)

    order = np.argsort(v64)
    vs = v64[order]

    # Per row, candidates form a contiguous range [start, stop) in sorted-v
    # space, found by binary search instead of an O(n) scan.
    start = np.zeros(m, dtype=np.intp)
    stop = np.zeros(m, dtype=np.intp)
    pos = u64 > 0
    neg = u64 < 0
    # u_i > 0:  v_j < -threshold / u_i  (prefix of vs)
    stop[pos] = np.searchsorted(vs, -threshold / u64[pos], side="left")
    # u_i < 0:  v_j > -threshold / u_i  (suffix of vs)
    start[neg] = np.searchsorted(vs, -threshold / u64[neg], side="right")
    stop[neg] = n
    # u_i == 0: empty range (threshold > 0)

    counts = stop - start
    total = int(counts.sum())
    empty = np.empty(0, dtype=np.intp)
    if total == 0:
        return empty, empty
    # Materialize all candidate (row, col) pairs in one shot.
    rows = np.repeat(np.arange(m, dtype=np.intp), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    flat = np.arange(total, dtype=np.intp) - offsets + np.repeat(start, counts)
    cols = order[flat]

    # Binary search only generates candidates: division can round across the
    # strict boundary, so enforce the requested product predicate directly.
    keep = u64[rows] * v64[cols] < -threshold
    rows, cols = rows[keep], cols[keep]

    # Drop candidates that lie on the sparse support, via one global sorted
    # membership test on linear indices.
    if X.nnz:
        if not X.has_sorted_indices:
            X.sort_indices()
        lin = rows * n + cols
        if support_rows is None:
            support_rows = np.repeat(
                np.arange(m, dtype=np.intp), np.diff(X.indptr)
            )
        x_lin = support_rows * n + X.indices
        p = np.searchsorted(x_lin, lin).clip(max=x_lin.size - 1)
        keep = x_lin[p] != lin
        rows, cols = rows[keep], cols[keep]
    if max_count is not None and rows.size > max_count:
        return None
    return rows, cols


def apply_clipping(
    X: sparse.csr_matrix,
    residual_nonzero: Float64Array,
    u: Float64Array,
    v: Float64Array,
    rows: IndexArray,
    *,
    clip: float | None,
    clip_mode: ClipMode,
    clip_max_nnz_ratio: float | None,
) -> sparse.csr_matrix:
    """Construct the sparse residual correction after clipping.

    Args:
        X: CSR count matrix defining the original sparse support.
        residual_nonzero: Residual values aligned with ``X.data``.
        u: Rank-one row factor.
        v: Rank-one column factor.
        rows: Precomputed row index aligned with ``X.data``.
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.

    Returns:
        A CSR matrix ``S`` such that ``S + u v.T`` equals the requested
        clipped residual matrix.

    Raises:
        RuntimeError: If exact symmetric clipping would meet or exceed the
            configured sparse support-growth limit.
    """
    uv_nonzero = u[rows] * v[X.indices]
    if clip is None:
        data = residual_nonzero - uv_nonzero
        return sparse.csr_matrix(
            (data, X.indices.copy(), X.indptr.copy()), shape=X.shape
        )
    if clip_mode == "upper":
        data = np.minimum(residual_nonzero, clip) - uv_nonzero
        S = sparse.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
        S.eliminate_zeros()
        return S

    data = np.clip(residual_nonzero, -clip, clip) - uv_nonzero
    # Bound clipped-zero growth before materializing a potentially huge set.
    max_count = None
    if clip_max_nnz_ratio is not None:
        max_possible_ratio = (X.shape[0] * X.shape[1]) / X.nnz
        if clip_max_nnz_ratio <= max_possible_ratio:
            max_count = int(np.ceil(clip_max_nnz_ratio * X.nnz)) - X.nnz

    locations = _clipped_zero_locations(
        X,
        u,
        v,
        clip,
        max_count=max_count,
        support_rows=rows,
    )
    if locations is None:
        raise RuntimeError(
            "Exact symmetric clipping would increase sparse support by a factor "
            f"meeting or exceeding clip_max_nnz_ratio={clip_max_nnz_ratio}"
        )
    correction_rows, correction_cols = locations
    count = correction_rows.size
    if count == 0:
        S = sparse.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
        S.eliminate_zeros()
        return S
    growth_ratio = (X.nnz + count) / X.nnz
    if clip_max_nnz_ratio is not None and growth_ratio >= clip_max_nnz_ratio:
        raise RuntimeError(
            "Exact symmetric clipping would increase sparse support by a factor of "
            f"{growth_ratio:.3g}, meeting or exceeding "
            f"clip_max_nnz_ratio={clip_max_nnz_ratio}"
        )

    # At clipped zeros the residual is u_i v_j < -clip, so the clipped value
    # is exactly -clip and the correction is -clip - u_i v_j.
    correction_data = -clip - u[correction_rows] * v[correction_cols]

    # The correction support is disjoint from X's support, so build the
    # result in one COO -> CSR conversion instead of a sparse add.
    S = sparse.coo_matrix(
        (
            np.concatenate([data, correction_data]),
            (
                np.concatenate([rows, correction_rows]),
                np.concatenate([X.indices.astype(np.intp), correction_cols]),
            ),
        ),
        shape=X.shape,
    ).tocsr()
    S.eliminate_zeros()
    return S
