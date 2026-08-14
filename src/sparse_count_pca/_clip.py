"""Exact clipping operations for sparse-plus-low-rank representations."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

ClipMode: TypeAlias = Literal["symmetric", "upper"]
Float64Array: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]
CSRMatrix: TypeAlias = sparse.csr_matrix | sparse.csr_array

CLIP_MODES: set[ClipMode] = {"symmetric", "upper"}
_CANDIDATE_CHUNK_SIZE = 65_536


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
    X: CSRMatrix,
    u: Float64Array,
    v: Float64Array,
    threshold: float,
    *,
    max_count: int | None = None,
) -> tuple[IndexArray, IndexArray] | None:
    """Find structural zeros (i, j) of X where ``u[i] * v[j] < -threshold``.

    Args:
        X: Canonical CSR count matrix defining the nonzero support.
        u: Rank-one row factor.
        v: Rank-one column factor.
        threshold: Positive clipping threshold.
        max_count: If given and the number of locations exceeds it, return
            ``None`` before allocating the output arrays.

    Returns:
        Flat row and column index arrays grouped by row, or ``None`` if the
        ``max_count`` bound was exceeded. Column order within rows is
        unspecified.
    """
    assert sparse.issparse(X) and X.format == "csr", "X must use CSR format"
    assert X.has_canonical_format, "X must be in canonical CSR format"
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
    # Widen each rounded quotient by one representable value. The direct
    # product filter below removes the harmless extras; without widening, a
    # product that rounds below the strict threshold can be missed entirely.
    # u_i > 0:  v_j < -threshold / u_i  (prefix of vs)
    cut_pos = np.nextafter(-threshold / u64[pos], np.inf)
    stop[pos] = np.searchsorted(vs, cut_pos, side="right")
    # u_i < 0:  v_j > -threshold / u_i  (suffix of vs)
    cut_neg = np.nextafter(-threshold / u64[neg], -np.inf)
    start[neg] = np.searchsorted(vs, cut_neg, side="left")
    stop[neg] = n
    # u_i == 0: empty range (threshold > 0)

    empty = np.empty(0, dtype=np.intp)
    if not (stop > start).any():
        return empty, empty

    def surviving_column_chunks():
        for row in range(m):
            support = X.indices[X.indptr[row] : X.indptr[row + 1]]
            for chunk_start in range(start[row], stop[row], _CANDIDATE_CHUNK_SIZE):
                chunk_stop = min(
                    chunk_start + _CANDIDATE_CHUNK_SIZE,
                    stop[row],
                )
                columns = order[chunk_start:chunk_stop]
                # Division only identifies a widened candidate interval. Apply
                # the exact floating-point product predicate at the boundary.
                columns = columns[u64[row] * v64[columns] < -threshold]
                if support.size and columns.size:
                    positions = np.searchsorted(support, columns)
                    on_support = positions < support.size
                    on_support[on_support] = (
                        support[positions[on_support]] == columns[on_support]
                    )
                    columns = columns[~on_support]
                if columns.size:
                    yield row, columns

    # Count first so a finite support-growth guard is enforced before output
    # allocation. The second pass exchanges a little computation for bounded
    # temporary memory even when candidate support is nearly dense.
    count = 0
    for _, columns in surviving_column_chunks():
        count += columns.size
        if max_count is not None and count > max_count:
            return None
    if count == 0:
        return empty, empty

    rows = np.empty(count, dtype=np.intp)
    cols = np.empty(count, dtype=np.intp)
    offset = 0
    for row, columns in surviving_column_chunks():
        next_offset = offset + columns.size
        rows[offset:next_offset] = row
        cols[offset:next_offset] = columns
        offset = next_offset
    return rows, cols


def _guarded_clipped_zero_locations(
    X: CSRMatrix,
    u: Float64Array,
    v: Float64Array,
    threshold: float,
    *,
    clip_max_nnz_ratio: float | None,
) -> tuple[IndexArray, IndexArray]:
    """Find clipped structural zeros while enforcing the support-growth limit."""
    max_count = None
    if clip_max_nnz_ratio is not None:
        max_possible_ratio = (X.shape[0] * X.shape[1]) / X.nnz
        if clip_max_nnz_ratio <= max_possible_ratio:
            max_count = max(
                0,
                int(np.ceil(clip_max_nnz_ratio * X.nnz)) - X.nnz - 1,
            )

    locations = _clipped_zero_locations(
        X,
        u,
        v,
        threshold,
        max_count=max_count,
    )
    if locations is None:
        raise RuntimeError(
            "Exact symmetric clipping would increase sparse support by a factor "
            f"meeting or exceeding clip_max_nnz_ratio={clip_max_nnz_ratio}. "
            "Raise clip_max_nnz_ratio or switch to clip_mode='upper', which "
            "leaves zero-count residuals unchanged and cannot expand support."
        )
    correction_rows, correction_cols = locations
    if correction_rows.size == 0:
        return correction_rows, correction_cols
    growth_ratio = (X.nnz + correction_rows.size) / X.nnz
    if clip_max_nnz_ratio is not None and growth_ratio >= clip_max_nnz_ratio:
        raise RuntimeError(
            "Exact symmetric clipping would increase sparse support by a factor of "
            f"{growth_ratio:.3g}, meeting or exceeding "
            f"clip_max_nnz_ratio={clip_max_nnz_ratio}. "
            "Raise clip_max_nnz_ratio or switch to clip_mode='upper', which "
            "leaves zero-count residuals unchanged and cannot expand support."
        )
    return correction_rows, correction_cols
