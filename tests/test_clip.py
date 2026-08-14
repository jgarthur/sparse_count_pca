"""Tests for exact clipping and sparse-support growth limits."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import _clip as clip_module
from sparse_count_pca._clip import (
    _CANDIDATE_CHUNK_SIZE,
    _clipped_zero_locations,
    _guarded_clipped_zero_locations,
)


def _brute_force_locations(X, u, v, threshold):
    """Dense O(m*n) oracle: structural zeros (i, j) with u_i v_j < -threshold."""
    dense_support = X.toarray() != 0
    uv = np.outer(u, v)
    mask = (uv < -threshold) & ~dense_support
    rows, cols = np.nonzero(mask)
    return rows.astype(np.intp), cols.astype(np.intp)


def _as_set(rows, cols):
    return set(zip(rows.tolist(), cols.tolist()))


def _assert_matches_oracle(X, u, v, threshold):
    result = _clipped_zero_locations(X, u, v, threshold)
    assert result is not None
    rows, cols = result
    exp_rows, exp_cols = _brute_force_locations(X, u, v, threshold)
    # Column order within each row is unspecified, so compare row-major views.
    order = np.lexsort((cols, rows))
    np.testing.assert_array_equal(rows[order], exp_rows)
    np.testing.assert_array_equal(cols[order], exp_cols)
    # Returned locations never lie on the stored support.
    assert _as_set(rows, cols).isdisjoint(zip(*X.nonzero()))


@pytest.fixture
def small(counts):
    return counts


def test_matches_dense_oracle_mixed_signs(small):
    """Clipped zero locations match a dense mixed-sign oracle."""
    # u with both signs exercises the prefix (u>0) and suffix (u<0) branches,
    # and a zero entry exercises the empty-range branch.
    u = np.array([-1.0, 0.8, -0.5, 0.0, 1.2, -0.9])
    v = np.array([0.4, -0.6, 0.9, -0.3])
    _assert_matches_oracle(small, u, v, threshold=0.2)


@pytest.mark.parametrize("seed", range(8))
def test_matches_dense_oracle_random(seed):
    """Clipped zero locations match a dense oracle on random inputs."""
    rng = np.random.default_rng(seed)
    m, n = rng.integers(1, 12, size=2)
    dense = rng.integers(0, 3, size=(m, n))
    X = sparse.csr_matrix(dense.astype(np.int64))
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    threshold = float(rng.uniform(0.05, 1.5))
    _assert_matches_oracle(X, u, v, threshold)


def test_threshold_one_ulp_below_a_zero_residual_still_clips():
    """A threshold one ULP below a structural-zero residual still includes it."""
    # u and v are a valid Poisson-Pearson factorization of the matrix below:
    # u = -sqrt(row totals), v = sqrt(column proportions). Putting the threshold
    # one ULP inside the residual at (0, 0) pins the strict product predicate at
    # the tightest margin float64 can express.
    X = sparse.csr_matrix([[0, 2], [5, 4]])
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    proportions = np.asarray(X.sum(axis=0)).ravel() / X.sum()
    u = -np.sqrt(totals)
    v = np.sqrt(proportions)
    threshold = np.nextafter(-(u[0] * v[0]), -np.inf)
    assert u[0] * v[0] < -threshold

    result = _clipped_zero_locations(X, u, v, threshold)
    assert result is not None
    np.testing.assert_array_equal(result[0], np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(result[1], np.array([0], dtype=np.intp))


def test_no_locations_returns_empty(small):
    """A threshold with no clipped zeros returns empty locations."""
    # A huge threshold means nothing crosses; returns empty arrays, not None.
    u = np.full(small.shape[0], -1.0)
    v = np.ones(small.shape[1])
    rows, cols = _clipped_zero_locations(small, u, v, threshold=1e9)
    assert rows.size == 0 and cols.size == 0


@pytest.mark.parametrize(
    ("offset", "materializes"),
    [(-1, False), (0, True), (5, True)],
    ids=["below-true-count", "at-true-count", "above-true-count"],
)
def test_max_count_guard_boundary(small, offset, materializes):
    """The search bails below its count bound and materializes at or above it."""
    # Drive u, v so that essentially every structural zero crosses the threshold.
    u = np.full(small.shape[0], -5.0)
    v = np.full(small.shape[1], 5.0)
    true_count = _brute_force_locations(small, u, v, 0.1)[0].size
    assert true_count > 1

    result = _clipped_zero_locations(small, u, v, 0.1, max_count=true_count + offset)

    if materializes:
        assert result is not None
        assert result[0].size == true_count
    else:
        assert result is None


def test_noncanonical_csr_is_rejected():
    """Clipping rejects CSR input that bypassed count canonicalization."""
    X = sparse.csr_matrix(
        (
            np.array([1, 1, 1], dtype=np.int64),
            np.array([3, 0, 2]),  # row 0 columns out of order
            np.array([0, 3]),
        ),
        shape=(1, 4),
    )
    assert not X.has_canonical_format
    u = np.array([-1.0])
    v = np.array([2.0, 2.0, 2.0, 2.0])
    with pytest.raises(AssertionError, match="canonical CSR"):
        _clipped_zero_locations(X, u, v, threshold=0.5)


def test_canonical_csr_array_is_accepted():
    """Canonical SciPy CSR arrays pass the internal format precondition."""
    X = sparse.csr_array([[1, 0]])
    result = _clipped_zero_locations(
        X, np.array([-1.0]), np.array([1.0, 2.0]), threshold=0.5
    )
    assert result is not None
    np.testing.assert_array_equal(result[0], np.array([0], dtype=np.intp))
    np.testing.assert_array_equal(result[1], np.array([1], dtype=np.intp))


def test_empty_support_matrix():
    """Clipping handles matrices with empty sparse support."""
    X = sparse.csr_matrix((3, 4), dtype=np.int64)
    u = np.array([-1.0, 0.0, 2.0])
    v = np.array([1.0, -1.0, 0.5, 2.0])
    _assert_matches_oracle(X, u, v, threshold=0.3)


def test_exact_clip_equality_does_not_grow_support_or_trigger_guard():
    """Values equal to the clip bound do not grow sparse support."""
    u = np.array([-19605.17544401196])
    equality_v = 2.1931007107567585e-07
    clip = 0.004299612420077357
    assert u[0] * equality_v == -clip
    X = sparse.csr_matrix([[1, 0]])
    rows, columns = _guarded_clipped_zero_locations(
        X,
        u,
        np.array([0.0, equality_v]),
        clip,
        clip_max_nnz_ratio=1.0,
    )
    assert rows.size == 0 and columns.size == 0


def test_growth_guard_stops_nearly_dense_search_before_output_allocation(monkeypatch):
    """A finite guard rejects near-dense candidates without dense-size output."""
    size = 2_500
    X = sparse.csr_matrix(([1], ([0], [0])), shape=(size, size))
    # Every structural zero crosses, so counting after materializing would build
    # two index arrays this long.
    unguarded_count = size * size - X.nnz
    allocations = []
    original_empty = np.empty

    def recording_empty(shape, *args, **kwargs):
        allocations.append(int(np.prod(shape)))
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(clip_module.np, "empty", recording_empty)

    result = _clipped_zero_locations(
        X,
        u=np.full(size, -1.0),
        v=np.ones(size),
        threshold=0.5,
        max_count=0,
    )

    assert result is None
    assert allocations, "the search allocates at least its empty sentinel"
    assert max(allocations) <= _CANDIDATE_CHUNK_SIZE
    assert max(allocations) < unguarded_count
