"""Tests for exact clipping and sparse-support growth limits."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca._clip import _clipped_zero_locations, apply_clipping


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
    # Row-major order is part of the contract.
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


@pytest.mark.parametrize("seed", range(25))
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


def test_no_locations_returns_empty(small):
    """A threshold with no clipped zeros returns empty locations."""
    # A huge threshold means nothing crosses; returns empty arrays, not None.
    u = np.full(small.shape[0], -1.0)
    v = np.ones(small.shape[1])
    rows, cols = _clipped_zero_locations(small, u, v, threshold=1e9)
    assert rows.size == 0 and cols.size == 0


def test_max_count_bail_returns_none(small):
    """The location search stops when its maximum count is exceeded."""
    # Drive u, v so that essentially every structural zero crosses the
    # threshold, then ask for a count bound that is provably exceeded.
    u = np.full(small.shape[0], -5.0)
    v = np.full(small.shape[1], 5.0)
    assert _clipped_zero_locations(small, u, v, 0.1, max_count=0) is None


def test_max_count_loose_bound_materializes(small):
    """A loose maximum count permits location materialization."""
    # A bound at least as large as the true count must not trigger the bail.
    u = np.full(small.shape[0], -5.0)
    v = np.full(small.shape[1], 5.0)
    exp_rows, _ = _brute_force_locations(small, u, v, 0.1)
    result = _clipped_zero_locations(small, u, v, 0.1, max_count=exp_rows.size)
    assert result is not None
    assert result[0].size == exp_rows.size


def test_bail_lower_bound_is_sound(small):
    """The early-exit lower bound never misses an allowed result."""
    # The bail uses total - X.nnz as a lower bound on the true count. It must
    # never fire when the true count is within the bound, even though many
    # stored entries also fall inside the candidate set.
    u = np.full(small.shape[0], -5.0)
    v = np.full(small.shape[1], 5.0)
    true_count = _brute_force_locations(small, u, v, 0.1)[0].size
    result = _clipped_zero_locations(small, u, v, 0.1, max_count=true_count)
    assert result is not None
    assert result[0].size == true_count


def test_unsorted_indices_handled():
    """Clipping handles CSR matrices with unsorted column indices."""
    # Construct a CSR matrix whose within-row column indices are not sorted;
    # the membership test must still exclude stored entries correctly.
    X = sparse.csr_matrix(
        (
            np.array([1, 1, 1], dtype=np.int64),
            np.array([3, 0, 2]),  # row 0 columns out of order
            np.array([0, 3]),
        ),
        shape=(1, 4),
    )
    assert not X.has_sorted_indices
    u = np.array([-1.0])
    v = np.array([2.0, 2.0, 2.0, 2.0])
    _assert_matches_oracle(X, u, v, threshold=0.5)


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
    rows = np.array([0], dtype=np.intp)

    S = apply_clipping(
        X,
        residual_nonzero=np.array([0.0]),
        u=u,
        v=np.array([0.0, equality_v]),
        rows=rows,
        clip=clip,
        clip_mode="symmetric",
        clip_max_nnz_ratio=1.0,
    )
    assert S.nnz == 0


def test_apply_clipping_reuses_precomputed_row_support(monkeypatch):
    """Clipping reuses the caller's precomputed sparse row support."""
    X = sparse.csr_matrix([[1, 0], [0, 2]])
    rows = np.array([0, 1], dtype=np.intp)

    def unexpected_repeat(*args, **kwargs):
        pytest.fail("apply_clipping rebuilt the CSR row-support vector")

    monkeypatch.setattr("sparse_count_pca._clip.np.repeat", unexpected_repeat)
    S = apply_clipping(
        X,
        residual_nonzero=np.array([0.2, 0.3]),
        u=np.array([-1.0, -1.0]),
        v=np.array([0.5, 0.5]),
        rows=rows,
        clip=None,
        clip_mode="symmetric",
        clip_max_nnz_ratio=2.0,
    )
    assert S.shape == X.shape
