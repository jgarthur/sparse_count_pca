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


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    ("clip", "clip_mode"),
    [(None, "symmetric"), (0.5, "symmetric"), (0.5, "upper")],
)
def test_apply_clipping_matches_dense_oracle(seed, clip, clip_mode):
    """Sparse corrections reconstruct the requested clipped residual matrix."""
    rng = np.random.default_rng(seed)
    m, n = rng.integers(1, 10, size=2)
    dense = rng.integers(0, 3, size=(m, n))
    if not dense.any():
        dense[0, 0] = 1
    X = sparse.csr_matrix(dense)
    rows = np.repeat(np.arange(m, dtype=np.intp), np.diff(X.indptr))
    residual_nonzero = rng.normal(size=X.nnz)
    # These signs are the residual-builder domain invariant: every structural-
    # zero residual supplied by u v.T is negative.
    u = -rng.uniform(0.1, 3.0, size=m)
    v = rng.uniform(0.1, 3.0, size=n)

    full = np.outer(u, v)
    full[rows, X.indices] = residual_nonzero
    if clip is None:
        expected = full
    elif clip_mode == "upper":
        expected = np.minimum(full, clip)
    else:
        expected = np.clip(full, -clip, clip)

    S = apply_clipping(
        X,
        residual_nonzero,
        u,
        v,
        rows,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=None,
    )
    np.testing.assert_allclose(
        S.toarray() + np.outer(u, v), expected, rtol=0.0, atol=1e-14
    )


def test_rounded_division_does_not_miss_crossing_structural_zero():
    """A quotient-rounding boundary still includes a truly clipped zero."""
    # This is a valid Poisson-Pearson factorization for the count matrix below:
    # u = -sqrt(row totals), v = sqrt(column proportions). The clip is one ULP
    # below the represented zero residual at (0, 0).
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
    with pytest.raises(AssertionError, match="canonical CSR"):
        apply_clipping(
            X,
            residual_nonzero=np.ones(X.nnz),
            u=u,
            v=v,
            rows=np.zeros(X.nnz, dtype=np.intp),
            clip=None,
            clip_mode="symmetric",
            clip_max_nnz_ratio=None,
        )


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
    import sparse_count_pca._clip as clip_module

    X = sparse.csr_matrix([[1, 0], [0, 2]])
    rows = np.array([0, 1], dtype=np.intp)
    seen = {}
    original = clip_module._clipped_zero_locations

    def spy(*args, **kwargs):
        seen["support_rows"] = kwargs.get("support_rows")
        return original(*args, **kwargs)

    monkeypatch.setattr(clip_module, "_clipped_zero_locations", spy)
    apply_clipping(
        X,
        residual_nonzero=np.array([0.2, 0.3]),
        u=np.array([-1.0, -1.0]),
        v=np.array([0.5, 0.5]),
        rows=rows,
        clip=10.0,
        clip_mode="symmetric",
        clip_max_nnz_ratio=None,
    )
    assert seen["support_rows"] is rows
