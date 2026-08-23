"""Tests for sparse-plus-low-rank matrix and operator behavior."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import _operator as operator_module
from sparse_count_pca._operator import (
    SparseLowRankLinearOperator,
    _squared_norm_is_numerically_zero,
)
from sparse_count_pca._representation import SparseLowRankMatrix
from sparse_count_pca._sparse import _support_row_blocks

_STATS_RTOL = {"float32": 2e-6, "float64": 1e-12}


def _set_stats_block_nnz(monkeypatch, target_nnz):
    """Point both statistics sweeps at the given row-block storage target."""
    monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", target_nnz)
    monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", target_nnz)


def _assert_statistics_close(operator, *, mean, uncentered, centered, tolerance):
    """Assert the operator's summary statistics match the expected values."""
    np.testing.assert_allclose(operator.mean, mean, rtol=tolerance, atol=0.0)
    assert operator.frobenius_squared_uncentered() == pytest.approx(
        uncentered,
        rel=tolerance,
        abs=0.0,
    )
    assert operator.frobenius_squared_centered() == pytest.approx(
        centered,
        rel=tolerance,
        abs=0.0,
    )


def _statistics_fixture():
    """Build a representation with empty, sparse, boundary, and dense columns."""
    sparse_part = sparse.csr_matrix(
        np.array(
            [
                [0.0, 8.0, -4.0, 2.0, 1.0],
                [0.0, 0.0, 0.0, -3.0, 2.0],
                [0.0, -7.0, 5.0, 4.0, -1.0],
                [0.0, 0.0, -6.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 5.0, -2.0],
                [0.0, 0.0, 0.0, -6.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, -5.0],
            ]
        )
    )
    left = np.array(
        [
            [1024.0, 0.5],
            [1024.25, -0.75],
            [1023.5, 1.25],
            [1025.0, -1.5],
            [1023.75, 0.25],
            [1024.5, 1.0],
            [1023.0, -0.25],
        ]
    )
    right = np.array(
        [
            [1.0, -0.5],
            [-0.75, 1.5],
            [0.5, 0.25],
            [-1.25, -0.75],
            [0.25, 2.0],
        ]
    )
    return SparseLowRankMatrix(sparse_part, left, right)


@pytest.mark.parametrize("target_nnz", [0, -1])
def test_support_row_blocks_rejects_nonpositive_target(target_nnz):
    """Row-block planning rejects zero and negative storage targets."""
    matrix = sparse.csr_matrix([[1.0]])

    with pytest.raises(ValueError, match="target_nnz must be positive"):
        list(_support_row_blocks(matrix, target_nnz))


@pytest.mark.parametrize("rank", [0, 1, 3])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_rank_k_operator_matches_dense(rank, center, dtype):
    """Rank-k operator products and statistics match dense calculations."""
    rng = np.random.default_rng(104 + rank)
    n_obs, n_vars = 9, 7
    dense_sparse = rng.normal(size=(n_obs, n_vars))
    dense_sparse[rng.random(size=dense_sparse.shape) < 0.75] = 0.0
    S = sparse.csr_matrix(dense_sparse)
    left = rng.normal(size=(n_obs, rank))
    right = rng.normal(size=(n_vars, rank))
    representation = SparseLowRankMatrix(S, left, right)
    operator = SparseLowRankLinearOperator(representation, center=center, dtype=dtype)

    expected = dense_sparse + left @ right.T
    if dtype == "float32":
        expected = expected.astype(np.float32)
    if center:
        expected = expected - expected.mean(axis=0, dtype=np.float64).astype(dtype)

    identity = np.eye(n_vars, dtype=dtype)
    materialized = operator @ identity
    tolerance = _STATS_RTOL[dtype]
    np.testing.assert_allclose(materialized, expected, rtol=tolerance, atol=tolerance)
    assert operator.frobenius_squared_uncentered() == pytest.approx(
        np.sum((dense_sparse + left @ right.T).astype(dtype).astype(float) ** 2),
        rel=tolerance,
        abs=0.0,
    )
    expected_squared = np.sum(materialized.astype(float) ** 2)
    assert operator.frobenius_squared_centered() == pytest.approx(
        expected_squared, rel=tolerance, abs=0.0
    )


def test_duplicate_sparse_entries_are_rejected():
    """A non-canonical sparse part with duplicate entries fails the guard."""
    duplicated = sparse.csr_matrix(
        (
            np.array([1.0, 2.0]),
            np.array([1, 1], dtype=np.int32),
            np.array([0, 2, 2], dtype=np.int32),
        ),
        shape=(2, 3),
    )
    assert not duplicated.has_canonical_format
    representation = SparseLowRankMatrix(
        duplicated,
        np.zeros((2, 1)),
        np.zeros((3, 1)),
    )

    with pytest.raises(AssertionError, match="canonical CSR"):
        SparseLowRankLinearOperator(representation)


def test_representation_selection_and_scaling():
    """Column selection and scalar multiplication preserve represented values."""
    S = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
    representation = SparseLowRankMatrix(S, left, right)
    dense = S.toarray() + left @ right.T

    selected = representation.select_columns(np.array([True, False, True]))
    row_scaled = selected.scale_rows(np.array([2.0, 3.0]))
    scaled = row_scaled.scale_columns(np.array([5.0, 7.0]))
    actual = scaled.sparse.toarray() + scaled.left @ scaled.right.T
    expected = dense[:, [0, 2]] * np.array([2.0, 3.0])[:, None]
    expected *= np.array([5.0, 7.0])[None, :]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    scalar_scaled = scaled.scaled(0.25)
    actual = scalar_scaled.sparse.toarray()
    actual += scalar_scaled.left @ scalar_scaled.right.T
    np.testing.assert_allclose(actual, 0.25 * expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_blocked_statistics_match_materialized_representation(monkeypatch, dtype):
    """Blocked means and norms match the explicitly represented matrix."""
    _set_stats_block_nnz(monkeypatch, 4)
    representation = _statistics_fixture()
    assert np.array_equal(
        np.bincount(representation.sparse.indices, minlength=5),
        [0, 2, 3, 5, 7],
    )
    operator = SparseLowRankLinearOperator(representation, center=True, dtype=dtype)
    represented = operator.S.toarray() + operator.left @ operator.right.T
    assert operator.mean is not None
    deviations = (represented - operator.mean).astype(np.float64)

    _assert_statistics_close(
        operator,
        mean=represented.mean(axis=0, dtype=np.float64).astype(dtype),
        uncentered=np.sum(represented.astype(np.float64) ** 2),
        centered=np.sum(deviations**2),
        tolerance=_STATS_RTOL[dtype],
    )


def test_ill_conditioned_sparse_norm_is_recomputed_directly():
    """A sparse concentrated baseline falls back to accurate direct norms."""
    n_obs = 100
    baseline = 1e13
    sparse_part = sparse.csr_matrix(
        (np.array([-baseline + 1.0]), ([0], [0])),
        shape=(n_obs, 1),
    )
    left = np.zeros((n_obs, 1))
    left[0, 0] = baseline
    representation = SparseLowRankMatrix(sparse_part, left, np.ones((1, 1)))

    operator = SparseLowRankLinearOperator(representation, center=True, dtype="float64")
    represented = operator.S.toarray() + operator.left @ operator.right.T
    assert operator.S.nnz < n_obs // 2
    assert operator.mean is not None
    deviations = represented - operator.mean

    _assert_statistics_close(
        operator,
        mean=represented.mean(axis=0),
        uncentered=np.sum(represented**2),
        centered=np.sum(deviations**2),
        tolerance=_STATS_RTOL["float64"],
    )


def test_norm_direct_recalculation_boundary_is_inclusive():
    """The cancellation threshold and negative results request direct norms."""
    term_scale = 3.0
    boundary = operator_module._STATS_MIN_NORM_TERM_RATIO * term_scale

    assert operator_module._norm_needs_direct_recalculation(
        boundary,
        term_scale=term_scale,
    )
    assert not operator_module._norm_needs_direct_recalculation(
        np.nextafter(boundary, np.inf),
        term_scale=term_scale,
    )
    assert operator_module._norm_needs_direct_recalculation(-1.0, term_scale=0.0)
    assert not operator_module._norm_needs_direct_recalculation(0.0, term_scale=0.0)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_statistics_agree_across_row_block_sizes(monkeypatch, dtype):
    """Changing row-block granularity preserves statistics within roundoff."""
    representation = _statistics_fixture()
    _set_stats_block_nnz(monkeypatch, representation.sparse.nnz)
    single = SparseLowRankLinearOperator(representation, center=True, dtype=dtype)

    _set_stats_block_nnz(monkeypatch, 4)
    assert len(list(_support_row_blocks(representation.sparse, 4))) >= 4
    blocked = SparseLowRankLinearOperator(representation, center=True, dtype=dtype)

    _assert_statistics_close(
        blocked,
        mean=single.mean,
        uncentered=single.frobenius_squared_uncentered(),
        centered=single.frobenius_squared_centered(),
        tolerance=_STATS_RTOL[dtype],
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numerically_zero_squared_norm_boundary_is_inclusive(dtype):
    """Squared norms at the roundoff boundary pass and one ULP above fails."""
    shape = (5, 4)
    scale = 3.0
    eps = np.finfo(dtype).eps
    boundary = eps * eps * np.prod(shape) * scale

    assert _squared_norm_is_numerically_zero(
        boundary, scale=scale, shape=shape, dtype=dtype
    )
    assert not _squared_norm_is_numerically_zero(
        np.nextafter(boundary, np.inf),
        scale=scale,
        shape=shape,
        dtype=dtype,
    )
