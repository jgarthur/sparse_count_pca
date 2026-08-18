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
    tolerance = 2e-6 if dtype == "float32" else 1e-12
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
    monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", 4)
    monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", 4)
    representation = _statistics_fixture()
    assert np.array_equal(
        np.bincount(representation.sparse.indices, minlength=5),
        [0, 2, 3, 5, 7],
    )
    operator = SparseLowRankLinearOperator(
        representation,
        center=True,
        dtype=dtype,
    )
    represented = operator.S.toarray()
    represented += operator.left @ operator.right.T
    expected_mean = represented.mean(axis=0, dtype=np.float64).astype(dtype)
    expected_uncentered = np.sum(represented.astype(np.float64) ** 2)
    assert operator.mean is not None
    centered = (represented - operator.mean).astype(np.float64)
    expected_centered = np.sum(centered**2)
    tolerance = 2e-6 if dtype == "float32" else 1e-12

    np.testing.assert_allclose(
        operator.mean,
        expected_mean,
        rtol=tolerance,
        atol=tolerance,
    )
    assert operator.frobenius_squared_uncentered() == pytest.approx(
        expected_uncentered,
        rel=tolerance,
        abs=0.0,
    )
    assert operator.frobenius_squared_centered() == pytest.approx(
        expected_centered,
        rel=tolerance,
        abs=0.0,
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_statistics_are_stable_across_row_block_sizes(monkeypatch, dtype):
    """Changing row-block granularity preserves statistics within roundoff."""
    representation = _statistics_fixture()
    monkeypatch.setattr(
        operator_module,
        "_STATS_MEAN_BLOCK_NNZ",
        representation.sparse.nnz,
    )
    monkeypatch.setattr(
        operator_module,
        "_STATS_NORM_BLOCK_NNZ",
        representation.sparse.nnz,
    )
    single = SparseLowRankLinearOperator(
        representation,
        center=True,
        dtype=dtype,
    )

    monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", 4)
    monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", 4)
    assert len(list(_support_row_blocks(representation.sparse, 4))) >= 4
    blocked = SparseLowRankLinearOperator(
        representation,
        center=True,
        dtype=dtype,
    )

    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(blocked.mean, single.mean, rtol=tolerance, atol=0.0)
    assert blocked.frobenius_squared_uncentered() == pytest.approx(
        single.frobenius_squared_uncentered(),
        rel=tolerance,
        abs=0.0,
    )
    assert blocked.frobenius_squared_centered() == pytest.approx(
        single.frobenius_squared_centered(),
        rel=tolerance,
        abs=0.0,
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
