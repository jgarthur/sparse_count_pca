"""Tests for sparse-plus-low-rank matrix and operator behavior."""

import math

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import residual_pca_matrix
from sparse_count_pca._operator import (
    SparseLowRankLinearOperator,
    _squared_norm_is_numerically_zero,
    _stripe_bounds,
)
from sparse_count_pca._representation import SparseLowRankMatrix


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


def _patterned_matrix(rng, n_obs, counts):
    """Build a CSR matrix whose columns hold the given stored-entry counts."""
    dense = np.zeros((n_obs, len(counts)))
    for column, count in enumerate(counts):
        rows = rng.choice(n_obs, size=count, replace=False)
        dense[rows, column] = rng.normal(size=count)
    return sparse.csr_matrix(dense)


def _reference_statistics(operator, *, center):
    """Recompute the operator's statistics from one full CSC copy.

    Reproduces the whole-matrix traversal that the chunked sweeps replaced, so
    the stripe results can be compared for exact equality rather than tolerance.
    """
    n_obs, n_vars = operator.shape
    csc = operator.S.tocsc()

    def column_support(column):
        start, stop = csc.indptr[column : column + 2]
        return csc.indices[start:stop], csc.data[start:stop]

    means = np.empty(n_vars, dtype=np.float64)
    for column in range(n_vars):
        rows, data = column_support(column)
        if rows.size > n_obs // 2:
            values = operator.left @ operator.right[column]
            values[rows] += data
            means[column] = math.fsum(values.astype(np.float64, copy=False)) / n_obs
        else:
            sparse_sum = math.fsum(data.astype(np.float64, copy=False))
            baseline_sum = float(
                operator._left_sum_float64 @ operator._right_float64[column]
            )
            means[column] = math.fsum((baseline_sum, sparse_sum)) / n_obs

    def squared_norm(center_vector):
        center_vector = np.asarray(center_vector, dtype=operator.dtype)
        column_norms = []
        for column in range(n_vars):
            rows, data = column_support(column)
            if rows.size > n_obs // 2:
                values = operator.left @ operator.right[column]
                values[rows] += data
                deviations = (values - center_vector[column]).astype(
                    np.float64, copy=False
                )
                column_norms.append(math.fsum(deviations * deviations))
                continue

            v = operator._right_float64[column]
            offset = float(v @ operator._left_mean_float64) - float(
                center_vector[column]
            )
            baseline_total = float(
                v @ operator._left_centered_gram_float64 @ v + n_obs * offset * offset
            )
            baseline_support = operator._left_float64[rows] @ v - float(
                center_vector[column]
            )
            stored_values = np.asarray(
                operator.left[rows] @ operator.right[column] + data,
                dtype=operator.dtype,
            )
            actual_support = (stored_values - center_vector[column]).astype(
                np.float64, copy=False
            )
            column_norms.append(
                max(
                    math.fsum(
                        (
                            baseline_total,
                            -math.fsum(baseline_support * baseline_support),
                            math.fsum(actual_support * actual_support),
                        )
                    ),
                    0.0,
                )
            )
        return math.fsum(column_norms)

    uncentered = squared_norm(np.zeros(n_vars, dtype=operator.dtype))
    centered = squared_norm(means.astype(operator.dtype)) if center else uncentered
    return means, uncentered, centered


# Column stored-entry counts chosen so that a chunk target of five values plans
# three stripes: an oversized dense column alone, then an empty column with a
# dense column on the closing boundary, then a ragged remainder.
_STRIPE_COUNTS = (6, 0, 2, 5, 1, 0, 3)


def test_stripe_bounds_group_whole_columns():
    """Stripe planning keeps oversized columns alone and never splits a column."""
    counts = np.array(_STRIPE_COUNTS)
    assert list(_stripe_bounds(counts, 5)) == [(0, 1), (1, 4), (4, 7)]
    assert list(_stripe_bounds(counts, 10_000)) == [(0, 7)]
    assert list(_stripe_bounds(np.array([], dtype=np.int64), 5)) == []
    assert list(_stripe_bounds(np.zeros(3, dtype=np.int64), 5)) == [(0, 3)]


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_chunked_statistics_match_full_csc_exactly(monkeypatch, rank, center, dtype):
    """Multi-stripe means and squared norms equal a full-CSC traversal bitwise."""
    monkeypatch.setattr("sparse_count_pca._operator._STATS_CHUNK_NNZ", 5)
    rng = np.random.default_rng(2026)
    n_obs = 9
    S = _patterned_matrix(rng, n_obs, _STRIPE_COUNTS)
    assert np.array_equal(np.bincount(S.indices, minlength=7), _STRIPE_COUNTS)

    representation = SparseLowRankMatrix(
        S,
        rng.normal(size=(n_obs, rank)),
        rng.normal(size=(len(_STRIPE_COUNTS), rank)),
    )
    operator = SparseLowRankLinearOperator(representation, center=center, dtype=dtype)
    assert len(list(operator._column_stripes())) == 3

    means, uncentered, centered = _reference_statistics(operator, center=center)
    if center:
        assert np.array_equal(operator.mean, means.astype(dtype))
    else:
        assert operator.mean is None
    assert operator.frobenius_squared_uncentered() == uncentered
    assert operator.frobenius_squared_centered() == centered


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_residual_pca_is_unchanged_by_stripe_count(monkeypatch, dtype):
    """Clipped residual PCA outputs are bitwise equal across stripe granularity."""
    rng = np.random.default_rng(11)
    counts = sparse.csr_matrix(rng.poisson(0.4, size=(24, 15)).astype(np.float64))
    keywords = {
        "n_comps": 3,
        "clip": 2.0,
        "clip_mode": "symmetric",
        "dtype": dtype,
        "solver": "arpack",
        "random_state": 0,
    }

    single = residual_pca_matrix(counts, **keywords)
    monkeypatch.setattr("sparse_count_pca._operator._STATS_CHUNK_NNZ", 4)
    chunked = residual_pca_matrix(counts, **keywords)

    assert np.array_equal(single.scores, chunked.scores)
    assert np.array_equal(single.components, chunked.components)
    assert np.array_equal(single.singular_values, chunked.singular_values)
    assert np.array_equal(
        single.explained_variance_ratio, chunked.explained_variance_ratio
    )
    assert single.total_variance == chunked.total_variance


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
