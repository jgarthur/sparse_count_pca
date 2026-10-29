"""Compatibility tests against Scanpy total normalization and log1p."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from sparse_count_pca import Log1pNormalized, log1p_norm_pca_matrix, transform
from sparse_count_pca import _operator as operator_module
from tests._oracles import _assert_pca_agreement

scanpy = pytest.importorskip("scanpy")


@pytest.mark.parametrize(
    "target_sum",
    [None, 1e4],
    ids=["median-target", "explicit-target"],
)
def test_log1p_norm_matches_scanpy_transform(target_sum):
    """Median and explicit targets agree with Scanpy on a six-by-four count table."""
    counts = sparse.csr_matrix(
        [
            [5, 1, 0, 2],
            [1, 4, 2, 0],
            [0, 2, 5, 1],
            [3, 0, 1, 4],
            [2, 3, 0, 2],
            [1, 1, 3, 2],
        ]
    )
    explicit = AnnData(counts.astype(np.float64))
    scanpy.pp.normalize_total(explicit, target_sum=target_sum)
    scanpy.pp.log1p(explicit)
    expected_values = explicit.X.toarray()

    actual_values = transform(
        counts,
        Log1pNormalized(target_sum=target_sum),
        dtype="float64",
    ).materialize()
    np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("target_nnz", [32, 1_000_000])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rank_zero_log1p_pca_matches_scanpy(monkeypatch, target_nnz, dtype):
    """CSR statistics preserve Scanpy PCA scores, loadings, and variance."""
    rng = np.random.default_rng(318)
    counts = rng.integers(1, 20, size=(32, 19))
    counts[rng.random(counts.shape) < 0.2] = 0
    counts[:, -1] = 0
    counts[::2, -1] = 2
    counts = sparse.csr_matrix(counts)
    explicit = AnnData(counts.astype(np.float64))
    scanpy.pp.normalize_total(explicit, target_sum=None)
    scanpy.pp.log1p(explicit)
    explicit.X = explicit.X.astype(dtype)
    scanpy.pp.pca(explicit, n_comps=5, svd_solver="arpack", random_state=0, dtype=dtype)
    monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", target_nnz)
    monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", target_nnz)

    def reject_dense_batches(*args, **kwargs):
        """Reject direct dense reconstruction for normalized-log1p statistics."""
        raise AssertionError("log1p statistics must use the rank-zero CSR path")

    monkeypatch.setattr(
        operator_module.SparseLowRankLinearOperator,
        "_represented_column_batches",
        reject_dense_batches,
    )

    actual = log1p_norm_pca_matrix(counts, n_comps=5, dtype=dtype, random_state=0)
    expected_scores = explicit.obsm["X_pca"]
    expected_components = explicit.varm["PCs"].T
    expected_variance = explicit.uns["pca"]["variance"]
    expected_ratios = explicit.uns["pca"]["variance_ratio"]
    _assert_pca_agreement(
        actual,
        np.sqrt(expected_variance.astype(np.float64) * (counts.shape[0] - 1)),
        expected_scores,
        expected_components,
    )
    tolerance = 2e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        actual.explained_variance, expected_variance, rtol=tolerance
    )
    np.testing.assert_allclose(
        actual.explained_variance_ratio, expected_ratios, rtol=tolerance
    )
    assert actual.total_variance == pytest.approx(
        expected_variance[0] / expected_ratios[0], rel=tolerance
    )
