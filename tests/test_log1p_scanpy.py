"""Compatibility tests against Scanpy total normalization and log1p."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from sparse_count_pca import Log1pNormalized, log1p_norm_pca_matrix, transform
from sparse_count_pca import _operator as operator_module

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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batched_log1p_pca_matches_scanpy(monkeypatch, dtype):
    """Batched statistics preserve Scanpy PCA scores, loadings, and variance."""
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
    monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", 64)
    monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", 32)
    monkeypatch.setattr(operator_module, "_STATS_DENSE_BATCH_VALUES", 9)

    actual = log1p_norm_pca_matrix(counts, n_comps=5, dtype=dtype, random_state=0)
    expected_scores = explicit.obsm["X_pca"]
    expected_components = explicit.varm["PCs"].T
    signs = np.sign(np.sum(actual.components * expected_components, axis=1))
    tolerance = 2e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(
        actual.scores * signs, expected_scores, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual.components * signs[:, None],
        expected_components,
        rtol=tolerance,
        atol=tolerance,
    )
    expected_variance = explicit.uns["pca"]["variance"]
    expected_ratios = explicit.uns["pca"]["variance_ratio"]
    np.testing.assert_allclose(
        actual.singular_values,
        np.sqrt(expected_variance * (counts.shape[0] - 1)),
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        actual.explained_variance, expected_variance, rtol=tolerance
    )
    np.testing.assert_allclose(
        actual.explained_variance_ratio, expected_ratios, rtol=tolerance
    )
    assert actual.total_variance == pytest.approx(
        expected_variance[0] / expected_ratios[0], rel=tolerance
    )
