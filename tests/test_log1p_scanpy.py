"""Compatibility tests against Scanpy total normalization and log1p PCA."""

import numpy as np
import pytest
from anndata import AnnData

from sparse_count_pca import log1p_norm_pca_matrix
from tests._oracles import _compare_subspaces

scanpy = pytest.importorskip("scanpy")


@pytest.mark.parametrize(
    "target_sum",
    [None, 1e4],
    ids=["median-target", "explicit-target"],
)
def test_log1p_norm_matches_scanpy_transform_and_pca(counts, target_sum):
    """Target-normalized log1p values and PCA agree with Scanpy."""
    explicit = AnnData(counts.astype(np.float64))
    scanpy.pp.normalize_total(explicit, target_sum=target_sum)
    scanpy.pp.log1p(explicit)
    expected_values = explicit.X.toarray()

    result = log1p_norm_pca_matrix(
        counts,
        n_comps=2,
        target_sum=target_sum,
        random_state=0,
        dtype="float64",
        return_operator=True,
    )
    expected_centered = expected_values - expected_values.mean(axis=0)
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]),
        expected_centered,
        rtol=0.0,
        atol=1e-12,
    )

    scanpy.pp.pca(
        explicit,
        n_comps=2,
        svd_solver="arpack",
        random_state=0,
        dtype="float64",
    )
    scanpy_singular_values = np.sqrt(
        explicit.uns["pca"]["variance"] * (counts.shape[0] - 1)
    )
    np.testing.assert_allclose(
        result.singular_values,
        scanpy_singular_values,
        rtol=1e-10,
        atol=1e-10,
    )
    assert _compare_subspaces(
        result.components.T,
        explicit.varm["PCs"],
        rtol=1e-8,
        atol=1e-8,
    )
