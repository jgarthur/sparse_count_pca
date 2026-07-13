"""Compatibility tests against Scanpy's Pearson-residual PCA workflow."""

import numpy as np
import pytest
from anndata import AnnData

from sparse_count_pca import residual_pca_matrix
from tests._oracles import _materialize_dense_residual

scanpy = pytest.importorskip("scanpy")


def test_poisson_pearson_matches_scanpy_residuals_and_pca(counts):
    """Poisson Pearson residuals and PCA agree with Scanpy's workflow."""
    explicit = AnnData(counts.copy())
    scanpy.experimental.pp.normalize_pearson_residuals(
        explicit,
        theta=np.inf,
        clip=np.inf,
    )
    expected_residuals = _materialize_dense_residual(counts)
    np.testing.assert_allclose(explicit.X, expected_residuals, atol=1e-12)

    scanpy.pp.pca(
        explicit,
        n_comps=2,
        svd_solver="arpack",
        random_state=0,
        dtype="float64",
    )
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model="poisson",
        residual="pearson",
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
    # Singular values are distinct for this fixture, so the ordered, sign-flipped
    # component vectors are uniquely comparable rather than only their subspace.
    np.testing.assert_allclose(
        result.loadings,
        explicit.varm["PCs"],
        atol=1e-8,
        rtol=1e-8,
    )
