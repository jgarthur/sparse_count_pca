"""Compatibility tests against Scanpy's Pearson-residual PCA workflow."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

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
    np.testing.assert_allclose(explicit.X, expected_residuals, rtol=0.0, atol=1e-12)

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


@pytest.mark.parametrize(
    ("package_clip", "scanpy_clip"),
    [(None, np.inf), (0.75, 0.75)],
    ids=["unclipped", "clipped"],
)
def test_equal_depth_scaled_nb_matches_sctransform_pearson_form(
    package_clip, scanpy_clip
):
    """Equal-depth scaled-NB residual PCA matches the SCTransform NB form."""
    counts = sparse.csr_matrix(
        [
            [8, 1, 0, 1],
            [0, 7, 2, 1],
            [1, 0, 8, 1],
            [2, 3, 0, 5],
            [6, 0, 1, 3],
            [0, 5, 4, 1],
        ]
    )
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    np.testing.assert_array_equal(row_totals, np.full(counts.shape[0], 10))
    theta = 4.0
    alpha = 1.0 / theta

    explicit = AnnData(counts.copy())
    scanpy.experimental.pp.normalize_pearson_residuals(
        explicit,
        theta=theta,
        clip=scanpy_clip,
    )
    expected_residuals = _materialize_dense_residual(
        counts,
        model="scaled_nb",
        alpha=alpha,
        clip=package_clip,
    )
    np.testing.assert_allclose(
        explicit.X,
        expected_residuals,
        rtol=0.0,
        atol=1e-12,
    )

    if package_clip is not None:
        unclipped = _materialize_dense_residual(
            counts,
            model="scaled_nb",
            alpha=alpha,
        )
        assert (np.abs(unclipped) > package_clip).any()
        assert not np.allclose(
            np.clip(
                unclipped - unclipped.mean(axis=0),
                -package_clip,
                package_clip,
            ),
            expected_residuals - expected_residuals.mean(axis=0),
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
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model="scaled_nb",
        residual="pearson",
        alpha=alpha,
        clip=package_clip,
        clip_max_nnz_ratio=None,
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
    np.testing.assert_allclose(
        result.loadings,
        explicit.varm["PCs"],
        rtol=1e-8,
        atol=1e-8,
    )
