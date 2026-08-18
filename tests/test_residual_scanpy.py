"""Compatibility tests against Scanpy's Pearson-residual transform."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp

scanpy = pytest.importorskip("scanpy")


def test_poisson_pearson_matches_scanpy_residuals(counts):
    """Poisson Pearson residuals agree with Scanpy's transform."""
    explicit = AnnData(counts.copy())
    scanpy.experimental.pp.normalize_pearson_residuals(
        explicit,
        theta=np.inf,
        clip=np.inf,
    )
    actual_residuals = scp.transform(
        counts,
        scp.Residual(model="poisson", residual="pearson", clip=None),
        dtype="float64",
    ).materialize()
    np.testing.assert_allclose(explicit.X, actual_residuals, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("package_clip", "scanpy_clip"),
    [(None, np.inf), (0.75, 0.75)],
    ids=["unclipped", "clipped"],
)
def test_equal_depth_scaled_nb_matches_sctransform_pearson_form(
    package_clip, scanpy_clip
):
    """Equal-depth scaled-NB residuals match the SCTransform NB form."""
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
    actual_residuals = scp.transform(
        counts,
        scp.Residual(
            model="scaled_nb",
            residual="pearson",
            alpha=alpha,
            clip=package_clip,
            clip_max_nnz_ratio=None,
        ),
        dtype="float64",
    ).materialize()
    np.testing.assert_allclose(
        explicit.X,
        actual_residuals,
        rtol=0.0,
        atol=1e-12,
    )

    if package_clip is not None:
        unclipped = scp.transform(
            counts,
            scp.Residual(
                model="scaled_nb",
                residual="pearson",
                alpha=alpha,
                clip=None,
            ),
            dtype="float64",
        ).materialize()
        assert (np.abs(unclipped) > package_clip).any()
        assert not np.allclose(
            np.clip(
                unclipped - unclipped.mean(axis=0),
                -package_clip,
                package_clip,
            ),
            actual_residuals - actual_residuals.mean(axis=0),
            rtol=0.0,
            atol=1e-12,
        )
