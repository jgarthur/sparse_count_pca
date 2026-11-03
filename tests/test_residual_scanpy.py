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


def _simulated_hvg_counts():
    """Return overdispersed counts with an empty gene and a two-cell gene.

    The two-cell gene's residuals exceed ``sqrt(n_obs)``, so Scanpy's default
    clipping changes its variance.
    """
    rng = np.random.default_rng(7)
    n_obs, n_vars = 300, 40
    gene_means = rng.gamma(0.6, 2.0, size=n_vars)
    depths = rng.gamma(4.0, 0.5, size=n_obs)
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + np.outer(depths, gene_means)))
    counts[:, 0] = 0
    counts[:, 1] = 0
    counts[[5, 9], 1] = [60, 45]
    assert (counts.sum(axis=1) > 0).all()
    return sparse.csr_matrix(counts.astype(np.float64))


@pytest.mark.parametrize(
    ("package_clip", "scanpy_clip"),
    [("scanpy", None), (None, np.inf)],
    ids=["clipped", "unclipped"],
)
def test_poisson_residual_variances_match_scanpy_hvg(package_clip, scanpy_clip):
    """Poisson residual variances reproduce Scanpy's Pearson-residual HVGs.

    Scanpy's ``theta=np.inf`` is the Poisson model, its default ``clip=None``
    clips at ``sqrt(n_obs)`` like ``clip="scanpy"``, and its residual variances
    divide by ``n_obs``. With one batch, Scanpy selects the genes with the
    largest residual variance, so ranking these variances selects the same
    genes. The tolerance allows float64 summation-order differences, measured
    near 2e-15, and detects a ``ddof`` error of ``1 / n_obs``.
    """
    counts = _simulated_hvg_counts()
    n_top_genes = 10
    reference = AnnData(counts.copy())
    scanpy.experimental.pp.highly_variable_genes(
        reference,
        flavor="pearson_residuals",
        theta=np.inf,
        clip=scanpy_clip,
        n_top_genes=n_top_genes,
    )

    stats = scp.transform(counts, scp.Residual(clip=package_clip)).variable_statistics(
        ddof=0
    )

    expected = reference.var["residual_variances"].to_numpy()
    np.testing.assert_allclose(stats.variance, expected, rtol=1e-12, atol=0.0)
    order = np.argsort(-stats.variance)
    assert stats.variance[order[n_top_genes - 1]] > stats.variance[order[n_top_genes]]
    selected = np.zeros(counts.shape[1], dtype=bool)
    selected[order[:n_top_genes]] = True
    np.testing.assert_array_equal(selected, reference.var["highly_variable"].to_numpy())
