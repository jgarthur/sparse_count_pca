"""Real-data tests for variable masks and all-zero gene handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp

FIXTURE_DIR = Path(__file__).resolve().parent
RAW_COUNTS_PATH = FIXTURE_DIR / "pbmc3k_raw_counts.npz"
MASK_ATOL = 2e-12


@pytest.fixture(scope="module")
def raw_counts() -> sparse.csr_matrix:
    """Load the pinned unequal-depth PBMC3k-derived count matrix."""
    return sparse.load_npz(RAW_COUNTS_PATH).tocsr()


def _gene_mask(n_vars: int) -> np.ndarray:
    """Select a deterministic, noncontiguous majority of real genes."""
    mask = np.arange(n_vars) % 3 != 1
    assert mask.any() and not mask.all()
    return mask


def _append_zero_gene(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Append one structurally empty gene without changing the real counts."""
    zero = sparse.csr_matrix((counts.shape[0], 1), dtype=counts.dtype)
    return sparse.hstack((counts, zero), format="csr")


@pytest.mark.parametrize(
    ("method", "expect_support_growth"),
    [
        (
            scp.Residual(
                model="scaled_nb",
                residual="deviance",
                alpha=0.1,
                clip=1.25,
                clip_mode="symmetric",
                clip_max_nnz_ratio=None,
            ),
            True,
        ),
        (scp.ShiftedCLR(count_shift=0.75), False),
        (scp.ProportionShiftedCLR(composition_shift=0.00075), False),
        (scp.DirichletLog(concentration=12.5), False),
        (scp.DirichletCLR(concentration=12.5), False),
    ],
    ids=[
        "clipped-scaled-nb-residual",
        "shifted-clr",
        "proportion-shifted-clr",
        "dirichlet-log",
        "dirichlet-clr",
    ],
)
def test_real_data_pca_mask_selects_from_full_fitted_transform(
    raw_counts: sparse.csr_matrix,
    method: scp.Transform,
    expect_support_growth: bool,
) -> None:
    """Every transform applies a realistic PCA mask after full-data fitting."""
    mask = _gene_mask(raw_counts.shape[1])
    transformed = scp.transform(raw_counts, method, dtype="float64")
    full = transformed.materialize(block_size=64)
    expected = full[:, mask]
    expected -= expected.mean(axis=0)

    result = transformed.pca(n_comps=2, mask_var=mask, return_operator=True)

    assert result.operator is not None
    actual = result.operator @ np.eye(mask.sum(), dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=MASK_ATOL)
    assert result.params["normalization_n_vars"] == raw_counts.shape[1]
    assert result.params["pca_n_vars"] == int(mask.sum())
    if expect_support_growth:
        assert transformed._sparse.nnz > raw_counts.nnz


def test_all_true_matrix_mask_matches_implicit_all_gene_selection(
    raw_counts: sparse.csr_matrix,
) -> None:
    """An explicit all-true matrix mask is equivalent to omitting the mask."""
    transformed = scp.transform(raw_counts, scp.ShiftedCLR(count_shift=0.75))
    implicit = transformed.pca(n_comps=2, return_operator=True)
    explicit = transformed.pca(
        n_comps=2,
        mask_var=np.ones(raw_counts.shape[1], dtype=bool),
        return_operator=True,
    )

    assert implicit.operator is not None
    assert explicit.operator is not None
    identity = np.eye(raw_counts.shape[1], dtype=np.float64)
    np.testing.assert_allclose(
        explicit.operator @ identity,
        implicit.operator @ identity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        explicit.singular_values,
        implicit.singular_values,
        rtol=0.0,
        atol=MASK_ATOL,
    )
    assert explicit.params["pca_n_vars"] == raw_counts.shape[1]


def test_all_false_matrix_mask_is_rejected(raw_counts: sparse.csr_matrix) -> None:
    """The matrix-backed two-step PCA API rejects a mask selecting no genes."""
    transformed = scp.transform(raw_counts, scp.ShiftedCLR(count_shift=0.75))

    with pytest.raises(ValueError, match="mask_var selected zero genes"):
        transformed.pca(
            n_comps=2,
            mask_var=np.zeros(raw_counts.shape[1], dtype=bool),
        )


def test_anndata_residual_is_unchanged_by_a_real_data_zero_gene(
    raw_counts: sparse.csr_matrix,
) -> None:
    """Clipped residual PCA on real counts is unchanged by an appended empty gene."""
    counts = _append_zero_gene(raw_counts)
    adata = AnnData(counts)
    adata.var_names = [
        *(f"gene-{index}" for index in range(raw_counts.shape[1])),
        "zero",
    ]
    kwargs = dict(
        n_comps=2,
        model="scaled_nb",
        residual="deviance",
        alpha=0.1,
        clip=1.25,
        clip_mode="symmetric",
        clip_max_nnz_ratio=None,
    )
    expected = scp.residual_pca_matrix(raw_counts, dtype="float64", **kwargs)

    result = scp.residual_pca(adata, copy=True, dtype="float64", **kwargs)

    np.testing.assert_allclose(
        result.uns["pca"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.varm["PCs"][:-1], expected.components.T, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(result.varm["PCs"][-1], 0.0, rtol=0.0, atol=1e-12)
    params = result.uns["pca"]["params"]
    assert params["pca_n_vars"] == counts.shape[1]
    assert params["n_empty_vars"] == 1


def test_correspondence_mask_excludes_a_real_data_zero_gene(
    raw_counts: sparse.csr_matrix,
) -> None:
    """CA excluding an appended empty gene equals CA of the original real table."""
    counts = _append_zero_gene(raw_counts)
    adata = AnnData(counts)
    mask = np.ones(counts.shape[1], dtype=bool)
    mask[-1] = False
    expected = scp.correspondence_analysis_matrix(raw_counts, n_comps=2)

    result = scp.correspondence_analysis(
        adata,
        n_comps=2,
        mask_var=mask,
        copy=True,
    )

    np.testing.assert_allclose(
        result.uns["ca"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=MASK_ATOL,
    )
    np.testing.assert_allclose(
        result.uns["ca"]["row_masses"], expected.row_masses, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result.uns["ca"]["column_masses"],
        expected.column_masses,
        rtol=0.0,
        atol=0.0,
    )
    assert np.isfinite(result.varm["CA"][:-1]).all()
    assert np.isnan(result.varm["CA"][-1]).all()
    params = result.uns["ca"]["params"]
    assert params["mask_var_details"]["n_vars_used"] == raw_counts.shape[1]
