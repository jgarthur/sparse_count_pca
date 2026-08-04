"""Tests for all-zero genes and their interaction with variable masks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp
from tests._oracles import (
    _dense_count_shifted_clr,
    _dense_dirichlet,
    _dense_proportion_shifted_clr,
)


def _with_zero_gene(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Append one all-zero gene without changing any cell totals."""
    zero_gene = sparse.csr_matrix((counts.shape[0], 1), dtype=counts.dtype)
    return sparse.hstack((counts, zero_gene), format="csr")


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("poisson", "deviance", None),
        ("binomial", "pearson", None),
        ("binomial", "deviance", None),
        ("scaled_nb", "pearson", np.array([0.0, 0.1, 0.3, 1.0])),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_residual_entry_points_reject_zero_gene_before_pca_mask(
    counts: sparse.csr_matrix,
    model: str,
    residual: str,
    alpha: np.ndarray | None,
) -> None:
    """Every residual family rejects an empty gene before PCA masking."""
    with_zero = _with_zero_gene(counts)
    adata = AnnData(with_zero)
    mask = np.ones(with_zero.shape[1], dtype=bool)
    mask[-1] = False
    full_alpha = None if alpha is None else np.append(alpha, 0.2)
    method = scp.Residual(model=model, residual=residual, alpha=full_alpha)

    with pytest.raises(ValueError, match="Genes with zero total counts"):
        scp.residual_pca(
            adata,
            n_comps=2,
            mask_var=mask,
            model=model,
            residual=residual,
            alpha=full_alpha,
        )

    with pytest.raises(ValueError, match="Genes with zero total counts"):
        scp.residual_pca_matrix(
            with_zero,
            n_comps=2,
            model=model,
            residual=residual,
            alpha=full_alpha,
        )

    with pytest.raises(ValueError, match="Genes with zero total counts"):
        scp.transform(with_zero, method)


@pytest.mark.parametrize(
    "entry_point",
    [
        lambda values: scp.residual_pca_matrix(values, n_comps=2),
        lambda values: scp.residual_pca(AnnData(values), n_comps=2, copy=True),
        lambda values: scp.transform(values, scp.Residual()),
    ],
    ids=["matrix-pca", "anndata-pca", "two-step-transform"],
)
def test_residual_entry_points_reject_zero_cell(
    counts: sparse.csr_matrix,
    entry_point: Callable[[sparse.csr_matrix], object],
) -> None:
    """Every residual entry point rejects an observation with no counts."""
    with_zero_cell = sparse.vstack(
        (counts, sparse.csr_matrix((1, counts.shape[1]), dtype=counts.dtype)),
        format="csr",
    )

    with pytest.raises(ValueError, match="Cells with zero total counts"):
        entry_point(with_zero_cell)


@pytest.mark.parametrize(
    ("method_factory", "dense_transform"),
    [
        (
            lambda prior: scp.ShiftedCLR(count_shift=0.7),
            lambda values, prior: _dense_count_shifted_clr(values, 0.7),
        ),
        (
            lambda prior: scp.ProportionShiftedCLR(composition_shift=0.05),
            lambda values, prior: _dense_proportion_shifted_clr(values, 0.05),
        ),
        (
            lambda prior: scp.DirichletLog(concentration=2.5, prior_proportions=prior),
            lambda values, prior: _dense_dirichlet(values, 2.5, prior, clr=False),
        ),
        (
            lambda prior: scp.DirichletCLR(concentration=2.5, prior_proportions=prior),
            lambda values, prior: _dense_dirichlet(values, 2.5, prior, clr=True),
        ),
    ],
    ids=[
        "shifted-clr",
        "proportion-shifted-clr",
        "dirichlet-log",
        "dirichlet-clr",
    ],
)
def test_log_and_dirichlet_transforms_accept_zero_genes(
    counts: sparse.csr_matrix,
    method_factory: Callable[[np.ndarray], scp.Transform],
    dense_transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Every shifted-CLR and Dirichlet transform matches its zero-gene formula."""
    with_zero = _with_zero_gene(counts)
    dense = with_zero.toarray().astype(np.float64)
    prior_weights = np.arange(1, with_zero.shape[1] + 1, dtype=np.float64)
    prior = prior_weights / prior_weights.sum()

    actual = scp.transform(with_zero, method_factory(prior)).materialize()
    expected = dense_transform(dense, prior)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    "method",
    [
        scp.ShiftedCLR(count_shift=0.7),
        scp.ProportionShiftedCLR(composition_shift=0.05),
        scp.DirichletLog(concentration=2.5),
        scp.DirichletCLR(concentration=2.5),
    ],
    ids=[
        "shifted-clr",
        "proportion-shifted-clr",
        "dirichlet-log",
        "dirichlet-clr",
    ],
)
def test_masked_zero_gene_remains_in_full_normalization_universe(
    counts: sparse.csr_matrix,
    method: scp.Transform,
) -> None:
    """Masking an empty gene happens after CLR or Dirichlet normalization."""
    with_zero = _with_zero_gene(counts)
    mask = np.ones(with_zero.shape[1], dtype=bool)
    mask[-1] = False
    transformed = scp.transform(with_zero, method)
    full_values = transformed.materialize()
    expected = full_values[:, mask] - full_values[:, mask].mean(axis=0)

    result = transformed.pca(n_comps=2, mask_var=mask, return_operator=True)

    assert result.operator is not None
    actual = result.operator @ np.eye(counts.shape[1])
    dropped = scp.transform(counts, method).materialize()
    dropped -= dropped.mean(axis=0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert not np.allclose(actual, dropped, rtol=0.0, atol=1e-8)
    assert result.params["normalization_n_vars"] == with_zero.shape[1]
    assert result.params["pca_n_vars"] == counts.shape[1]


def test_correspondence_mask_excludes_zero_gene_before_computing_margins(
    counts: sparse.csr_matrix,
) -> None:
    """CA excluding an empty gene matches analysis of the original table."""
    with_zero = _with_zero_gene(counts)
    mask = np.ones(with_zero.shape[1], dtype=bool)
    mask[-1] = False
    expected = scp.correspondence_analysis_matrix(counts, n_comps=2)
    actual = scp.correspondence_analysis(
        AnnData(with_zero),
        n_comps=2,
        mask_var=mask,
        key_added="zero_gene_ca",
        copy=True,
    )

    np.testing.assert_allclose(
        actual.uns["zero_gene_ca"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.obsm["zero_gene_ca"],
        expected.row_principal_coordinates,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.varm["zero_gene_ca"][mask],
        expected.column_principal_coordinates,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.uns["zero_gene_ca"]["row_masses"],
        expected.row_masses,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.uns["zero_gene_ca"]["column_masses"],
        expected.column_masses,
        rtol=0.0,
        atol=0.0,
    )
    assert np.isnan(actual.varm["zero_gene_ca"][~mask]).all()
