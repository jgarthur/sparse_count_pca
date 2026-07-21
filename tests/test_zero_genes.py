"""Tests for all-zero genes and their interaction with variable masks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp


def _with_zero_gene(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """Append one all-zero gene without changing any cell totals."""
    zero_gene = sparse.csr_matrix((counts.shape[0], 1), dtype=counts.dtype)
    return sparse.hstack((counts, zero_gene), format="csr")


def _center_rows(values: np.ndarray) -> np.ndarray:
    """Return values centered independently within each row."""
    return values - values.mean(axis=1, keepdims=True)


def _shifted_log(values: np.ndarray) -> np.ndarray:
    """Evaluate the shifted-log test formula densely."""
    return np.log1p(values / 0.7)


def _shifted_clr(values: np.ndarray) -> np.ndarray:
    """Evaluate the count-shifted CLR test formula densely."""
    return _center_rows(np.log(values + 0.7))


def _proportion_shifted_clr(values: np.ndarray) -> np.ndarray:
    """Evaluate the proportion-shifted CLR test formula densely."""
    proportions = values / values.sum(axis=1, keepdims=True)
    return _center_rows(np.log(proportions + 0.05))


def _dirichlet(
    values: np.ndarray,
    prior: np.ndarray,
    *,
    clr: bool,
) -> np.ndarray:
    """Evaluate the Dirichlet posterior-mean log test formula densely."""
    concentration = 2.5
    posterior = (values + concentration * prior) / (
        values.sum(axis=1, keepdims=True) + concentration
    )
    logged = np.log(posterior)
    return _center_rows(logged) if clr else logged


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
def test_residual_pca_rejects_selected_zero_gene_and_accepts_masked_zero_gene(
    counts: sparse.csr_matrix,
    model: str,
    residual: str,
    alpha: np.ndarray | None,
) -> None:
    """Every residual family accepts an empty gene only when it is masked out."""
    with_zero = _with_zero_gene(counts)
    adata = AnnData(with_zero)
    mask = np.ones(with_zero.shape[1], dtype=bool)
    mask[-1] = False
    full_alpha = None if alpha is None else np.append(alpha, 0.2)

    with pytest.raises(ValueError, match="Selected genes with zero total counts"):
        scp.residual_pca(
            adata,
            n_comps=2,
            mask_var=None,
            model=model,
            residual=residual,
            alpha=full_alpha,
        )

    expected = scp.residual_pca_matrix(
        counts,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=alpha,
        dtype="float64",
    )
    actual = scp.residual_pca(
        adata,
        n_comps=2,
        mask_var=mask,
        model=model,
        residual=residual,
        alpha=full_alpha,
        copy=True,
        dtype="float64",
    )

    np.testing.assert_allclose(
        actual.uns["pca"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.obsm["X_pca"], expected.scores, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.varm["PCs"][mask],
        expected.components.T,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isnan(actual.varm["PCs"][~mask]).all()
    assert actual.uns["pca"]["params"]["normalization_n_vars"] == with_zero.shape[1]
    assert actual.uns["pca"]["params"]["pca_n_vars"] == counts.shape[1]


@pytest.mark.parametrize(
    ("method_factory", "dense_transform"),
    [
        (
            lambda prior: scp.ShiftedLog(count_shift=0.7),
            lambda values, prior: _shifted_log(values),
        ),
        (
            lambda prior: scp.ShiftedCLR(count_shift=0.7),
            lambda values, prior: _shifted_clr(values),
        ),
        (
            lambda prior: scp.ProportionShiftedCLR(composition_shift=0.05),
            lambda values, prior: _proportion_shifted_clr(values),
        ),
        (
            lambda prior: scp.DirichletLog(concentration=2.5, prior_proportions=prior),
            lambda values, prior: _dirichlet(values, prior, clr=False),
        ),
        (
            lambda prior: scp.DirichletCLR(concentration=2.5, prior_proportions=prior),
            lambda values, prior: _dirichlet(values, prior, clr=True),
        ),
    ],
    ids=[
        "shifted-log",
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
    """Every shifted-log and Dirichlet transform matches its zero-gene formula."""
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
