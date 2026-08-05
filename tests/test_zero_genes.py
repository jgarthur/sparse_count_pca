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
def test_residual_empty_gene_is_inert_and_changes_no_retained_output(
    counts: sparse.csr_matrix,
    model: str,
    residual: str,
    alpha: np.ndarray | None,
) -> None:
    """Every residual family gives an empty gene a zero column and zero loading."""
    with_zero = _with_zero_gene(counts)
    full_alpha = None if alpha is None else np.append(alpha, 0.2)
    expected = scp.residual_pca_matrix(
        counts,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=alpha,
        dtype="float64",
    )

    padded = scp.residual_pca_matrix(
        with_zero,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=full_alpha,
        dtype="float64",
    )
    # The variable universe is preserved, so components stay aligned to input
    # columns and the empty gene simply carries no loading.
    assert padded.components.shape[1] == with_zero.shape[1]
    np.testing.assert_allclose(
        padded.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(padded.scores, expected.scores, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        padded.components[:, :-1], expected.components, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(padded.components[:, -1], 0.0, rtol=0.0, atol=1e-12)
    assert padded.params["n_empty_vars"] == 1

    actual = scp.residual_pca(
        AnnData(with_zero),
        n_comps=2,
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
        actual.varm["PCs"][:-1], expected.components.T, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(actual.varm["PCs"][-1], 0.0, rtol=0.0, atol=1e-12)


def test_two_step_transform_keeps_the_source_variable_universe(
    counts: sparse.csr_matrix,
) -> None:
    """A fitted residual transform accepts masks sized to its original input."""
    with_zero = _with_zero_gene(counts)
    transformed = scp.transform(with_zero, scp.Residual())
    mask = np.ones(with_zero.shape[1], dtype=bool)
    mask[0] = False

    assert transformed.shape[1] == with_zero.shape[1]
    np.testing.assert_allclose(
        transformed.materialize()[:, -1], 0.0, rtol=0.0, atol=1e-12
    )

    result = transformed.pca(n_comps=2, mask_var=mask)

    assert result.components.shape[1] == int(mask.sum())
    assert result.params["pca_n_vars"] == int(mask.sum())


def test_residual_rejects_n_comps_beyond_the_nonempty_column_count(
    counts: sparse.csr_matrix,
) -> None:
    """Empty genes cannot buy components that carry no variance."""
    with_zero = _with_zero_gene(counts)

    with pytest.raises(ValueError, match="identically zero and carry no variance"):
        scp.residual_pca_matrix(with_zero, n_comps=counts.shape[1])


@pytest.mark.parametrize("clip_mode", ["upper", "symmetric"])
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
def test_clipped_residuals_tolerate_an_empty_gene(
    counts: sparse.csr_matrix,
    model: str,
    residual: str,
    alpha: np.ndarray | None,
    clip_mode: str,
) -> None:
    """Clipping an empty gene's zero column leaves every retained value alone."""
    with_zero = _with_zero_gene(counts)
    full_alpha = None if alpha is None else np.append(alpha, 0.3)
    kwargs = dict(
        n_comps=2,
        model=model,
        residual=residual,
        clip=1.5,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=None,
        dtype="float64",
    )

    expected = scp.residual_pca_matrix(counts, alpha=alpha, **kwargs)
    actual = scp.residual_pca_matrix(with_zero, alpha=full_alpha, **kwargs)

    np.testing.assert_allclose(
        actual.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(actual.components[:, -1], 0.0, rtol=0.0, atol=1e-12)


def test_residual_ignores_overdispersion_of_an_empty_gene(
    counts: sparse.csr_matrix,
) -> None:
    """A NaN scaled-NB estimate on an empty gene cannot reach any output."""
    with_zero = _with_zero_gene(counts)
    alpha = np.full(with_zero.shape[1], 0.1)
    expected = scp.residual_pca_matrix(
        with_zero, n_comps=2, model="scaled_nb", alpha=alpha, dtype="float64"
    )
    alpha[-1] = np.nan

    result = scp.residual_pca_matrix(
        with_zero, n_comps=2, model="scaled_nb", alpha=alpha, dtype="float64"
    )

    np.testing.assert_allclose(
        result.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    assert np.isfinite(result.components).all()

    # A retained gene's overdispersion is still validated, as is array shape.
    alpha[0] = np.nan
    with pytest.raises(ValueError, match="alpha must contain only finite values"):
        scp.residual_pca_matrix(with_zero, n_comps=2, model="scaled_nb", alpha=alpha)
    with pytest.raises(ValueError, match="alpha must be a scalar or have shape"):
        scp.residual_pca_matrix(
            with_zero, n_comps=2, model="scaled_nb", alpha=np.full(3, 0.1)
        )


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
    assert result.params["n_empty_vars"] == 1


def test_shifted_clr_reports_the_count_behind_its_centering_scale(
    counts: sparse.csr_matrix,
) -> None:
    """n_empty_vars recovers the factor by which empty genes scale CLR centering."""
    n_empty = 5
    padded = sparse.hstack(
        (counts, sparse.csr_matrix((counts.shape[0], n_empty), dtype=counts.dtype)),
        format="csr",
    )
    method = scp.ShiftedCLR(count_shift=0.7)
    transformed = scp.transform(padded, method)
    params = transformed.params

    assert params["n_empty_vars"] == n_empty

    n_vars = params["normalization_n_vars"]
    scale = (n_vars - params["n_empty_vars"]) / n_vars
    logged = np.log(counts.toarray().astype(np.float64) + 0.7)
    expected = logged - scale * logged.mean(axis=1, keepdims=True)

    actual = transformed.materialize()[:, : counts.shape[1]]
    np.testing.assert_allclose(
        actual - actual.mean(axis=0),
        expected - expected.mean(axis=0),
        rtol=0.0,
        atol=1e-12,
    )


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
