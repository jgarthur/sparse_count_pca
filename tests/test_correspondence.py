"""Tests for correspondence-analysis computations and AnnData behavior."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from sparse_count_pca import (
    correspondence_analysis,
    correspondence_analysis_matrix,
)
from tests._oracles import _dense_correspondence


def test_correspondence_analysis_matches_dense_oracle(counts):
    """Correspondence analysis matches an independent dense oracle."""
    expected, row_masses, column_masses = _dense_correspondence(counts)
    result = correspondence_analysis_matrix(
        counts, n_comps=2, dtype="float64", return_operator=True
    )
    U, singular_values, Vt = np.linalg.svd(expected, full_matrices=False)

    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]),
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.singular_values, singular_values[:2], rtol=1e-10, atol=0.0
    )
    np.testing.assert_allclose(result.row_masses, row_masses, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        result.column_masses, column_masses, rtol=1e-14, atol=0.0
    )
    assert not hasattr(result, "row_standard_coordinates")
    assert not hasattr(result, "column_standard_coordinates")
    np.testing.assert_allclose(
        np.abs(result.row_principal_coordinates),
        np.abs(U[:, :2] * singular_values[:2] / np.sqrt(row_masses)[:, None]),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.abs(result.column_principal_coordinates),
        np.abs(Vt[:2].T * singular_values[:2] / np.sqrt(column_masses)[:, None]),
        rtol=1e-9,
        atol=1e-9,
    )


def test_total_inertia_equals_pearson_chi_squared_over_total(counts):
    """Classical total inertia equals Pearson chi-squared over grand total."""
    X = counts.toarray().astype(float)
    total = X.sum()
    expected_counts = np.outer(X.sum(axis=1), X.sum(axis=0)) / total
    chi_squared = np.sum((X - expected_counts) ** 2 / expected_counts)
    result = correspondence_analysis_matrix(counts, n_comps=2)
    assert result.total_inertia == pytest.approx(
        chi_squared / total, rel=1e-12, abs=0.0
    )
    np.testing.assert_allclose(
        result.inertia_ratio,
        result.principal_inertias / result.total_inertia,
        rtol=0.0,
        atol=0.0,
    )


def test_experimental_scaled_nb_ca_matches_dense_residual_ordination(counts):
    """Scaled-NB correspondence analysis matches dense residual ordination."""
    alpha = np.array([0.0, 0.05, 0.2, 1.0])
    expected, _, _ = _dense_correspondence(counts, model="scaled_nb", alpha=alpha)
    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        result = correspondence_analysis_matrix(
            counts,
            n_comps=2,
            model="scaled_nb",
            alpha=alpha,
            return_operator=True,
        )

    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.total_inertia == pytest.approx(
        np.sum(expected**2), rel=1e-12, abs=0.0
    )
    assert result.params["model"] == "scaled_nb"
    assert result.params["experimental"] is True
    assert (
        result.params["inertia_interpretation"] == "scaled_nb_pearson_residual_inertia"
    )


def test_correspondence_model_validates_alpha(counts):
    """Correspondence models enforce their alpha parameter contracts."""
    with pytest.raises(ValueError, match="alpha is required"):
        correspondence_analysis_matrix(counts, model="scaled_nb")
    with pytest.raises(ValueError, match="alpha is only used"):
        correspondence_analysis_matrix(counts, model="poisson", alpha=0.1)
    with pytest.raises(ValueError, match="model must be"):
        correspondence_analysis_matrix(counts, model="binomial")


def test_ca_row_principal_coordinates_equal_unscaled_pearson_scores_over_sqrt_depth(
    counts,
):
    """Row principal coordinates use Pearson scores scaled by row depth."""
    X = counts.toarray().astype(float)
    row_totals = X.sum(axis=1)
    expected = np.outer(row_totals, X.sum(axis=0) / X.sum())
    pearson = (X - expected) / np.sqrt(expected)
    U, singular_values, _ = np.linalg.svd(pearson, full_matrices=False)
    expected_coordinates = U[:, :2] * singular_values[:2] / np.sqrt(row_totals)[:, None]
    result = correspondence_analysis_matrix(counts, n_comps=2)
    np.testing.assert_allclose(
        np.abs(result.row_principal_coordinates),
        np.abs(expected_coordinates),
        rtol=1e-9,
        atol=1e-9,
    )


def test_correspondence_coordinates_have_weighted_zero_centroids(counts):
    """Correspondence coordinates have mass-weighted zero centroids."""
    result = correspondence_analysis_matrix(counts, n_comps=2)
    np.testing.assert_allclose(
        result.row_masses @ result.row_principal_coordinates,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.column_masses @ result.column_principal_coordinates,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_ca_mask_recomputes_selected_table_margins(adata):
    """Correspondence masking recomputes margins on the selected table."""
    mask = np.array([True, True, True, False])
    expected = correspondence_analysis_matrix(adata.X[:, mask], n_comps=2)
    result = correspondence_analysis(
        adata, n_comps=2, mask_var=mask, copy=True, key_added="coords"
    )
    np.testing.assert_allclose(
        result.obsm["coords"],
        expected.row_principal_coordinates,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isnan(result.varm["coords"][~mask]).all()
    np.testing.assert_allclose(
        result.uns["coords"]["column_masses"],
        expected.column_masses,
        rtol=0.0,
        atol=1e-14,
    )


def test_experimental_scaled_nb_ca_resolves_anndata_alpha_after_masking(adata):
    """Scaled-NB AnnData alpha values remain aligned after CA masking."""
    mask = np.array([True, True, True, False])
    alpha = np.array([0.0, 0.05, 0.2, 1.0])
    adata.var["overdispersion"] = alpha
    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        expected = correspondence_analysis_matrix(
            adata.X[:, mask],
            n_comps=2,
            model="scaled_nb",
            alpha=alpha[mask],
        )
    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        result = correspondence_analysis(
            adata,
            n_comps=2,
            mask_var=mask,
            model="scaled_nb",
            alpha="overdispersion",
            copy=True,
            key_added="nb_coords",
        )

    np.testing.assert_allclose(
        result.obsm["nb_coords"],
        expected.row_principal_coordinates,
        rtol=0.0,
        atol=1e-12,
    )
    assert result.uns["nb_coords"]["params"]["alpha"] == "overdispersion"
    assert result.uns["nb_coords"]["params"]["experimental"] is True


def test_ca_rejects_zero_mass_rows():
    """Correspondence analysis rejects rows with zero mass."""
    with pytest.raises(ValueError, match="zero total counts"):
        correspondence_analysis_matrix(
            sparse.csr_matrix([[1, 2, 3], [0, 0, 0], [2, 1, 3]]),
            n_comps=2,
        )


def test_ca_keeps_zero_mass_columns_and_reports_undefined_coordinates():
    """A zero-mass column stays in the table with NaN principal coordinates."""
    values = [[1, 3], [2, 1], [3, 2]]
    padded = sparse.csr_matrix([[row[0], 0, row[1]] for row in values])

    result = correspondence_analysis_matrix(padded, n_comps=1)
    expected = correspondence_analysis_matrix(sparse.csr_matrix(values), n_comps=1)

    # The analyzed table keeps its width; only the rescaled coordinate of the
    # zero-mass column is undefined.
    assert result.column_principal_coordinates.shape[0] == 3
    assert np.isnan(result.column_principal_coordinates[1]).all()
    assert np.isfinite(result.column_principal_coordinates[[0, 2]]).all()
    assert result.column_masses[1] == 0.0
    assert result.params["n_empty_vars"] == 1

    np.testing.assert_allclose(
        result.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.row_principal_coordinates,
        expected.row_principal_coordinates,
        rtol=0.0,
        atol=1e-12,
    )


def test_ca_ignores_overdispersion_of_a_zero_mass_column():
    """A non-finite scaled-NB estimate on a zero-mass column cannot reach output."""
    values = np.array([[5, 0, 3, 2], [2, 0, 1, 4], [3, 0, 6, 1], [1, 0, 2, 5]])
    X = sparse.csr_matrix(values.astype(np.float64))
    alpha = np.full(4, 0.1)
    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        expected = correspondence_analysis_matrix(
            X, n_comps=2, model="scaled_nb", alpha=alpha
        )
    alpha[1] = np.nan

    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        result = correspondence_analysis_matrix(
            X, n_comps=2, model="scaled_nb", alpha=alpha
        )

    np.testing.assert_allclose(
        result.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    assert np.isfinite(result.row_principal_coordinates).all()

    # A column that carries mass is still validated, as is array shape.
    alpha[0] = np.nan
    with pytest.raises(ValueError, match="alpha must contain only finite values"):
        correspondence_analysis_matrix(X, n_comps=2, model="scaled_nb", alpha=alpha)
    with pytest.raises(ValueError, match="alpha must be a scalar or have shape"):
        correspondence_analysis_matrix(
            X, n_comps=2, model="scaled_nb", alpha=np.full(3, 0.1)
        )


def test_ca_validates_alpha_of_a_masked_out_empty_column():
    """The alpha exception is scoped to the analyzed table, not the input."""
    values = np.array([[5, 0, 3, 2], [2, 0, 1, 4], [3, 0, 6, 1], [1, 0, 2, 5]])
    adata = AnnData(sparse.csr_matrix(values.astype(np.float64)))
    alpha = np.full(4, 0.1)
    alpha[1] = np.nan
    mask = np.array([True, False, True, True])

    # Column 1 is empty, but the mask removes it from the analyzed table, so it
    # is not covered by the exception and its value is still rejected.
    with pytest.raises(ValueError, match="alpha must contain only finite values"):
        correspondence_analysis(
            adata, n_comps=2, model="scaled_nb", alpha=alpha, mask_var=mask
        )


def test_ca_writes_nan_varm_for_a_zero_mass_gene():
    """A zero-mass column receives NaN principal coordinates in varm."""
    adata = AnnData(sparse.csr_matrix([[1, 0, 3], [2, 0, 1], [3, 0, 2]]))

    result = correspondence_analysis(adata, n_comps=1, copy=True)

    assert np.isnan(result.varm["CA"][1]).all()
    assert np.isfinite(result.varm["CA"][[0, 2]]).all()
    assert result.uns["ca"]["params"]["n_empty_vars"] == 1


def test_ca_rejects_rows_emptied_by_variable_mask():
    """CA rejects a nonempty input row that has no mass after masking."""
    X = sparse.csr_matrix([[1, 0, 3], [0, 2, 0], [2, 1, 3]])
    mask = np.array([True, False, True])
    adata = AnnData(X)

    with pytest.raises(ValueError, match="Rows with zero mass"):
        correspondence_analysis(adata, n_comps=1, mask_var=mask)


def test_ca_rejects_independence_table_with_zero_inertia():
    """Correspondence analysis rejects independent tables with zero inertia."""
    X = sparse.csr_matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    with pytest.raises(ValueError, match="zero inertia"):
        correspondence_analysis_matrix(X, n_comps=2)
