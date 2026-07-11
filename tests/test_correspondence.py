import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import (
    correspondence_analysis,
    correspondence_analysis_matrix,
)


def _dense_ca_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    total = X.sum()
    P = X / total
    row_masses = P.sum(axis=1)
    column_masses = P.sum(axis=0)
    standardized = P - np.outer(row_masses, column_masses)
    standardized /= np.sqrt(np.outer(row_masses, column_masses))
    return standardized, row_masses, column_masses


def test_correspondence_analysis_matches_dense_oracle(counts):
    X = counts.toarray()
    expected, row_masses, column_masses = _dense_ca_matrix(X)
    result = correspondence_analysis_matrix(
        counts, n_comps=2, dtype="float64", return_operator=True
    )
    U, singular_values, Vt = np.linalg.svd(expected, full_matrices=False)

    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), expected, atol=1e-12
    )
    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=1e-10)
    np.testing.assert_allclose(result.row_masses, row_masses)
    np.testing.assert_allclose(result.column_masses, column_masses)
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
    X = counts.toarray().astype(float)
    total = X.sum()
    expected_counts = np.outer(X.sum(axis=1), X.sum(axis=0)) / total
    chi_squared = np.sum((X - expected_counts) ** 2 / expected_counts)
    result = correspondence_analysis_matrix(counts, n_comps=2)
    assert result.total_inertia == pytest.approx(chi_squared / total, rel=1e-12)
    np.testing.assert_allclose(
        result.inertia_ratio,
        result.principal_inertias / result.total_inertia,
    )


def test_ca_row_principal_coordinates_equal_unscaled_pearson_scores_over_sqrt_depth(
    counts,
):
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
    result = correspondence_analysis_matrix(counts, n_comps=2)
    np.testing.assert_allclose(
        result.row_masses @ result.row_principal_coordinates,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.column_masses @ result.column_principal_coordinates,
        0.0,
        atol=1e-12,
    )


def test_ca_mask_recomputes_selected_table_margins(adata):
    mask = np.array([True, True, True, False])
    expected = correspondence_analysis_matrix(adata.X[:, mask], n_comps=2)
    result = correspondence_analysis(
        adata, n_comps=2, mask_var=mask, copy=True, key_added="coords"
    )
    np.testing.assert_allclose(
        result.obsm["coords"], expected.row_principal_coordinates
    )
    assert np.isnan(result.varm["coords"][~mask]).all()
    np.testing.assert_allclose(
        result.uns["coords"]["column_masses"], expected.column_masses
    )


def test_ca_rejects_zero_mass_rows_and_columns():
    with pytest.raises(ValueError, match="Rows with zero mass"):
        correspondence_analysis_matrix(
            sparse.csr_matrix([[1, 2, 3], [0, 0, 0], [2, 1, 3]]),
            n_comps=2,
        )
    with pytest.raises(ValueError, match="Columns with zero mass"):
        correspondence_analysis_matrix(
            sparse.csr_matrix([[1, 0, 3], [2, 0, 1], [3, 0, 2]]),
            n_comps=2,
        )


def test_ca_rejects_independence_table_with_zero_inertia():
    X = sparse.csr_matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    with pytest.raises(ValueError, match="zero inertia"):
        correspondence_analysis_matrix(X, n_comps=2)
