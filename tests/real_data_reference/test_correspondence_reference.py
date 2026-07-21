"""Real-data dense oracles for correspondence-analysis representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import correspondence_analysis_matrix

REFERENCE_DIR = Path(__file__).resolve().parent
COUNTS_PATH = REFERENCE_DIR / "pbmc3k_raw_counts.npz"


def _load_counts() -> sparse.csr_matrix:
    """Load the pinned unequal-depth PBMC3k-derived matrix."""
    return sparse.load_npz(COUNTS_PATH).tocsr()


def _dense_classical_correspondence(
    counts: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the classical standardized residual matrix from probabilities."""
    observed = counts.toarray().astype(np.float64)
    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    grand_total = row_totals.sum(dtype=np.float64)
    probabilities = observed / grand_total
    row_masses = row_totals / grand_total
    column_masses = column_totals / grand_total
    independence = np.outer(row_masses, column_masses)
    standardized = (probabilities - independence) / np.sqrt(independence)
    return standardized, row_masses, column_masses


def _dense_scaled_nb_correspondence(
    counts: sparse.csr_matrix, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the package's experimental scaled-NB CA matrix independently."""
    observed = counts.toarray().astype(np.float64)
    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    grand_total = row_totals.sum(dtype=np.float64)
    row_masses = row_totals / grand_total
    column_masses = column_totals / grand_total
    expected = np.outer(row_totals, column_masses)
    mean_depth = row_totals.mean(dtype=np.float64)
    variance_scale = 1.0 + alpha * mean_depth * column_masses
    residuals = (observed - expected) / np.sqrt(expected * variance_scale[None, :])
    return residuals / np.sqrt(grand_total), row_masses, column_masses


def _materialize_public_operator(result, n_vars: int) -> np.ndarray:
    """Materialize a returned public operator on the complete standard basis."""
    assert result.operator is not None
    return result.operator @ np.eye(n_vars, dtype=np.float64)


def test_real_data_classical_correspondence_matches_full_dense_formula() -> None:
    """Classical CA represents every PBMC3k standardized residual exactly."""
    counts = _load_counts()
    expected, row_masses, column_masses = _dense_classical_correspondence(counts)
    result = correspondence_analysis_matrix(
        counts,
        n_comps=2,
        dtype="float64",
        random_state=0,
        return_operator=True,
    )
    actual = _materialize_public_operator(result, counts.shape[1])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.row_masses, row_masses, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.column_masses, column_masses, rtol=0.0, atol=0.0)
    assert result.total_inertia == pytest.approx(
        np.sum(expected * expected, dtype=np.float64), rel=1e-12, abs=0.0
    )
    assert result.params["zero_center"] is False


def test_real_data_scaled_nb_correspondence_matches_full_dense_formula() -> None:
    """Experimental scaled-NB CA represents every PBMC3k residual exactly."""
    counts = _load_counts()
    alpha = 0.01
    expected, row_masses, column_masses = _dense_scaled_nb_correspondence(counts, alpha)
    with pytest.warns(UserWarning, match="no classical Pearson chi-square"):
        result = correspondence_analysis_matrix(
            counts,
            n_comps=2,
            model="scaled_nb",
            alpha=alpha,
            dtype="float64",
            random_state=0,
            return_operator=True,
        )
    actual = _materialize_public_operator(result, counts.shape[1])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.row_masses, row_masses, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.column_masses, column_masses, rtol=0.0, atol=0.0)
    assert result.total_inertia == pytest.approx(
        np.sum(expected * expected, dtype=np.float64), rel=1e-12, abs=0.0
    )
    assert result.params["alpha"] == alpha
    assert result.params["experimental"] is True
    assert (
        result.params["inertia_interpretation"] == "scaled_nb_pearson_residual_inertia"
    )
