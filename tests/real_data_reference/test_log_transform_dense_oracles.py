"""Real-data dense-oracle tests for logarithmic count transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp

REFERENCE_DIR = Path(__file__).resolve().parent
RAW_COUNTS_PATH = REFERENCE_DIR / "pbmc3k_raw_counts.npz"
EQUAL_DEPTH_COUNTS_PATH = REFERENCE_DIR / "pbmc3k_equal_depth_counts.npz"

COUNT_SHIFT = 0.75
COMPOSITION_SHIFT = 0.00075
DIRICHLET_CONCENTRATION = 12.5
ORACLE_ATOL = 2e-14


@pytest.fixture(scope="module")
def raw_counts() -> sparse.csr_matrix:
    """Load the pinned unequal-depth PBMC3k-derived counts."""
    return sparse.load_npz(RAW_COUNTS_PATH).tocsr()


@pytest.fixture(scope="module")
def equal_depth_counts() -> sparse.csr_matrix:
    """Load the pinned PBMC3k-derived counts with 1,000 reads per cell."""
    return sparse.load_npz(EQUAL_DEPTH_COUNTS_PATH).tocsr()


def _dense_counts(counts: sparse.csr_matrix) -> np.ndarray:
    """Convert a sparse count fixture to an independent float64 dense array."""
    return counts.toarray().astype(np.float64)


def _dense_clr(logged: np.ndarray) -> np.ndarray:
    """Center logged compositions independently within every cell."""
    return logged - logged.mean(axis=1, keepdims=True)


def _dense_shifted_log(counts: sparse.csr_matrix, count_shift: float) -> np.ndarray:
    """Evaluate the fixed-count shifted-log definition densely."""
    return np.log1p(_dense_counts(counts) / count_shift)


def _dense_count_shifted_clr(
    counts: sparse.csr_matrix, count_shift: float
) -> np.ndarray:
    """Evaluate the fixed-count shifted-CLR definition densely."""
    return _dense_clr(np.log(_dense_counts(counts) + count_shift))


def _dense_proportion_shifted_clr(
    counts: sparse.csr_matrix, composition_shift: float
) -> np.ndarray:
    """Evaluate the fixed-composition shifted-CLR definition densely."""
    dense = _dense_counts(counts)
    proportions = dense / dense.sum(axis=1, keepdims=True)
    return _dense_clr(np.log(proportions + composition_shift))


def _dense_dirichlet(
    counts: sparse.csr_matrix,
    concentration: float,
    prior_proportions: np.ndarray,
    *,
    clr: bool,
) -> np.ndarray:
    """Evaluate a Dirichlet posterior-mean log transform densely."""
    dense = _dense_counts(counts)
    posterior = (dense + concentration * prior_proportions) / (
        dense.sum(axis=1, keepdims=True) + concentration
    )
    logged = np.log(posterior)
    return _dense_clr(logged) if clr else logged


def _prior_proportions(n_vars: int, kind: str) -> np.ndarray:
    """Return a deterministic uniform or nonuniform prior composition."""
    if kind == "uniform":
        return np.full(n_vars, 1.0 / n_vars, dtype=np.float64)
    weights = np.linspace(1.0, 3.0, n_vars, dtype=np.float64)
    return weights / weights.sum()


def test_shifted_log_matches_dense_oracle_on_unequal_depth_real_data(
    raw_counts: sparse.csr_matrix,
) -> None:
    """Every real-data shifted-log entry matches its dense NumPy formula."""
    actual = scp.transform(
        raw_counts, scp.ShiftedLog(count_shift=COUNT_SHIFT)
    ).materialize()
    expected = _dense_shifted_log(raw_counts, COUNT_SHIFT)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ORACLE_ATOL)


def test_count_shifted_clr_matches_dense_oracle_on_unequal_depth_real_data(
    raw_counts: sparse.csr_matrix,
) -> None:
    """Every real-data count-shifted CLR entry matches its dense formula."""
    actual = scp.transform(
        raw_counts, scp.ShiftedCLR(count_shift=COUNT_SHIFT)
    ).materialize()
    expected = _dense_count_shifted_clr(raw_counts, COUNT_SHIFT)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ORACLE_ATOL)


def test_proportion_shifted_clr_matches_dense_oracle_on_unequal_depth_real_data(
    raw_counts: sparse.csr_matrix,
) -> None:
    """Every real-data proportion-shifted CLR entry matches its dense formula."""
    actual = scp.transform(
        raw_counts,
        scp.ProportionShiftedCLR(composition_shift=COMPOSITION_SHIFT),
    ).materialize()
    expected = _dense_proportion_shifted_clr(raw_counts, COMPOSITION_SHIFT)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ORACLE_ATOL)


def test_equal_depth_count_and_proportion_shifts_match_both_dense_oracles(
    equal_depth_counts: sparse.csr_matrix,
) -> None:
    """Equivalent shifts on 1,000-count cells match both dense CLR formulas."""
    count_actual = scp.transform(
        equal_depth_counts, scp.ShiftedCLR(count_shift=COUNT_SHIFT)
    ).materialize()
    proportion_actual = scp.transform(
        equal_depth_counts,
        scp.ProportionShiftedCLR(composition_shift=COMPOSITION_SHIFT),
    ).materialize()
    count_expected = _dense_count_shifted_clr(equal_depth_counts, COUNT_SHIFT)
    proportion_expected = _dense_proportion_shifted_clr(
        equal_depth_counts, COMPOSITION_SHIFT
    )

    np.testing.assert_allclose(count_actual, count_expected, rtol=0.0, atol=ORACLE_ATOL)
    np.testing.assert_allclose(
        proportion_actual, proportion_expected, rtol=0.0, atol=ORACLE_ATOL
    )
    np.testing.assert_allclose(
        count_actual, proportion_actual, rtol=0.0, atol=ORACLE_ATOL
    )


@pytest.mark.parametrize(
    ("method_type", "clr"),
    [(scp.DirichletLog, False), (scp.DirichletCLR, True)],
    ids=["log", "clr"],
)
@pytest.mark.parametrize("prior_kind", ["uniform", "nonuniform"])
def test_dirichlet_transforms_match_dense_oracles_on_unequal_depth_real_data(
    raw_counts: sparse.csr_matrix,
    method_type: type[scp.DirichletLog] | type[scp.DirichletCLR],
    clr: bool,
    prior_kind: str,
) -> None:
    """Dirichlet log and CLR match dense formulas for two real-data priors."""
    prior = _prior_proportions(raw_counts.shape[1], prior_kind)
    method_prior = None if prior_kind == "uniform" else prior
    actual = scp.transform(
        raw_counts,
        method_type(
            concentration=DIRICHLET_CONCENTRATION,
            prior_proportions=method_prior,
        ),
    ).materialize()
    expected = _dense_dirichlet(
        raw_counts,
        DIRICHLET_CONCENTRATION,
        prior,
        clr=clr,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ORACLE_ATOL)
