"""Full-matrix residual equality tests on pinned real PBMC3k counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp
from tests._oracles import _materialize_dense_residual

FIXTURE_DIR = Path(__file__).resolve().parent
RAW_COUNTS_PATH = FIXTURE_DIR / "pbmc3k_raw_counts.npz"
SCALED_NB_ALPHA_PATTERN = np.array([0.02, 0.05, 0.1, 0.2])
UPPER_CLIP = 2.5
SYMMETRIC_CLIP = 1.25
DEFAULT_ORACLE_ATOL = 2e-14
# The independent dense NB-deviance formula evaluates a difference of large,
# nearby terms. Its worst real-data cancellation error is about 3.6e-11; the
# production path is separately checked against Decimal at near-mean inputs.
SCALED_NB_DEVIANCE_ATOL = 1e-10


@pytest.fixture(scope="module")
def raw_counts() -> sparse.csr_matrix:
    """Load the pinned unequal-depth PBMC3k-derived count matrix."""
    return sparse.load_npz(RAW_COUNTS_PATH).tocsr()


@pytest.fixture(scope="module")
def scaled_nb_alpha(raw_counts: sparse.csr_matrix) -> np.ndarray:
    """Return explicit positive dispersions that stay away from zero."""
    repetitions = raw_counts.shape[1] // SCALED_NB_ALPHA_PATTERN.size
    assert repetitions * SCALED_NB_ALPHA_PATTERN.size == raw_counts.shape[1]
    return np.tile(SCALED_NB_ALPHA_PATTERN, repetitions)


@pytest.mark.parametrize(
    ("model", "residual", "atol"),
    [
        ("poisson", "pearson", DEFAULT_ORACLE_ATOL),
        ("poisson", "deviance", DEFAULT_ORACLE_ATOL),
        ("binomial", "pearson", DEFAULT_ORACLE_ATOL),
        ("binomial", "deviance", DEFAULT_ORACLE_ATOL),
        ("scaled_nb", "pearson", DEFAULT_ORACLE_ATOL),
        ("scaled_nb", "deviance", SCALED_NB_DEVIANCE_ATOL),
    ],
)
def test_unclipped_residuals_match_dense_oracle_on_real_counts(
    raw_counts, scaled_nb_alpha, model, residual, atol
):
    """Every residual family matches its dense formula over all PBMC entries."""
    alpha = scaled_nb_alpha if model == "scaled_nb" else None
    expected = _materialize_dense_residual(
        raw_counts,
        model=model,
        residual=residual,
        alpha=alpha,
    )

    actual = scp.transform(
        raw_counts,
        scp.Residual(model=model, residual=residual, alpha=alpha),
        dtype="float64",
    ).materialize(block_size=64)

    assert actual.shape == raw_counts.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def test_upper_clipping_matches_dense_oracle_on_real_counts(raw_counts):
    """Upper clipping changes a real positive tail without growing support."""
    unclipped = _materialize_dense_residual(
        raw_counts,
        model="poisson",
        residual="pearson",
    )
    expected = _materialize_dense_residual(
        raw_counts,
        model="poisson",
        residual="pearson",
        clip=UPPER_CLIP,
        clip_mode="upper",
    )
    clipped = unclipped > UPPER_CLIP

    transformed = scp.transform(
        raw_counts,
        scp.Residual(
            model="poisson",
            residual="pearson",
            clip=UPPER_CLIP,
            clip_mode="upper",
            clip_max_nnz_ratio=1.0,
        ),
        dtype="float64",
    )
    actual = transformed.materialize(block_size=64)

    assert np.count_nonzero(clipped) > 1_000
    np.testing.assert_array_equal(expected[clipped], UPPER_CLIP)
    assert transformed._sparse.nnz == raw_counts.nnz
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=DEFAULT_ORACLE_ATOL)


def test_symmetric_clipping_expands_zero_support_and_matches_dense_oracle(
    raw_counts, scaled_nb_alpha
):
    """Symmetric clipping corrects real structural zeros and both residual tails."""
    unclipped = _materialize_dense_residual(
        raw_counts,
        model="scaled_nb",
        residual="deviance",
        alpha=scaled_nb_alpha,
    )
    expected = _materialize_dense_residual(
        raw_counts,
        model="scaled_nb",
        residual="deviance",
        alpha=scaled_nb_alpha,
        clip=SYMMETRIC_CLIP,
        clip_mode="symmetric",
    )
    structural_zero_clipped = (raw_counts.toarray() == 0) & (
        unclipped < -SYMMETRIC_CLIP
    )

    transformed = scp.transform(
        raw_counts,
        scp.Residual(
            model="scaled_nb",
            residual="deviance",
            alpha=scaled_nb_alpha,
            clip=SYMMETRIC_CLIP,
            clip_mode="symmetric",
            clip_max_nnz_ratio=None,
        ),
        dtype="float64",
    )
    actual = transformed.materialize(block_size=64)

    added_support = np.count_nonzero(structural_zero_clipped)
    assert added_support > 1_000
    assert np.count_nonzero(unclipped > SYMMETRIC_CLIP) > 1_000
    np.testing.assert_array_equal(expected[structural_zero_clipped], -SYMMETRIC_CLIP)
    assert transformed._sparse.nnz == raw_counts.nnz + added_support
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=SCALED_NB_DEVIANCE_ATOL)
