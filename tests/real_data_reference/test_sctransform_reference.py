"""Real-data equality tests against pinned SCTransform v2 residuals."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp

REFERENCE_DIR = Path(__file__).resolve().parent
COUNTS_PATH = REFERENCE_DIR / "pbmc3k_equal_depth_counts.npz"
ORACLE_PATH = REFERENCE_DIR / "sctransform_equal_depth_reference.npz"
ORACLE_SHA256 = "0180a25ed0ec7cb42829817d7d8219c0e285313076ad2037bbe68eef8870d86e"


@pytest.fixture(scope="module")
def counts() -> sparse.csr_matrix:
    """Load the pinned equal-depth PBMC3k-derived count matrix."""
    return sparse.load_npz(COUNTS_PATH).tocsr()


@pytest.fixture(scope="module")
def oracle() -> dict[str, np.ndarray]:
    """Load all SCTransform arrays while closing the underlying NPZ archive."""
    with np.load(ORACLE_PATH) as archive:
        return {name: archive[name] for name in archive.files}


def test_sctransform_reference_artifact_has_pinned_checksum() -> None:
    """The external oracle file matches the artifact documented at generation."""
    digest = hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest()

    assert digest == ORACLE_SHA256


def test_sctransform_final_parameters_match_the_shared_equal_depth_model(
    counts, oracle
):
    """Pinned SCT theta and fixed means map exactly to scaled-NB parameters."""
    row_totals = np.asarray(counts.sum(axis=1, dtype=np.int64)).ravel()
    gene_means = np.asarray(counts.mean(axis=0)).ravel()
    theta = oracle["theta"]
    expected_alpha = np.zeros_like(theta)
    finite = np.isfinite(theta)
    expected_alpha[finite] = 1.0 / theta[finite]

    np.testing.assert_array_equal(row_totals, np.full(counts.shape[0], 1000))
    assert finite.sum() == 736
    assert (~finite).sum() == 288
    assert (theta[finite] > 0).all()
    np.testing.assert_allclose(oracle["alpha"], expected_alpha, rtol=0.0, atol=0.0)
    assert oracle["alpha"][oracle["alpha"] > 0].min() > 1e-8
    np.testing.assert_allclose(oracle["fitted_mean"], gene_means, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        oracle["clip"], [np.sqrt(counts.shape[0] / 30)], rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(oracle["clip_counts"], [0.0, 7358.0])


@pytest.mark.parametrize(
    ("reference_name", "use_clip"),
    [
        ("centered_unclipped", False),
        ("centered_clipped", True),
    ],
    ids=["unclipped", "symmetric-clipped"],
)
def test_scaled_nb_pearson_matrix_matches_sctransform_real_data_reference(
    counts, oracle, reference_name, use_clip
):
    """Every centered scaled-NB residual matches SCT v2 on equal-depth data."""
    clip = float(oracle["clip"][0]) if use_clip else None
    transformed = scp.transform(
        counts,
        scp.Residual(
            model="scaled_nb",
            residual="pearson",
            alpha=oracle["alpha"],
            clip=clip,
            clip_mode="symmetric",
            clip_max_nnz_ratio=None,
        ),
    )
    actual = transformed.materialize(block_size=64)
    actual -= actual.mean(axis=0)
    expected = oracle[reference_name]

    assert expected.shape == counts.shape
    np.testing.assert_allclose(
        expected.mean(axis=0), np.zeros(counts.shape[1]), rtol=0.0, atol=3e-15
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    if use_clip:
        # Centering first and clipping second is a distinct transform. This
        # guards the Seurat order encoded by the reference artifact.
        wrong_order = np.clip(oracle["centered_unclipped"], -clip, clip)
        assert not np.allclose(wrong_order, expected, rtol=0.0, atol=1e-6)
