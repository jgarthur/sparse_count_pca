import numpy as np
import pytest

from sparse_residual_pca import shifted_clr_pca, shifted_clr_pca_matrix
from sparse_residual_pca._operator import SparseLowRankLinearOperator
from sparse_residual_pca._shifted_clr import build_shifted_clr_representation
from tests.shifted_clr_reference.oracle import norm_clr


@pytest.mark.parametrize("pseudocount", [0.25, 1.0, 2.0])
def test_shifted_clr_representation_matches_paper_formula(counts, pseudocount):
    representation = build_shifted_clr_representation(counts, pseudocount=pseudocount)
    operator = SparseLowRankLinearOperator(
        representation, center=False, dtype="float64"
    )
    actual = operator @ np.eye(counts.shape[1])
    expected = norm_clr(counts, pseudocount)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, atol=1e-15)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 1


def test_shifted_clr_pca_matches_dense_svd(counts):
    result = shifted_clr_pca_matrix(
        counts, n_comps=2, dtype="float64", return_operator=True
    )
    dense = norm_clr(counts)
    dense -= dense.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-12
    )
    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=1e-10)
    np.testing.assert_allclose(
        np.abs(result.components), np.abs(Vt[:2]), rtol=1e-9, atol=1e-9
    )
    assert result.params["transform"] == "shifted_clr"
    assert result.params["pseudocount"] == 1.0


def test_shifted_clr_mask_is_applied_after_full_feature_normalization(adata):
    mask = np.array([True, True, True, False])
    dense = norm_clr(adata.X)[:, mask]
    dense -= dense.mean(axis=0)
    result = shifted_clr_pca(
        adata,
        n_comps=2,
        mask_var=mask,
        key_added="shifted_clr",
        copy=True,
    )
    _, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.uns["shifted_clr"]["singular_values"],
        singular_values[:2],
        rtol=1e-10,
    )
    assert result.uns["shifted_clr"]["params"]["normalization_n_vars"] == adata.n_vars
    assert np.isnan(result.varm["shifted_clr"][~mask]).all()


@pytest.mark.parametrize("pseudocount", [0.0, -1.0, np.inf, np.nan])
def test_shifted_clr_rejects_invalid_pseudocount(counts, pseudocount):
    with pytest.raises(ValueError, match="finite and positive"):
        shifted_clr_pca_matrix(counts, n_comps=2, pseudocount=pseudocount)
