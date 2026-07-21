"""Tests for Dirichlet log and CLR representations and PCA."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import (
    dirichlet_clr_pca,
    dirichlet_clr_pca_matrix,
    dirichlet_log_pca,
    dirichlet_log_pca_matrix,
)
from sparse_count_pca._log_transforms import (
    PRIOR_SUM_ATOL,
    PRIOR_SUM_RTOL,
    build_dirichlet_clr_representation,
    build_dirichlet_log_representation,
    validate_dirichlet_prior,
)
from sparse_count_pca._operator import SparseLowRankLinearOperator
from tests._oracles import _dense_dirichlet


@pytest.mark.parametrize(
    "prior_proportions",
    [None, np.array([0.1, 0.2, 0.3, 0.4])],
)
@pytest.mark.parametrize(
    ("builder", "clr"),
    [
        (build_dirichlet_log_representation, False),
        (build_dirichlet_clr_representation, True),
    ],
)
def test_dirichlet_representations_match_dense_oracle(
    counts, prior_proportions, builder, clr
):
    """Dirichlet representations match independent dense formulas."""
    concentration = 2.5
    representation = builder(
        counts,
        concentration=concentration,
        prior_proportions=prior_proportions,
    )
    operator = SparseLowRankLinearOperator(
        representation, center=False, dtype="float64"
    )
    actual = operator @ np.eye(counts.shape[1])
    expected = _dense_dirichlet(counts, concentration, prior_proportions, clr=clr)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 2
    if clr:
        np.testing.assert_allclose(actual.mean(axis=1), 0.0, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("pca", "clr", "transform"),
    [
        (dirichlet_log_pca_matrix, False, "dirichlet_log"),
        (dirichlet_clr_pca_matrix, True, "dirichlet_clr"),
    ],
)
def test_dirichlet_pca_matches_dense_svd(counts, pca, clr, transform):
    """Dirichlet PCA matches a dense singular-value decomposition."""
    concentration = 3.0
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    result = pca(
        counts,
        n_comps=2,
        concentration=concentration,
        prior_proportions=prior,
        return_operator=True,
    )
    dense = _dense_dirichlet(counts, concentration, prior, clr=clr)
    dense -= dense.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)

    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]),
        dense,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.singular_values, singular_values[:2], rtol=1e-10, atol=0.0
    )
    np.testing.assert_allclose(
        np.abs(result.components), np.abs(Vt[:2]), rtol=1e-9, atol=1e-9
    )
    assert result.params["transform"] == transform
    assert result.params["shift_domain"] == "dirichlet_prior_counts"
    assert result.params["concentration"] == concentration
    np.testing.assert_array_equal(result.params["prior_proportions"], prior)


@pytest.mark.parametrize(
    ("pca", "clr"),
    [
        (dirichlet_log_pca, False),
        (dirichlet_clr_pca, True),
    ],
)
def test_dirichlet_mask_is_applied_after_full_normalization(adata, pca, clr):
    """Dirichlet normalization uses all genes before PCA masking."""
    mask = np.array([True, True, True, False])
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    dense = _dense_dirichlet(adata.X, 2.0, prior, clr=clr)[:, mask]
    dense -= dense.mean(axis=0)
    result = pca(
        adata,
        n_comps=2,
        mask_var=mask,
        key_added="dirichlet",
        concentration=2.0,
        prior_proportions=prior,
        copy=True,
    )
    _, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.uns["dirichlet"]["singular_values"],
        singular_values[:2],
        rtol=1e-10,
        atol=0.0,
    )
    params = result.uns["dirichlet"]["params"]
    assert params["normalization_n_vars"] == adata.n_vars
    assert params["pca_n_vars"] == int(mask.sum())
    assert np.isnan(result.varm["dirichlet"][~mask]).all()


def test_dirichlet_anndata_resolves_prior_proportions_from_var(adata):
    """AnnData APIs resolve prior proportions from variable metadata."""
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    adata.var["prior"] = prior
    result = dirichlet_clr_pca(
        adata,
        n_comps=2,
        prior_proportions="prior",
        key_added="dirichlet",
        copy=True,
    )
    expected = dirichlet_clr_pca_matrix(adata.X, n_comps=2, prior_proportions=prior)
    np.testing.assert_allclose(
        result.uns["dirichlet"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    assert result.uns["dirichlet"]["params"]["prior_proportions"] == "prior"


@pytest.mark.parametrize(
    "pca",
    [dirichlet_log_pca_matrix, dirichlet_clr_pca_matrix],
)
def test_dirichlet_rejects_zero_count_rows(pca, counts):
    """Dirichlet transforms reject rows with zero total counts."""
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    with pytest.raises(ValueError, match="zero total counts"):
        pca(with_empty, n_comps=2, return_operator=True)


@pytest.mark.parametrize(
    "builder",
    [build_dirichlet_log_representation, build_dirichlet_clr_representation],
)
def test_dirichlet_transforms_are_invariant_to_joint_global_scaling(counts, builder):
    """Scaling counts and total prior concentration preserves values."""
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    original = builder(counts, concentration=2.0, prior_proportions=prior)
    rescaled = builder(counts * 3, concentration=6.0, prior_proportions=prior)
    original_values = SparseLowRankLinearOperator(
        original, center=False, dtype="float64"
    ) @ np.eye(counts.shape[1])
    rescaled_values = SparseLowRankLinearOperator(
        rescaled, center=False, dtype="float64"
    ) @ np.eye(counts.shape[1])

    np.testing.assert_allclose(
        original_values,
        rescaled_values,
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("concentration", [0.0, -1.0, np.inf, np.nan])
def test_dirichlet_rejects_invalid_concentration(counts, concentration):
    """Dirichlet transforms reject nonpositive or nonfinite concentration."""
    with pytest.raises(ValueError, match="finite and positive"):
        dirichlet_log_pca_matrix(counts, n_comps=2, concentration=concentration)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ([0.2, 0.3, 0.5], "shape"),
        ([0.1, 0.2, 0.3, np.nan], "finite"),
        ([0.1, 0.2, 0.7, 0.0], "strictly positive"),
        ([0.1, 0.2, 0.3, 0.5], "sum to one"),
    ],
)
def test_dirichlet_rejects_invalid_prior(counts, prior, message):
    """Dirichlet transforms validate prior shape, values, and normalization."""
    with pytest.raises(ValueError, match=message):
        dirichlet_clr_pca_matrix(counts, n_comps=2, prior_proportions=prior)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_dirichlet_prior_sum_tolerance_has_explicit_boundary(direction):
    """Prior sums pass within and fail outside the documented tolerance."""
    tolerance = PRIOR_SUM_ATOL + PRIOR_SUM_RTOL
    inside = np.array([1.0 + direction * 0.5 * tolerance])
    outside = np.array([1.0 + direction * 2.0 * tolerance])

    _, normalized = validate_dirichlet_prior(1.0, inside, n_vars=1)
    np.testing.assert_array_equal(normalized, np.array([1.0]))
    with pytest.raises(ValueError, match="sum to one"):
        validate_dirichlet_prior(1.0, outside, n_vars=1)


def test_dirichlet_rejects_unrepresentable_prior_counts(counts):
    """A positive concentration that underflows per-gene priors fails clearly."""
    tiny = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="prior counts must be representable"):
        dirichlet_clr_pca_matrix(counts, n_comps=2, concentration=tiny)
