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
    build_dirichlet_clr_representation,
    build_dirichlet_log_representation,
)
from sparse_count_pca._operator import SparseLowRankLinearOperator


def _dense_dirichlet(X, concentration, prior_proportions, *, clr):
    X = np.asarray(X, dtype=np.float64)
    if prior_proportions is None:
        prior_proportions = np.full(X.shape[1], 1.0 / X.shape[1])
    posterior = (X + concentration * prior_proportions) / (
        X.sum(axis=1)[:, None] + concentration
    )
    transformed = np.log(posterior)
    if clr:
        transformed -= transformed.mean(axis=1)[:, None]
    return transformed


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
    expected = _dense_dirichlet(
        counts.toarray(), concentration, prior_proportions, clr=clr
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 2
    if clr:
        np.testing.assert_allclose(actual.mean(axis=1), 0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("pca", "clr", "transform"),
    [
        (dirichlet_log_pca_matrix, False, "dirichlet_log"),
        (dirichlet_clr_pca_matrix, True, "dirichlet_clr"),
    ],
)
def test_dirichlet_pca_matches_dense_svd(counts, pca, clr, transform):
    concentration = 3.0
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    result = pca(
        counts,
        n_comps=2,
        concentration=concentration,
        prior_proportions=prior,
        return_operator=True,
    )
    dense = _dense_dirichlet(counts.toarray(), concentration, prior, clr=clr)
    dense -= dense.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)

    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-12
    )
    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=1e-10)
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
    mask = np.array([True, True, True, False])
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    dense = _dense_dirichlet(adata.X.toarray(), 2.0, prior, clr=clr)[:, mask]
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
    )
    assert result.uns["dirichlet"]["params"]["normalization_n_vars"] == adata.n_vars
    assert np.isnan(result.varm["dirichlet"][~mask]).all()


def test_dirichlet_anndata_resolves_prior_proportions_from_var(adata):
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
        result.uns["dirichlet"]["singular_values"], expected.singular_values
    )
    assert result.uns["dirichlet"]["params"]["prior_proportions"] == "prior"


@pytest.mark.parametrize(
    "pca",
    [dirichlet_log_pca_matrix, dirichlet_clr_pca_matrix],
)
def test_dirichlet_allows_zero_count_rows(pca, counts):
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    result = pca(with_empty, n_comps=2, return_operator=True)
    assert np.isfinite(result.singular_values).all()
    assert result.operator.shape == with_empty.shape


@pytest.mark.parametrize("concentration", [0.0, -1.0, np.inf, np.nan])
def test_dirichlet_rejects_invalid_concentration(counts, concentration):
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
    with pytest.raises(ValueError, match=message):
        dirichlet_clr_pca_matrix(counts, n_comps=2, prior_proportions=prior)
