import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import (
    proportion_shifted_clr_pca,
    proportion_shifted_clr_pca_matrix,
    shifted_clr_pca,
    shifted_clr_pca_matrix,
    shifted_log_pca,
    shifted_log_pca_matrix,
)
from sparse_count_pca._log_transforms import (
    build_dirichlet_clr_representation,
    build_proportion_shifted_clr_representation,
    build_shifted_clr_representation,
    build_shifted_log_representation,
)
from sparse_count_pca._operator import SparseLowRankLinearOperator
from tests._oracles import _compare_subspaces
from tests.proportion_shifted_clr_reference.oracle import (
    proportion_shifted_clr,
)
from tests.shifted_clr_reference.oracle import count_shifted_clr, pflog


def _materialize(representation, *, center=False):
    operator = SparseLowRankLinearOperator(
        representation, center=center, dtype="float64"
    )
    return operator @ np.eye(representation.shape[1])


def _shifted_log_dense(counts, count_shift):
    return np.log1p(counts.toarray() / count_shift)


@pytest.mark.parametrize("count_shift", [0.25, 1.0, 2.0])
def test_shifted_log_representation_is_exactly_sparse(counts, count_shift):
    representation = build_shifted_log_representation(counts, count_shift=count_shift)
    actual = _materialize(representation)
    expected = _shifted_log_dense(counts, count_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 0


def test_shifted_log_gauge_has_same_centered_pca_matrix_as_log_counts(counts):
    count_shift = 0.6
    zero_baseline = _shifted_log_dense(counts, count_shift)
    shifted_counts = np.log(counts.toarray() + count_shift)
    zero_baseline -= zero_baseline.mean(axis=0)
    shifted_counts -= shifted_counts.mean(axis=0)
    np.testing.assert_allclose(zero_baseline, shifted_counts, atol=1e-15)


@pytest.mark.parametrize("count_shift", [0.25, 1.0, 2.0])
def test_count_shifted_clr_representation_matches_reference(counts, count_shift):
    representation = build_shifted_clr_representation(counts, count_shift=count_shift)
    actual = _materialize(representation)
    expected = count_shifted_clr(counts, count_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, atol=1e-15)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 1


@pytest.mark.parametrize("composition_shift", [0.25, 1.0, 2.0])
def test_proportion_shifted_clr_matches_historical_reference(counts, composition_shift):
    representation = build_proportion_shifted_clr_representation(
        counts, composition_shift=composition_shift
    )
    actual = _materialize(representation)
    expected = proportion_shifted_clr(counts, composition_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, atol=1e-15)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 1


def test_count_shifted_clr_matches_current_pflog_formula(counts):
    alpha = 0.37
    representation = build_shifted_clr_representation(
        counts, count_shift=1.0 / (4.0 * alpha)
    )
    np.testing.assert_allclose(
        _materialize(representation), pflog(counts, alpha), atol=1e-14
    )


def test_scalar_shifted_clr_equals_uniform_dirichlet_clr(counts):
    count_shift = 0.7
    shifted = build_shifted_clr_representation(counts, count_shift=count_shift)
    dirichlet = build_dirichlet_clr_representation(
        counts,
        concentration=counts.shape[1] * count_shift,
        prior_proportions=None,
    )
    np.testing.assert_allclose(
        _materialize(shifted), _materialize(dirichlet), atol=1e-15
    )


@pytest.mark.parametrize(
    ("pca", "kwargs", "dense_transform"),
    [
        (
            shifted_log_pca_matrix,
            {"count_shift": 0.7},
            lambda X: _shifted_log_dense(X, 0.7),
        ),
        (
            shifted_clr_pca_matrix,
            {"count_shift": 0.7},
            lambda X: count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca_matrix,
            {"composition_shift": 0.7},
            lambda X: proportion_shifted_clr(X, 0.7),
        ),
    ],
)
def test_log_pca_matches_dense_svd(counts, pca, kwargs, dense_transform):
    result = pca(
        counts,
        n_comps=2,
        dtype="float64",
        return_operator=True,
        **kwargs,
    )
    dense = dense_transform(counts)
    dense -= dense.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-12
    )
    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=1e-10)
    assert _compare_subspaces(result.components.T, Vt[:2].T, atol=1e-9)


@pytest.mark.parametrize(
    ("pca", "kwargs", "dense_transform"),
    [
        (
            shifted_log_pca,
            {"count_shift": 0.7},
            lambda X: _shifted_log_dense(X, 0.7),
        ),
        (
            shifted_clr_pca,
            {"count_shift": 0.7},
            lambda X: count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca,
            {"composition_shift": 0.7},
            lambda X: proportion_shifted_clr(X, 0.7),
        ),
    ],
)
def test_log_pca_mask_is_applied_after_full_transform(
    adata, pca, kwargs, dense_transform
):
    mask = np.array([True, True, True, False])
    dense = dense_transform(adata.X)[:, mask]
    dense -= dense.mean(axis=0)
    result = pca(
        adata,
        n_comps=2,
        mask_var=mask,
        key_added="log_pca",
        copy=True,
        **kwargs,
    )
    _, singular_values, _ = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.uns["log_pca"]["singular_values"],
        singular_values[:2],
        rtol=1e-10,
    )
    params = result.uns["log_pca"]["params"]
    assert params["normalization_n_vars"] == adata.n_vars
    assert np.isnan(result.varm["log_pca"][~mask]).all()


def test_count_shifted_transforms_allow_empty_cells(counts):
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    for pca in (shifted_log_pca_matrix, shifted_clr_pca_matrix):
        result = pca(with_empty, n_comps=2, count_shift=1.0)
        assert np.isfinite(result.singular_values).all()


def test_proportion_shifted_clr_rejects_empty_cells(counts):
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    with pytest.raises(ValueError, match="zero total counts"):
        proportion_shifted_clr_pca_matrix(with_empty, n_comps=2, composition_shift=1.0)


def test_proportion_shifted_clr_is_row_scale_invariant(counts):
    scales = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    scaled = sparse.diags(scales) @ counts
    original = build_proportion_shifted_clr_representation(
        counts, composition_shift=0.3
    )
    rescaled = build_proportion_shifted_clr_representation(
        scaled.tocsr(), composition_shift=0.3
    )
    np.testing.assert_allclose(
        _materialize(original), _materialize(rescaled), atol=1e-15
    )


@pytest.mark.parametrize(
    ("pca", "kwargs", "transform", "domain", "parameter"),
    [
        (
            shifted_log_pca_matrix,
            {"count_shift": 0.4},
            "shifted_log",
            "count",
            "count_shift",
        ),
        (
            shifted_clr_pca_matrix,
            {"count_shift": 0.4},
            "shifted_clr",
            "count",
            "count_shift",
        ),
        (
            proportion_shifted_clr_pca_matrix,
            {"composition_shift": 0.4},
            "proportion_shifted_clr",
            "composition",
            "composition_shift",
        ),
    ],
)
def test_log_pca_metadata(counts, pca, kwargs, transform, domain, parameter):
    result = pca(counts, n_comps=2, **kwargs)
    assert result.params["transform"] == transform
    assert result.params["shift_domain"] == domain
    assert result.params[parameter] == 0.4
    assert result.params["normalization_n_vars"] == counts.shape[1]


@pytest.mark.parametrize(
    ("pca", "parameter"),
    [
        (shifted_log_pca_matrix, "count_shift"),
        (shifted_clr_pca_matrix, "count_shift"),
        (proportion_shifted_clr_pca_matrix, "composition_shift"),
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, [1.0], "1"])
def test_log_pca_rejects_invalid_shifts(counts, pca, parameter, value):
    with pytest.raises(ValueError, match="finite and positive"):
        pca(counts, n_comps=2, **{parameter: value})


@pytest.mark.parametrize(
    "pca",
    [
        shifted_log_pca_matrix,
        shifted_clr_pca_matrix,
        proportion_shifted_clr_pca_matrix,
    ],
)
def test_log_pca_requires_explicit_shift(counts, pca):
    with pytest.raises(TypeError):
        pca(counts, n_comps=2)


def test_old_ambiguous_pseudocount_keyword_is_rejected(counts):
    with pytest.raises(TypeError):
        shifted_clr_pca_matrix(counts, n_comps=2, pseudocount=1.0)
