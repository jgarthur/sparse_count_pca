"""Tests for shifted-log and shifted-CLR representations and PCA."""

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
from tests._oracles import (
    _compare_subspaces,
    _dense_count_shifted_clr,
    _dense_proportion_shifted_clr,
    _dense_shifted_log,
)
from tests.shifted_clr_reference.oracle import pflog


def _materialize(representation, *, center=False):
    operator = SparseLowRankLinearOperator(
        representation, center=center, dtype="float64"
    )
    return operator @ np.eye(representation.shape[1])


@pytest.mark.parametrize("count_shift", [0.25, 1.0, 2.0])
def test_shifted_log_representation_is_exactly_sparse(counts, count_shift):
    """The fixed-count shifted-log representation is exactly sparse."""
    representation = build_shifted_log_representation(counts, count_shift=count_shift)
    actual = _materialize(representation)
    expected = _dense_shifted_log(counts, count_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 0


def test_shifted_log_gauge_has_same_centered_pca_matrix_as_log_counts(counts):
    """The sparse shifted-log gauge preserves the centered log-count matrix."""
    count_shift = 0.6
    zero_baseline = _dense_shifted_log(counts, count_shift)
    shifted_counts = np.log(counts.toarray() + count_shift)
    zero_baseline -= zero_baseline.mean(axis=0)
    shifted_counts -= shifted_counts.mean(axis=0)
    np.testing.assert_allclose(zero_baseline, shifted_counts, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("count_shift", [0.25, 1.0, 2.0])
def test_count_shifted_clr_representation_matches_reference(counts, count_shift):
    """Count-shifted CLR values match the independent reference formula."""
    representation = build_shifted_clr_representation(counts, count_shift=count_shift)
    actual = _materialize(representation)
    expected = _dense_count_shifted_clr(counts, count_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, rtol=0.0, atol=1e-15)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 1


@pytest.mark.parametrize("composition_shift", [0.25, 1.0, 2.0])
def test_proportion_shifted_clr_matches_historical_reference(counts, composition_shift):
    """Values match the pinned June 10, 2026 BHGP formula reference."""
    representation = build_proportion_shifted_clr_representation(
        counts, composition_shift=composition_shift
    )
    actual = _materialize(representation)
    expected = _dense_proportion_shifted_clr(counts, composition_shift)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.mean(axis=1), 0.0, rtol=0.0, atol=1e-15)
    assert representation.sparse.nnz == counts.nnz
    assert representation.rank == 1


def test_count_shifted_clr_matches_current_pflog_formula(counts):
    """Values match the pinned June 24, 2026 BHGP PFlog formula."""
    alpha = 0.37
    representation = build_shifted_clr_representation(
        counts, count_shift=1.0 / (4.0 * alpha)
    )
    np.testing.assert_allclose(
        _materialize(representation),
        pflog(counts, alpha),
        rtol=0.0,
        atol=1e-14,
    )


def test_scalar_shifted_clr_equals_uniform_dirichlet_clr(counts):
    """Scalar shifted CLR equals CLR under a uniform Dirichlet prior."""
    count_shift = 0.7
    shifted = build_shifted_clr_representation(counts, count_shift=count_shift)
    dirichlet = build_dirichlet_clr_representation(
        counts,
        concentration=counts.shape[1] * count_shift,
        prior_proportions=None,
    )
    np.testing.assert_allclose(
        _materialize(shifted),
        _materialize(dirichlet),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("pca", "kwargs", "dense_transform"),
    [
        (
            shifted_log_pca_matrix,
            {"count_shift": 0.7},
            lambda X: _dense_shifted_log(X, 0.7),
        ),
        (
            shifted_clr_pca_matrix,
            {"count_shift": 0.7},
            lambda X: _dense_count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca_matrix,
            {"composition_shift": 0.7},
            lambda X: _dense_proportion_shifted_clr(X, 0.7),
        ),
    ],
)
def test_log_pca_matches_dense_svd(counts, pca, kwargs, dense_transform):
    """Every logarithmic PCA method matches a dense SVD."""
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
        result.operator @ np.eye(counts.shape[1]),
        dense,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.singular_values, singular_values[:2], rtol=1e-10, atol=0.0
    )
    assert _compare_subspaces(result.components.T, Vt[:2].T, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("pca", "kwargs", "dense_transform"),
    [
        (
            shifted_log_pca,
            {"count_shift": 0.7},
            lambda X: _dense_shifted_log(X, 0.7),
        ),
        (
            shifted_clr_pca,
            {"count_shift": 0.7},
            lambda X: _dense_count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca,
            {"composition_shift": 0.7},
            lambda X: _dense_proportion_shifted_clr(X, 0.7),
        ),
    ],
)
def test_log_pca_mask_is_applied_after_full_transform(
    adata, pca, kwargs, dense_transform
):
    """Log transforms use the full gene universe before PCA masking."""
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
        atol=0.0,
    )
    params = result.uns["log_pca"]["params"]
    assert params["normalization_n_vars"] == adata.n_vars
    assert params["pca_n_vars"] == int(mask.sum())
    assert np.isnan(result.varm["log_pca"][~mask]).all()


def test_count_shifted_transforms_reject_empty_cells(counts):
    """Fixed-count shifted transforms reject cells with no counts."""
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    for pca in (shifted_log_pca_matrix, shifted_clr_pca_matrix):
        with pytest.raises(ValueError, match="zero total counts"):
            pca(with_empty, n_comps=2, count_shift=1.0)


def test_proportion_shifted_clr_rejects_empty_cells(counts):
    """Proportion-shifted CLR rejects cells with zero totals."""
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]))], format="csr"
    )
    with pytest.raises(ValueError, match="zero total counts"):
        proportion_shifted_clr_pca_matrix(with_empty, n_comps=2, composition_shift=1.0)


def test_proportion_shifted_clr_is_row_scale_invariant(counts):
    """Proportion-shifted CLR is invariant to row-wise count scaling."""
    scales = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    scaled = sparse.diags(scales) @ counts
    original = build_proportion_shifted_clr_representation(
        counts, composition_shift=0.3
    )
    rescaled = build_proportion_shifted_clr_representation(
        scaled.tocsr(), composition_shift=0.3
    )
    np.testing.assert_allclose(
        _materialize(original),
        _materialize(rescaled),
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "builder",
    [build_shifted_log_representation, build_shifted_clr_representation],
)
def test_count_shifted_transforms_are_invariant_to_joint_global_scaling(
    counts, builder
):
    """Scaling counts and the raw-count shift together preserves values."""
    original = builder(counts, count_shift=0.3)
    rescaled = builder(counts * 7, count_shift=2.1)

    np.testing.assert_allclose(
        _materialize(original),
        _materialize(rescaled),
        rtol=0.0,
        atol=1e-15,
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
    """Log PCA results record transform and shift-domain metadata."""
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
    """Log PCA rejects shifts that are nonpositive or nonfinite."""
    with pytest.raises(ValueError, match="finite and positive"):
        pca(counts, n_comps=2, **{parameter: value})


def test_log_transforms_reject_float64_ratio_overflow():
    """Tiny shifts fail clearly when a stored ratio cannot fit in float64."""
    counts = sparse.csr_matrix([[1.0, 1.0]])
    tiny = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="Count-to-prior ratios overflow"):
        build_shifted_log_representation(counts, count_shift=tiny)
    with pytest.raises(ValueError, match="Count-to-composition ratios overflow"):
        build_proportion_shifted_clr_representation(counts, composition_shift=tiny)


@pytest.mark.parametrize(
    "pca",
    [
        shifted_log_pca_matrix,
        shifted_clr_pca_matrix,
        proportion_shifted_clr_pca_matrix,
    ],
)
def test_log_pca_requires_explicit_shift(counts, pca):
    """Shifted-log APIs require an explicit domain-specific shift."""
    with pytest.raises(TypeError):
        pca(counts, n_comps=2)


def test_old_ambiguous_pseudocount_keyword_is_rejected(counts):
    """Shifted-log APIs reject the obsolete ambiguous pseudocount keyword."""
    with pytest.raises(TypeError):
        shifted_clr_pca_matrix(counts, n_comps=2, pseudocount=1.0)
