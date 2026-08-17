"""Tests for shifted-CLR representations and PCA."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import (
    TransformedMatrix,
    log1p_norm_pca,
    log1p_norm_pca_matrix,
    proportion_shifted_clr_pca,
    proportion_shifted_clr_pca_matrix,
    shifted_clr_pca,
    shifted_clr_pca_matrix,
)
from sparse_count_pca._log_transforms import (
    build_dirichlet_clr_representation,
    build_log1p_norm_representation,
    build_proportion_shifted_clr_representation,
    build_shifted_clr_representation,
)
from sparse_count_pca._operator import SparseLowRankLinearOperator
from tests._oracles import (
    _compare_subspaces,
    _dense_count_shifted_clr,
    _dense_log1p_norm,
    _dense_proportion_shifted_clr,
)
from tests.shifted_clr_reference.oracle import pflog


def _materialize(representation, *, center=False):
    operator = SparseLowRankLinearOperator(
        representation, center=center, dtype="float64"
    )
    return operator @ np.eye(representation.shape[1])


@pytest.mark.parametrize("count_shift", [0.25, 1.0, 2.0])
def test_count_shifted_clr_representation_matches_reference(counts, count_shift):
    """Count-scale shifted CLR values match the independent reference formula."""
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


def test_log1p_norm_representation_is_exact_sparse_rank_zero(counts):
    """Size-factor log1p preserves count support with no low-rank correction."""
    size_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    representation = build_log1p_norm_representation(
        counts,
        size_factors=size_factors,
    )

    np.testing.assert_allclose(
        _materialize(representation),
        _dense_log1p_norm(counts, size_factors),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(representation.sparse.indices, counts.indices)
    np.testing.assert_array_equal(representation.sparse.indptr, counts.indptr)
    assert representation.sparse.nnz == counts.nnz
    assert representation.left.shape == (counts.shape[0], 0)
    assert representation.right.shape == (counts.shape[1], 0)
    assert representation.rank == 0


def test_log1p_norm_is_proportion_shifted_clr_sparse_part(counts):
    """Target-sum log1p equals the sparse term of composition-shifted CLR."""
    target_sum = 7.0
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    log1p_norm = build_log1p_norm_representation(
        counts,
        size_factors=row_totals / target_sum,
    )
    clr = build_proportion_shifted_clr_representation(
        counts,
        composition_shift=1.0 / target_sum,
    )

    np.testing.assert_allclose(
        log1p_norm.sparse.toarray(),
        clr.sparse.toarray(),
        rtol=0.0,
        atol=1e-15,
    )


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
            shifted_clr_pca_matrix,
            {"count_shift": 0.7},
            lambda X: _dense_count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca_matrix,
            {"composition_shift": 0.7},
            lambda X: _dense_proportion_shifted_clr(X, 0.7),
        ),
        (
            log1p_norm_pca_matrix,
            {"target_sum": 7.0},
            lambda X: _dense_log1p_norm(
                X,
                np.asarray(X.sum(axis=1)).ravel() / 7.0,
            ),
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
            shifted_clr_pca,
            {"count_shift": 0.7},
            lambda X: _dense_count_shifted_clr(X, 0.7),
        ),
        (
            proportion_shifted_clr_pca,
            {"composition_shift": 0.7},
            lambda X: _dense_proportion_shifted_clr(X, 0.7),
        ),
        (
            log1p_norm_pca,
            {"target_sum": 7.0},
            lambda X: _dense_log1p_norm(
                X,
                np.asarray(X.sum(axis=1)).ravel() / 7.0,
            ),
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


def test_proportion_shifted_clr_is_row_scale_invariant(counts):
    """Composition-scale shifted CLR is invariant to row-wise count scaling."""
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


def test_count_shifted_clr_is_invariant_to_joint_global_scaling(counts):
    """Scaling counts and the raw-count shift together preserves CLR values."""
    original = build_shifted_clr_representation(counts, count_shift=0.3)
    rescaled = build_shifted_clr_representation(counts * 7, count_shift=2.1)

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


def test_log1p_norm_default_records_resolved_median_target(counts):
    """Default log1p normalization records median-depth factor metadata."""
    result = log1p_norm_pca_matrix(counts, n_comps=2)
    row_totals = np.asarray(counts.sum(axis=1)).ravel()

    assert result.params["transform"] == "log1p_norm"
    assert result.params["shift_domain"] == "normalized_count"
    assert result.params["target_sum"] is None
    assert result.params["resolved_target_sum"] == np.median(row_totals)
    assert result.params["size_factor_source"] == "median_target_sum"
    assert result.params["size_factor_median"] == pytest.approx(1.0)
    assert result.params["effective_count_shift"] == "size_factor"


def test_log1p_norm_explicit_target_records_target_recipe(counts):
    """Explicit target normalization records its requested and resolved target."""
    result = log1p_norm_pca_matrix(counts, n_comps=2, target_sum=1e4)

    assert result.params["target_sum"] == 1e4
    assert result.params["resolved_target_sum"] == 1e4
    assert result.params["size_factor_source"] == "target_sum"


def test_log1p_norm_uses_supplied_size_factors_without_rescaling(counts):
    """Supplied factors retain their scale in transformed values and metadata."""
    size_factors = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    transformed = log1p_norm_pca_matrix(
        counts,
        n_comps=2,
        size_factors=size_factors,
        return_operator=True,
    )
    expected = _dense_log1p_norm(counts, size_factors)
    expected -= expected.mean(axis=0)

    np.testing.assert_allclose(
        transformed.operator @ np.eye(counts.shape[1]),
        expected,
        rtol=0.0,
        atol=1e-12,
    )
    assert transformed.params["resolved_target_sum"] is None
    assert transformed.params["size_factor_source"] == "supplied"
    assert transformed.params["size_factor_median"] == np.median(size_factors)
    np.testing.assert_array_equal(transformed.params["size_factors"], size_factors)


def test_log1p_norm_anndata_resolves_size_factors_from_obs(adata):
    """AnnData log1p normalization aligns named size factors from observations."""
    size_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    adata.obs["scran_size_factor"] = size_factors.astype(object)
    assert adata.obs["scran_size_factor"].dtype == object

    result = log1p_norm_pca(
        adata,
        n_comps=2,
        size_factors="scran_size_factor",
        key_added="log1p_norm",
        copy=True,
    )
    expected = log1p_norm_pca_matrix(
        adata.X,
        n_comps=2,
        size_factors=size_factors,
    )

    np.testing.assert_allclose(
        result.uns["log1p_norm"]["singular_values"],
        expected.singular_values,
        rtol=1e-10,
        atol=0.0,
    )
    params = result.uns["log1p_norm"]["params"]
    assert params["size_factors"] == "scran_size_factor"
    assert params["size_factor_source"] == "supplied"


def test_log1p_norm_explicit_mask_is_not_replaced_by_highly_variable(adata):
    """An explicit PCA mask is applied once even when highly_variable exists."""
    adata.var["highly_variable"] = [False, True, True, True]
    mask = np.array([True, True, True, False])

    result = log1p_norm_pca(
        adata,
        n_comps=2,
        target_sum=7.0,
        mask_var=mask,
        copy=True,
    )
    expected = log1p_norm_pca_matrix(
        adata.X,
        n_comps=2,
        target_sum=7.0,
        return_operator=True,
    )
    expected_values = (expected.operator @ np.eye(adata.n_vars))[:, mask]

    np.testing.assert_allclose(
        result.uns["pca"]["singular_values"],
        np.linalg.svd(expected_values, compute_uv=False)[:2],
        rtol=1e-10,
        atol=0.0,
    )
    assert result.uns["pca"]["params"]["pca_n_vars"] == int(mask.sum())


def test_log1p_norm_anndata_wrapper_does_not_retain_metadata(adata, monkeypatch):
    """One-step log1p PCA passes only counts and obs metadata to its transform."""
    retained_metadata = []
    original_pca = TransformedMatrix.pca

    def recording_pca(self, *args, **kwargs):
        retained_metadata.append((self._var, self.obs_names, self.var_names))
        return original_pca(self, *args, **kwargs)

    monkeypatch.setattr(TransformedMatrix, "pca", recording_pca)

    log1p_norm_pca(adata, n_comps=2, copy=True)

    assert retained_metadata == [(None, None, None)]


@pytest.mark.parametrize("target_sum", [0.0, -1.0, np.inf, [1.0], "1"])
def test_log1p_norm_rejects_invalid_target_sum(counts, target_sum):
    """Log1p normalization rejects nonpositive, nonfinite, or nonscalar targets."""
    with pytest.raises(ValueError, match="target_sum must be finite and positive"):
        log1p_norm_pca_matrix(counts, n_comps=2, target_sum=target_sum)


@pytest.mark.parametrize(
    "size_factors",
    [
        np.ones(5),
        np.ones((6, 1)),
        np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, np.inf, 1.0, 1.0]),
        ["1", "1", "1", "1", "1", "1"],
    ],
)
def test_log1p_norm_rejects_invalid_size_factors(counts, size_factors):
    """Log1p normalization requires one finite positive divisor per observation."""
    with pytest.raises(ValueError, match="size_factors"):
        log1p_norm_pca_matrix(counts, n_comps=2, size_factors=size_factors)


def test_log1p_norm_rejects_both_normalization_recipes(counts):
    """Target sums and supplied size factors cannot be requested together."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        log1p_norm_pca_matrix(
            counts,
            n_comps=2,
            target_sum=1e4,
            size_factors=np.ones(counts.shape[0]),
        )


def test_log1p_norm_reports_target_sum_derived_factor_overflow(counts):
    """A tiny valid target reports overflow in the target-derived recipe."""
    tiny = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="target_sum produces size factors"):
        log1p_norm_pca_matrix(counts, n_comps=2, target_sum=tiny)


def test_log1p_norm_matrix_rejects_named_size_factors(counts):
    """Named size factors require observation metadata from AnnData."""
    with pytest.raises(TypeError, match="requires an AnnData input"):
        log1p_norm_pca_matrix(counts, n_comps=2, size_factors="size_factor")


def test_log1p_norm_anndata_rejects_missing_size_factor_key(adata):
    """Named size factors must exist in AnnData observation metadata."""
    with pytest.raises(KeyError, match="not found in adata.obs"):
        log1p_norm_pca(
            adata,
            n_comps=2,
            size_factors="missing",
            copy=True,
        )


@pytest.mark.parametrize(
    ("pca", "parameter"),
    [
        (shifted_clr_pca_matrix, "count_shift"),
        (proportion_shifted_clr_pca_matrix, "composition_shift"),
    ],
)
@pytest.mark.parametrize("value", [0.0, np.inf, [1.0], "1"])
def test_log_pca_rejects_invalid_shifts(counts, pca, parameter, value):
    """Log PCA rejects shifts that are nonpositive, nonfinite, or non-scalar."""
    with pytest.raises(ValueError, match="finite and positive"):
        pca(counts, n_comps=2, **{parameter: value})


def test_log_transforms_reject_float64_ratio_overflow():
    """Tiny shifts fail clearly when a stored ratio cannot fit in float64."""
    counts = sparse.csr_matrix([[1.0, 1.0]])
    tiny = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="Count-to-prior ratios overflow"):
        build_shifted_clr_representation(counts, count_shift=tiny)
    with pytest.raises(ValueError, match="Count-to-composition ratios overflow"):
        build_proportion_shifted_clr_representation(counts, composition_shift=tiny)
    with pytest.raises(ValueError, match="Count-to-size-factor ratios overflow"):
        build_log1p_norm_representation(counts, size_factors=np.array([tiny]))


def test_proportion_shifted_clr_rejects_row_divisor_overflow():
    """A huge composition shift cannot silently collapse transformed counts."""
    counts = sparse.csr_matrix([[1.0, 1.0]])

    with pytest.raises(ValueError, match="Count-to-composition ratios overflow"):
        build_proportion_shifted_clr_representation(
            counts,
            composition_shift=np.finfo(np.float64).max,
        )


@pytest.mark.parametrize(
    "pca",
    [
        shifted_clr_pca_matrix,
        proportion_shifted_clr_pca_matrix,
    ],
)
def test_log_pca_requires_explicit_shift(counts, pca):
    """Shifted-CLR APIs require an explicit domain-specific shift."""
    with pytest.raises(TypeError):
        pca(counts, n_comps=2)


def test_old_ambiguous_pseudocount_keyword_is_rejected(counts):
    """Shifted-CLR APIs reject the obsolete ambiguous pseudocount keyword."""
    with pytest.raises(TypeError):
        shifted_clr_pca_matrix(counts, n_comps=2, pseudocount=1.0)
