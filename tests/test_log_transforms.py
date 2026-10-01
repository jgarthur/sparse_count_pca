"""Tests for owned log-transform storage and pre-cast float64 factor fitting."""

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca import _log_transforms as log_module
from sparse_count_pca import _transform as transform_module
from sparse_count_pca._operator import SparseLowRankLinearOperator
from tests._oracles import (
    _dense_count_shifted_clr,
    _dense_dirichlet,
    _dense_log1p_norm,
    _dense_proportion_shifted_clr,
)

_FAMILIES = [
    "log1p",
    "shifted_clr",
    "proportion_shifted_clr",
    "dirichlet_log",
    "dirichlet_clr",
]


def _log_case(family, counts):
    """Return a builder, public method, parameters, and independent dense oracle."""
    factors = np.linspace(0.7, 1.8, counts.shape[0])
    prior = np.arange(1, counts.shape[1] + 1, dtype=np.float64)
    prior /= prior.sum()
    cases = {
        "log1p": (
            log_module.build_log1p_norm_representation,
            scp.Log1pNormalized,
            {"size_factors": factors},
            _dense_log1p_norm(counts, factors),
        ),
        "shifted_clr": (
            log_module.build_shifted_clr_representation,
            scp.ShiftedCLR,
            {"count_shift": 0.7},
            _dense_count_shifted_clr(counts, 0.7),
        ),
        "proportion_shifted_clr": (
            log_module.build_proportion_shifted_clr_representation,
            scp.ProportionShiftedCLR,
            {"composition_shift": 0.3},
            _dense_proportion_shifted_clr(counts, 0.3),
        ),
        "dirichlet_log": (
            log_module.build_dirichlet_log_representation,
            scp.DirichletLog,
            {"concentration": 2.5, "prior_proportions": prior},
            _dense_dirichlet(counts, 2.5, prior, clr=False),
        ),
        "dirichlet_clr": (
            log_module.build_dirichlet_clr_representation,
            scp.DirichletCLR,
            {"concentration": 2.5, "prior_proportions": prior},
            _dense_dirichlet(counts, 2.5, prior, clr=True),
        ),
    }
    return cases[family]


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("target_nnz", [1, 7, 100_000])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_log_builders_write_owned_storage_and_preserve_factors(
    counts, monkeypatch, family, dtype, target_nnz, index_dtype
):
    """Blocked logs match dense oracles while keeping factors fitted before casting."""
    monkeypatch.setattr(log_module, "_LOG_TRANSFORM_BLOCK_NNZ", target_nnz)
    counts.indices = counts.indices.astype(index_dtype)
    counts.indptr = counts.indptr.astype(index_dtype)
    builder, _, kwargs, expected = _log_case(family, counts)
    original = counts.toarray()
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    reference = builder(counts, **kwargs)
    actual = builder(counts, dtype=dtype, **kwargs)

    assert actual.sparse.dtype == np.dtype(dtype)
    assert actual.sparse.indices.dtype == np.dtype(np.int32)
    assert actual.sparse.indptr.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        actual.sparse.data, reference.sparse.data.astype(dtype)
    )
    np.testing.assert_array_equal(actual.left, reference.left)
    np.testing.assert_array_equal(actual.right, reference.right)
    for name in ("data", "indices", "indptr"):
        assert not np.shares_memory(getattr(actual.sparse, name), getattr(counts, name))
    operator = SparseLowRankLinearOperator(actual, center=False, dtype=dtype)
    tolerance = 2e-6 if dtype == "float32" else 1e-14
    np.testing.assert_allclose(
        operator @ np.eye(counts.shape[1], dtype=dtype),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_array_equal(counts.toarray(), original)


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_public_log_transform_reuses_builder_storage(
    counts, monkeypatch, family, dtype
):
    """Public transforms forward storage dtype and keep the builder's sparse arrays."""
    builder, method, kwargs, _ = _log_case(family, counts)
    built = []

    def capture(*args, **parameters):
        """Record the owned matrix returned by the log builder."""
        representation = builder(*args, **parameters)
        built.append(representation.sparse)
        return representation

    monkeypatch.setattr(transform_module, builder.__name__, capture)
    transformed = scp.transform(counts, method(**kwargs), dtype=dtype)

    assert built[0].dtype == np.dtype(dtype)
    assert transformed._sparse is built[0]


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_pca_matches_precast_factor_dense_svd(counts, family, dtype):
    """Direct-storage PCA matches a dense SVD with float64-fitted factors."""
    builder, method, kwargs, _ = _log_case(family, counts)
    reference = builder(counts, **kwargs)
    dense = reference.sparse.toarray().astype(dtype)
    dense += reference.left.astype(dtype) @ reference.right.astype(dtype).T
    mean = dense.mean(axis=0, dtype=np.float64).astype(dtype)
    centered = (dense - mean).astype(np.float64)
    left, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    actual = scp.transform(counts, method(**kwargs), dtype=dtype).pca(n_comps=2)
    signs = np.sign(np.sum(actual.components * components[:2], axis=1))
    tolerance = 2e-6 if dtype == "float32" else 1e-12

    np.testing.assert_allclose(actual.mean, mean, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual.singular_values, singular_values[:2], rtol=tolerance
    )
    np.testing.assert_allclose(
        actual.components * signs[:, None],
        components[:2],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.scores * signs,
        left[:, :2] * singular_values[:2],
        rtol=tolerance,
        atol=tolerance,
    )
    total_variance = np.sum(centered**2) / (counts.shape[0] - 1)
    variance = singular_values[:2] ** 2 / (counts.shape[0] - 1)
    assert actual.total_variance == pytest.approx(total_variance, rel=tolerance)
    np.testing.assert_allclose(actual.explained_variance, variance, rtol=tolerance)
    np.testing.assert_allclose(
        actual.explained_variance_ratio, variance / total_variance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("by_row", [False, True])
def test_blocked_logs_reject_late_ratio_overflow(monkeypatch, dtype, by_row):
    """An overflowing ratio in a later block fails before output storage casting."""
    monkeypatch.setattr(log_module, "_LOG_TRANSFORM_BLOCK_NNZ", 1)
    counts = sparse.csr_matrix([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="ratios overflow float64"):
        if by_row:
            log_module.log1p_row_scaled_counts(
                counts,
                np.array([1.0, 1.0, 1e-320]),
                dtype=dtype,
                overflow_message="Count-to-size-factor ratios overflow float64",
            )
        else:
            log_module.log1p_count_correction(
                counts, np.array([1.0, 1e-320]), dtype=dtype
            )


def test_float32_log_storage_uses_float64_ratios():
    """A ratio exceeding float32 range remains valid when its logged value fits."""
    counts = sparse.csr_matrix(np.array([[1e30]], dtype=np.float32))
    result = log_module.build_log1p_norm_representation(
        counts, size_factors=[1e-30], dtype="float32"
    )
    expected = np.float32(np.log1p(float(counts.data[0]) / 1e-30))

    np.testing.assert_array_equal(result.sparse.data, [expected])
