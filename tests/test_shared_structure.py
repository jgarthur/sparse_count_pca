"""Tests for opt-in CSR support sharing and transient one-step ownership."""

import gc
import weakref

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca._operator import SparseLowRankLinearOperator

_TRANSFORMS = [
    scp.Log1pNormalized(),
    scp.ShiftedCLR(count_shift=0.7),
    scp.ProportionShiftedCLR(composition_shift=0.3),
    scp.DirichletLog(concentration=2.5),
    scp.DirichletCLR(concentration=2.5),
    scp.Residual(clip=None),
    scp.Residual(model="binomial", clip=None),
    scp.Residual(model="scaled_nb", alpha=0.2, clip=None),
]

_ONE_STEP = [
    (scp.log1p_norm_pca_matrix, {}),
    (scp.shifted_clr_pca_matrix, {"count_shift": 0.7}),
    (scp.proportion_shifted_clr_pca_matrix, {"composition_shift": 0.3}),
    (scp.dirichlet_log_pca_matrix, {"concentration": 2.5}),
    (scp.dirichlet_clr_pca_matrix, {"concentration": 2.5}),
    (scp.residual_pca_matrix, {"clip": None}),
    (scp.correspondence_analysis_matrix, {}),
]


@pytest.mark.parametrize("method", _TRANSFORMS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_transform_shares_only_compatible_support(counts, method, dtype, index_dtype):
    """Opt-in sharing preserves fitted values while default transforms own support."""
    counts.indices = counts.indices.astype(index_dtype)
    counts.indptr = counts.indptr.astype(index_dtype)
    original_flags = [
        values.flags.writeable for values in (counts.indices, counts.indptr)
    ]
    owned = scp.transform(counts, method, dtype=dtype)
    shared = scp.transform(counts, method, dtype=dtype, share_structure=True)
    expected = owned.materialize()

    np.testing.assert_array_equal(shared.materialize(), expected)
    np.testing.assert_array_equal(shared._left, owned._left)
    np.testing.assert_array_equal(shared._right, owned._right)
    assert not np.shares_memory(shared._sparse.data, counts.data)
    for name, writeable in zip(("indices", "indptr"), original_flags, strict=True):
        assert not np.shares_memory(getattr(owned._sparse, name), getattr(counts, name))
        assert np.shares_memory(
            getattr(shared._sparse, name), getattr(counts, name)
        ) == (index_dtype == np.int32)
        assert getattr(counts, name).flags.writeable == writeable

    counts.data[:] *= 2
    np.testing.assert_array_equal(shared.materialize(), expected)


@pytest.mark.parametrize("method", _TRANSFORMS)
def test_shared_transform_pca_preserves_operator_isolation(counts, method):
    """A retained PCA operator owns support independently of an opted-in transform."""
    shared = scp.transform(counts, method, share_structure=True)
    expected = shared.materialize()
    result = shared.pca(n_comps=2, return_operator=True)
    centered = result.operator @ np.eye(counts.shape[1])

    for name in ("data", "indices", "indptr"):
        assert not np.shares_memory(
            getattr(result.operator.S, name), getattr(shared._sparse, name)
        )
    original_value = result.operator.S.data[0]
    result.operator.S.data[0] += 1.0
    np.testing.assert_array_equal(shared.materialize(), expected)
    result.operator.S.data[0] = original_value
    start, stop = counts.indptr[:2]
    counts.indices[start:stop] = counts.indices[start:stop][::-1]
    np.testing.assert_array_equal(result.operator @ np.eye(counts.shape[1]), centered)


@pytest.mark.parametrize(("entry_point", "kwargs"), _ONE_STEP)
@pytest.mark.parametrize("return_operator", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_one_step_shares_structure_only_for_transient_operators(
    counts, monkeypatch, entry_point, kwargs, return_operator, dtype
):
    """One-step calls borrow support transiently and isolate retained operators."""
    original = SparseLowRankLinearOperator.__init__
    sharing = []
    snapshots = [
        values.copy() for values in (counts.data, counts.indices, counts.indptr)
    ]
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False

    def record_support(self, representation, **parameters):
        """Record whether the solver's sparse support borrows caller-owned arrays."""
        original(self, representation, **parameters)
        sharing.append(
            [
                np.shares_memory(getattr(self.S, name), getattr(counts, name))
                for name in ("indices", "indptr")
            ]
        )

    monkeypatch.setattr(SparseLowRankLinearOperator, "__init__", record_support)
    result = entry_point(
        counts, n_comps=2, dtype=dtype, return_operator=return_operator, **kwargs
    )

    assert sharing == [[not return_operator, not return_operator]]
    for name, snapshot in zip(("data", "indices", "indptr"), snapshots, strict=True):
        np.testing.assert_array_equal(getattr(counts, name), snapshot)
        assert not getattr(counts, name).flags.writeable
    if return_operator:
        expected = result.operator @ np.eye(counts.shape[1])
        counts.indices.flags.writeable = True
        counts.indices[:] = (counts.indices + 1) % counts.shape[1]
        np.testing.assert_array_equal(
            result.operator @ np.eye(counts.shape[1]), expected
        )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
@pytest.mark.parametrize("clip", [None, 0.5])
@pytest.mark.parametrize("clip_mode", ["symmetric", "upper"])
def test_residual_pruning_owns_support_when_sharing_is_requested(
    counts, residual, clip, clip_mode
):
    """Residual pruning and support growth never mutate borrowed input structure."""
    method = scp.Residual(
        residual=residual, clip=clip, clip_mode=clip_mode, clip_max_nnz_ratio=None
    )
    expected = scp.transform(counts, method).materialize()
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    result = scp.transform(counts, method, share_structure=True)

    np.testing.assert_array_equal(result.materialize(), expected)
    for name in ("indices", "indptr"):
        assert np.shares_memory(
            getattr(result._sparse, name), getattr(counts, name)
        ) == (residual == "pearson" and clip is None)


def test_shared_ann_data_transform_uses_selected_layer(adata):
    """AnnData opt-in sharing borrows the selected count layer without freezing it."""
    transformed = scp.transform(
        adata, scp.Log1pNormalized(), layer="counts", share_structure=True
    )
    for name in ("indices", "indptr"):
        assert np.shares_memory(
            getattr(transformed._sparse, name), getattr(adata.layers["counts"], name)
        )
        assert not np.shares_memory(
            getattr(transformed._sparse, name), getattr(adata.X, name)
        )
        assert getattr(adata.layers["counts"], name).flags.writeable


def test_shared_transform_keeps_support_after_input_is_deleted(counts):
    """Borrowed support remains alive independently of the input matrix object."""
    counts = counts.copy()
    transformed = scp.transform(counts, scp.Log1pNormalized(), share_structure=True)
    expected = transformed.materialize()
    reference = weakref.ref(counts)
    del counts
    gc.collect()

    assert reference() is None
    np.testing.assert_array_equal(transformed.materialize(), expected)


def test_shared_transform_canonicalization_keeps_input_isolated():
    """Noncanonical input is copied before any output structure can be borrowed."""
    counts = sparse.csr_matrix(
        ([2.0, 3.0, 1.0, 4.0], [1, 0, 0, 1], [0, 2, 4]), shape=(2, 2)
    )
    original = counts.indices.copy()
    with pytest.warns(UserWarning, match="copied for canonicalization"):
        transformed = scp.transform(counts, scp.Log1pNormalized(), share_structure=True)

    assert not np.shares_memory(transformed._sparse.indices, counts.indices)
    assert not np.shares_memory(transformed._sparse.indptr, counts.indptr)
    np.testing.assert_array_equal(counts.indices, original)


@pytest.mark.parametrize("share_structure", [None, 0, 1, "false"])
def test_transform_rejects_nonboolean_sharing(counts, share_structure):
    """Sharing requires an explicit boolean rather than a truthy ownership request."""
    with pytest.raises(TypeError, match="share_structure must be a boolean"):
        scp.transform(counts, scp.Log1pNormalized(), share_structure=share_structure)


@pytest.mark.parametrize(("entry_point", "kwargs"), _ONE_STEP)
def test_one_step_ann_data_selection_preserves_read_only_count_structure(
    adata, entry_point, kwargs
):
    """Masked AnnData analyses preserve read-only count support."""
    adata.var["highly_variable"] = [True, False, True, True]
    snapshots = [
        values.copy() for values in (adata.X.data, adata.X.indices, adata.X.indptr)
    ]
    for values in (adata.X.data, adata.X.indices, adata.X.indptr):
        values.flags.writeable = False
    analysis = getattr(scp, entry_point.__name__.removesuffix("_matrix"))
    analysis(adata, n_comps=2, **kwargs)
    if entry_point == scp.correspondence_analysis_matrix:
        assert adata.obsm["X_ca"].shape == (6, 2)
    else:
        assert adata.obsm["X_pca"].shape == (6, 2)
        assert np.isnan(adata.varm["PCs"][1]).all()
    for name, snapshot in zip(("data", "indices", "indptr"), snapshots, strict=True):
        np.testing.assert_array_equal(getattr(adata.X, name), snapshot)
    assert not adata.X.indices.flags.writeable
    assert not adata.X.indptr.flags.writeable
