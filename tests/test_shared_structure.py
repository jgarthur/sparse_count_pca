"""Tests for opt-in CSR support sharing and one-step operator isolation."""

import gc
import warnings
import weakref
from contextlib import nullcontext

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca import _residuals as residuals_module
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
    scp.Residual(clip=0.5, clip_mode="upper"),
    scp.Residual(residual="deviance", clip=0.5, clip_mode="upper"),
]

_NOT_SHARED = "did not share the count matrix's sparsity structure"

_ONE_STEP = [
    (scp.log1p_norm_pca_matrix, {}),
    (scp.shifted_clr_pca_matrix, {"count_shift": 0.7}),
    (scp.proportion_shifted_clr_pca_matrix, {"composition_shift": 0.3}),
    (scp.dirichlet_log_pca_matrix, {"concentration": 2.5}),
    (scp.dirichlet_clr_pca_matrix, {"concentration": 2.5}),
    (scp.residual_pca_matrix, {"clip": None}),
    (scp.residual_pca_matrix, {"clip": 0.5, "clip_mode": "upper"}),
    (
        scp.residual_pca_matrix,
        {"residual": "deviance", "clip": 0.5, "clip_mode": "upper"},
    ),
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
        assert np.shares_memory(getattr(shared._sparse, name), getattr(counts, name))
        assert getattr(counts, name).flags.writeable == writeable

    counts.data[:] *= 2
    np.testing.assert_array_equal(shared.materialize(), expected)


@pytest.mark.parametrize("method", _TRANSFORMS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_shared_transform_pca_preserves_operator_isolation(
    counts, method, dtype, index_dtype
):
    """A retained PCA operator owns support independently of an opted-in transform."""
    counts.indices = counts.indices.astype(index_dtype)
    counts.indptr = counts.indptr.astype(index_dtype)
    shared = scp.transform(counts, method, dtype=dtype, share_structure=True)
    expected = shared.materialize()
    result = shared.pca(n_comps=2, return_operator=True)
    centered = result.operator @ np.eye(counts.shape[1])

    for name in ("data", "indices", "indptr"):
        assert not np.shares_memory(
            getattr(result.operator.S, name), getattr(shared._sparse, name)
        )
    assert result.operator.S.dtype == np.dtype(dtype)
    assert result.operator.S.indices.dtype == np.dtype(np.int32)
    assert result.operator.S.indptr.dtype == np.dtype(np.int32)
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
def test_one_step_shares_structure_only_without_returned_operators(
    counts, monkeypatch, entry_point, kwargs, return_operator, dtype
):
    """One-step calls borrow support only when the operator stays inside the call."""
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
@pytest.mark.parametrize("clip", [None, 0.5, 3.0])
@pytest.mark.parametrize("clip_mode", ["symmetric", "upper"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_residual_sharing_depends_on_actual_support_changes(
    counts, residual, clip, clip_mode, dtype
):
    """Residuals borrow unchanged support and own support modified by clipping."""
    method = scp.Residual(
        residual=residual, clip=clip, clip_mode=clip_mode, clip_max_nnz_ratio=None
    )
    owned = scp.transform(counts, method, dtype=dtype)
    expected = owned.materialize()
    preserves_support = clip is None or clip_mode == "upper" or clip == 3.0
    if preserves_support:
        assert owned._sparse.nnz == counts.nnz
    else:
        assert owned._sparse.nnz > counts.nnz
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    expectation = (
        nullcontext()
        if preserves_support
        else pytest.warns(UserWarning, match=_NOT_SHARED)
    )
    with expectation:
        result = scp.transform(counts, method, dtype=dtype, share_structure=True)

    np.testing.assert_array_equal(result.materialize(), expected)
    for name in ("indices", "indptr"):
        assert (
            np.shares_memory(getattr(result._sparse, name), getattr(counts, name))
            == preserves_support
        )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_residual_zero_removal_preserves_input_across_blocks(
    residual, dtype, monkeypatch
):
    """Computed zeros require owned support even when later blocks have no zeros."""
    counts = sparse.csr_matrix([[1.0, 1e32], [1e32, 1.0], [1e32, 1e32]])
    snapshots = [
        values.copy() for values in (counts.data, counts.indices, counts.indptr)
    ]
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    monkeypatch.setattr(residuals_module, "_RESIDUAL_SUPPORT_BLOCK_SIZE", 2)
    method = scp.Residual(residual=residual, clip=100.0, clip_mode="upper")
    owned = scp.transform(counts, method, dtype=dtype)
    with pytest.warns(UserWarning, match=_NOT_SHARED):
        shared = scp.transform(counts, method, dtype=dtype, share_structure=True)

    assert shared._sparse.nnz == counts.nnz - 2
    assert np.all(shared._sparse.data != 0)
    np.testing.assert_array_equal(shared.materialize(), owned.materialize())
    for name, snapshot in zip(("data", "indices", "indptr"), snapshots, strict=True):
        assert not np.shares_memory(
            getattr(shared._sparse, name), getattr(counts, name)
        )
        np.testing.assert_array_equal(getattr(counts, name), snapshot)
        assert not getattr(counts, name).flags.writeable


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_residual_zero_detection_uses_storage_dtype(residual, dtype):
    """Corrections that underflow only during storage casting are pruned safely."""
    counts = sparse.csr_matrix(np.full((2, 2), 1e-100))
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    expectation = (
        pytest.warns(UserWarning, match=_NOT_SHARED)
        if dtype == "float32"
        else nullcontext()
    )
    with expectation:
        result = scp.transform(
            counts,
            scp.Residual(residual=residual, clip=0.5, clip_mode="upper"),
            dtype=dtype,
            check_values=False,
            share_structure=True,
        )

    assert result._sparse.nnz == (0 if dtype == "float32" else counts.nnz)
    assert np.all(result._sparse.data != 0)
    for name in ("indices", "indptr"):
        assert np.shares_memory(
            getattr(result._sparse, name), getattr(counts, name)
        ) == (dtype == "float64")
    np.testing.assert_array_equal(counts.data, np.full(4, 1e-100))


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
    with (
        pytest.warns(UserWarning, match="copied for canonicalization"),
        pytest.warns(UserWarning, match=_NOT_SHARED),
    ):
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


def _remove_first_entry(counts):
    counts.data[0] = 0
    counts.eliminate_zeros()


def _move_first_entry(counts):
    counts.indices[1] = 2


@pytest.mark.parametrize("mutate", [_remove_first_entry, _move_first_entry])
def test_shared_transform_rejects_changed_structure(counts, mutate):
    """PCA and materialization refuse support that changed after sharing."""
    transformed = scp.transform(counts, scp.Log1pNormalized(), share_structure=True)
    mutate(counts)

    with pytest.raises(RuntimeError, match="sparsity structure changed"):
        transformed.materialize()
    with pytest.raises(RuntimeError, match="sparsity structure changed"):
        transformed.pca(2)


def test_shared_transform_products_reject_removed_entries(counts):
    """Matrix products detect removed entries without reading all support."""
    transformed = scp.transform(counts, scp.Log1pNormalized(), share_structure=True)
    _remove_first_entry(counts)

    for product in (
        lambda: transformed @ np.ones(counts.shape[1]),
        lambda: transformed.T @ np.ones(counts.shape[0]),
        lambda: transformed @ np.ones((counts.shape[1], 2)),
        lambda: transformed.T @ np.ones((counts.shape[0], 2)),
    ):
        with pytest.raises(RuntimeError, match="sparsity structure changed"):
            product()


def test_structure_checks_allow_value_changes_and_owned_support(counts):
    """Changed count values and owned transforms pass the structure checks."""
    shared = scp.transform(counts, scp.Log1pNormalized(), share_structure=True)
    owned = scp.transform(counts, scp.Log1pNormalized())
    expected = shared.materialize()
    counts.data[:] = 7

    np.testing.assert_array_equal(shared.materialize(), expected)
    shared.pca(2)
    _remove_first_entry(counts)
    np.testing.assert_array_equal(owned.materialize(), expected)
    owned.pca(2)


def test_owned_clipping_growth_adds_corrections_to_read_only_support(counts):
    """Support growth leaves read-only input intact and returns owned support."""
    method = scp.Residual(clip=0.5, clip_max_nnz_ratio=None)
    expected = scp.transform(counts.copy(), method).materialize()
    snapshots = [
        values.copy() for values in (counts.data, counts.indices, counts.indptr)
    ]
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False
    result = scp.transform(counts, method)

    assert result._sparse.nnz > counts.nnz
    np.testing.assert_array_equal(result.materialize(), expected)
    for name, snapshot in zip(("data", "indices", "indptr"), snapshots, strict=True):
        assert not np.shares_memory(
            getattr(result._sparse, name), getattr(counts, name)
        )
        np.testing.assert_array_equal(getattr(counts, name), snapshot)


@pytest.mark.parametrize(
    ("prepare", "method", "borrows"),
    [
        (lambda counts: counts, scp.Log1pNormalized(), True),
        (sparse.csc_matrix, scp.Log1pNormalized(), False),
        (lambda counts: counts, scp.Residual(clip=0.5, clip_max_nnz_ratio=None), False),
    ],
)
def test_shared_transform_checks_only_borrowed_support(
    counts, prepare, method, borrows
):
    """Requested sharing records a checksum if it borrows and warns if not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = scp.transform(prepare(counts), method, share_structure=True)

    assert (transformed._support_checksum is not None) == borrows
    not_shared = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and _NOT_SHARED in str(warning.message)
    ]
    assert len(not_shared) == (0 if borrows else 1)
