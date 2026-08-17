"""Tests for the public two-step transformed-matrix API."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

import sparse_count_pca as scp
from sparse_count_pca._operator import SparseLowRankLinearOperator
from tests._oracles import (
    _dense_count_shifted_clr,
    _dense_log1p_norm,
    _materialize_dense_residual,
)


@pytest.mark.parametrize(
    ("method", "dense_oracle", "atol"),
    [
        (
            scp.ShiftedCLR(count_shift=0.7),
            lambda X: _dense_count_shifted_clr(X, 0.7),
            1e-14,
        ),
        (
            scp.Log1pNormalized(size_factors=np.arange(1.0, 7.0)),
            lambda X: _dense_log1p_norm(X, np.arange(1.0, 7.0)),
            1e-14,
        ),
        (scp.Residual(), _materialize_dense_residual, 1e-12),
    ],
    ids=["shifted-clr", "log1p-normalized", "residual"],
)
def test_transformed_matrix_is_linear_operator_and_materializes_exactly(
    counts, method, dense_oracle, atol
):
    """A transformed matrix is an exact uncentered SciPy linear operator."""
    transformed = scp.transform(counts, method)
    expected = dense_oracle(counts)

    assert isinstance(transformed, LinearOperator)
    np.testing.assert_allclose(transformed.materialize(), expected, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        transformed @ np.eye(counts.shape[1]), expected, rtol=0.0, atol=atol
    )


def test_transformed_matrix_is_isolated_from_later_input_mutation(counts):
    """A fitted transform owns its sparse values and support arrays."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=1.0))
    expected = transformed.materialize()

    counts.data[0] = 999
    start, stop = counts.indptr[:2]
    counts.indices[start:stop] = counts.indices[start:stop][::-1]

    np.testing.assert_allclose(transformed.materialize(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "entry_point",
    [
        lambda X: scp.transform(X, scp.Residual()),
        lambda X: scp.transform(X, scp.ShiftedCLR(count_shift=1.0)),
        lambda X: scp.transform(X, scp.ProportionShiftedCLR(composition_shift=1.0)),
        lambda X: scp.transform(X, scp.Log1pNormalized()),
        lambda X: scp.transform(X, scp.DirichletLog()),
        lambda X: scp.transform(X, scp.DirichletCLR()),
        lambda X: scp.residual_pca_matrix(X, n_comps=2),
        lambda X: scp.residual_pca(AnnData(X), n_comps=2, copy=True),
        lambda X: scp.shifted_clr_pca_matrix(X, n_comps=2, count_shift=1.0),
        lambda X: scp.shifted_clr_pca(
            AnnData(X), n_comps=2, count_shift=1.0, copy=True
        ),
        lambda X: scp.proportion_shifted_clr_pca_matrix(
            X, n_comps=2, composition_shift=1.0
        ),
        lambda X: scp.proportion_shifted_clr_pca(
            AnnData(X), n_comps=2, composition_shift=1.0, copy=True
        ),
        lambda X: scp.log1p_norm_pca_matrix(X, n_comps=2),
        lambda X: scp.log1p_norm_pca(AnnData(X), n_comps=2, copy=True),
        lambda X: scp.dirichlet_log_pca_matrix(X, n_comps=2),
        lambda X: scp.dirichlet_log_pca(AnnData(X), n_comps=2, copy=True),
        lambda X: scp.dirichlet_clr_pca_matrix(X, n_comps=2),
        lambda X: scp.dirichlet_clr_pca(AnnData(X), n_comps=2, copy=True),
        lambda X: scp.correspondence_analysis_matrix(X, n_comps=2),
        lambda X: scp.correspondence_analysis(AnnData(X), n_comps=2, copy=True),
    ],
    ids=[
        "transform-residual",
        "transform-shifted-clr",
        "transform-proportion-shifted-clr",
        "transform-log1p-normalized",
        "transform-dirichlet-log",
        "transform-dirichlet-clr",
        "residual-pca-matrix",
        "residual-pca-anndata",
        "shifted-clr-pca-matrix",
        "shifted-clr-pca-anndata",
        "proportion-shifted-clr-pca-matrix",
        "proportion-shifted-clr-pca-anndata",
        "log1p-norm-pca-matrix",
        "log1p-norm-pca-anndata",
        "dirichlet-log-pca-matrix",
        "dirichlet-log-pca-anndata",
        "dirichlet-clr-pca-matrix",
        "dirichlet-clr-pca-anndata",
        "correspondence-analysis-matrix",
        "correspondence-analysis-anndata",
    ],
)
def test_every_public_entry_point_rejects_empty_cells(counts, entry_point):
    """Every public entry point rejects an observation with no counts."""
    with_empty = sparse.vstack(
        [counts, sparse.csr_matrix((1, counts.shape[1]), dtype=counts.dtype)],
        format="csr",
    )

    with pytest.raises(ValueError, match="Cells with zero total counts"):
        entry_point(with_empty)


def test_returned_operator_is_isolated_from_fitted_transform(counts):
    """A retained PCA operator cannot mutate its source transformed matrix."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=1.0))
    expected = transformed.materialize()

    result = transformed.pca(n_comps=2, return_operator=True)
    result.operator.S.data[0] += 1.0

    np.testing.assert_allclose(transformed.materialize(), expected, rtol=0.0, atol=0.0)

    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=1.0))
    expected = transformed.materialize()
    result = transformed.pca(n_comps=2, return_operator=True)
    result.operator.left[0, 0] += 1.0
    result.operator.right[0, 0] += 1.0

    np.testing.assert_allclose(transformed.materialize(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "analysis",
    ["residual", "shifted_clr", "log1p_norm", "dirichlet", "correspondence"],
)
def test_one_step_returned_operator_takes_ownership_without_copy(
    counts, monkeypatch, analysis
):
    """Every one-step family transfers its private representation to the operator."""
    copy_arguments = []
    original_init = SparseLowRankLinearOperator.__init__

    def recording_init(
        self, representation, *, center=True, dtype="float64", copy=True
    ):
        copy_arguments.append(copy)
        original_init(
            self,
            representation,
            center=center,
            dtype=dtype,
            copy=copy,
        )

    monkeypatch.setattr(SparseLowRankLinearOperator, "__init__", recording_init)
    if analysis == "residual":
        result = scp.residual_pca_matrix(counts, n_comps=2, return_operator=True)
    elif analysis == "shifted_clr":
        result = scp.shifted_clr_pca_matrix(
            counts, n_comps=2, count_shift=1.0, return_operator=True
        )
    elif analysis == "log1p_norm":
        result = scp.log1p_norm_pca_matrix(
            counts,
            n_comps=2,
            return_operator=True,
        )
    elif analysis == "dirichlet":
        result = scp.dirichlet_clr_pca_matrix(counts, n_comps=2, return_operator=True)
    else:
        result = scp.correspondence_analysis_matrix(
            counts, n_comps=2, return_operator=True
        )

    assert result.operator is not None
    assert copy_arguments == [False]


def test_transformed_matrix_dtype_controls_products_materialization_and_pca(counts):
    """Transform dtype controls its operator, materialized values, and PCA."""
    transformed = scp.transform(
        counts, scp.ShiftedCLR(count_shift=0.7), dtype="float32"
    )

    assert transformed.dtype == np.dtype("float32")
    assert transformed.materialize().dtype == np.float32
    assert (transformed @ np.ones(counts.shape[1])).dtype == np.float32
    result = transformed.pca(n_comps=2)
    assert result.scores.dtype == np.float32
    assert result.components.dtype == np.float32


def test_materialize_accepts_arbitrary_positional_selections_and_out(counts, tmp_path):
    """Materialization supports ordered selections, scalar indices, and out."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    expected = _dense_count_shifted_clr(counts, 0.7)
    out = np.memmap(tmp_path / "subset.dat", mode="w+", shape=(3, 2), dtype=np.float32)

    actual = transformed.materialize(obs=[4, 1, 4], var=[3, 0], out=out, block_size=2)

    assert actual is out
    np.testing.assert_allclose(
        actual, expected[[4, 1, 4]][:, [3, 0]], rtol=1e-6, atol=1e-6
    )
    assert transformed.materialize(obs=2, var=1).shape == (1, 1)


def test_materialize_accepts_slice_and_boolean_mask_selections(counts):
    """Materialization supports slices and boolean masks on both axes."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    expected = _dense_count_shifted_clr(counts, 0.7)

    sliced = transformed.materialize(obs=slice(1, None, 2), var=slice(None, None, -2))
    np.testing.assert_allclose(sliced, expected[1::2, ::-2], rtol=1e-14, atol=1e-14)

    obs_mask = np.arange(counts.shape[0]) % 2 == 0
    var_mask = np.arange(counts.shape[1]) % 3 != 0
    masked = transformed.materialize(obs=obs_mask, var=var_mask)
    np.testing.assert_allclose(
        masked, expected[obs_mask][:, var_mask], rtol=1e-14, atol=1e-14
    )


def test_materialize_densifies_only_bounded_sparse_blocks(
    counts, tmp_path, monkeypatch
):
    """Memmap materialization densifies no sparse block larger than block_size."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    expected = _dense_count_shifted_clr(counts, 0.7)
    out = np.memmap(
        tmp_path / "full.dat", mode="w+", shape=counts.shape, dtype=np.float64
    )
    block_size = 2
    dense_block_shapes = []
    sparse_type = type(transformed._sparse)
    original_toarray = sparse_type.toarray

    def recording_toarray(matrix, *args, **kwargs):
        dense_block_shapes.append(matrix.shape)
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse_type, "toarray", recording_toarray)

    actual = transformed.materialize(out=out, block_size=block_size)

    assert actual is out
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert dense_block_shapes == [
        (min(block_size, counts.shape[0] - start), counts.shape[1])
        for start in range(0, counts.shape[0], block_size)
    ]
    assert all(shape[0] <= block_size for shape in dense_block_shapes)


def test_materialize_accepts_anndata_names(adata):
    """Transforms built from AnnData accept observation and variable names."""
    transformed = scp.transform(adata, scp.ShiftedCLR(count_shift=1.0))
    expected = _dense_count_shifted_clr(adata.X, 1.0)

    actual = transformed.materialize(
        obs=[adata.obs_names[3], adata.obs_names[0]],
        var=[adata.var_names[2], adata.var_names[1]],
    )

    np.testing.assert_allclose(
        actual, expected[[3, 0]][:, [2, 1]], rtol=1e-14, atol=1e-14
    )


def test_full_variable_access_warns_for_csr_backend(counts):
    """Selecting variables across every row warns about CSR access cost."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=1.0))

    with pytest.warns(sparse.SparseEfficiencyWarning, match="Selecting variables"):
        column = transformed.materialize(var=1)

    assert column.shape == (counts.shape[0], 1)


def test_materialize_validates_out_and_block_size(counts):
    """Materialization validates destination shape, dtype, and block size."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=1.0))

    with pytest.raises(ValueError, match="out must have shape"):
        transformed.materialize(obs=[0, 1], out=np.empty((1, counts.shape[1])))
    with pytest.raises(TypeError, match="floating-point"):
        transformed.materialize(out=np.empty(counts.shape, dtype=np.int64))
    with pytest.raises(ValueError, match="positive integer"):
        transformed.materialize(block_size=0)


def test_transformed_pca_matches_one_step_matrix_api(counts):
    """PCA of a fitted transform matches its existing one-step wrapper."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    actual = transformed.pca(n_comps=2)
    expected = scp.shifted_clr_pca_matrix(counts, n_comps=2, count_shift=0.7)

    np.testing.assert_allclose(
        actual.singular_values, expected.singular_values, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.components, expected.components, rtol=0.0, atol=1e-12
    )


def test_matrix_transform_records_normalization_and_pca_widths(counts):
    """Matrix transforms distinguish fitted and decomposed variable counts."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    mask = np.array([True, True, True, False])

    result = transformed.pca(n_comps=2, mask_var=mask)

    assert result.params["normalization_n_vars"] == counts.shape[1]
    assert result.params["pca_n_vars"] == int(mask.sum())


def test_transformed_pca_resolves_anndata_mask(adata):
    """PCA resolves AnnData variable metadata after full transformation."""
    adata.var["selected"] = [True, True, True, False]
    transformed = scp.transform(adata, scp.DirichletCLR(concentration=2.0))
    actual = transformed.pca(n_comps=2, mask_var="selected")
    expected = scp.dirichlet_clr_pca(
        adata,
        n_comps=2,
        concentration=2.0,
        mask_var="selected",
        copy=True,
    )

    np.testing.assert_allclose(
        actual.singular_values,
        expected.uns["pca"]["singular_values"],
        rtol=0.0,
        atol=1e-12,
    )


def test_anndata_transform_resolves_per_variable_parameters(adata):
    """Transform specifications resolve per-variable AnnData parameters."""
    adata.var["dispersion"] = [0.0, 0.1, 0.2, 0.3]
    transformed = scp.transform(
        adata, scp.Residual(model="scaled_nb", alpha="dispersion")
    )

    assert transformed.params["alpha"] == "dispersion"
    assert np.isfinite(transformed.materialize()).all()


def test_correspondence_analysis_is_not_a_transform_method(counts):
    """Correspondence analysis remains outside the two-step transform API."""
    with pytest.raises(TypeError, match="Transform instance"):
        scp.transform(counts, "correspondence_analysis")
