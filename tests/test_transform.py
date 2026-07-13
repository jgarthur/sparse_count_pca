"""Tests for the public two-step transformed-matrix API."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

import sparse_count_pca as scp
from tests._oracles import _materialize_dense_residual


def test_transformed_matrix_is_linear_operator_and_materializes_exactly(counts):
    """A transformed matrix is an exact uncentered SciPy linear operator."""
    transformed = scp.transform(counts, scp.ShiftedLog(count_shift=0.7))
    expected = np.log1p(counts.toarray() / 0.7)

    assert isinstance(transformed, LinearOperator)
    np.testing.assert_allclose(transformed.materialize(), expected)
    np.testing.assert_allclose(
        transformed @ np.eye(counts.shape[1]), expected
    )


def test_transformed_matrix_dtype_controls_products_materialization_and_pca(counts):
    """Transform dtype controls its operator, materialized values, and PCA."""
    transformed = scp.transform(
        counts, scp.ShiftedLog(count_shift=0.7), dtype="float32"
    )

    assert transformed.dtype == np.dtype("float32")
    assert transformed.materialize().dtype == np.float32
    assert (transformed @ np.ones(counts.shape[1])).dtype == np.float32
    result = transformed.pca(n_comps=2)
    assert result.scores.dtype == np.float32
    assert result.components.dtype == np.float32


def test_materialize_accepts_arbitrary_positional_selections_and_out(
    counts, tmp_path
):
    """Materialization supports ordered selections, scalar indices, and out."""
    transformed = scp.transform(counts, scp.ShiftedLog(count_shift=0.7))
    expected = np.log1p(counts.toarray() / 0.7)
    out = np.memmap(
        tmp_path / "subset.dat", mode="w+", shape=(3, 2), dtype=np.float32
    )

    actual = transformed.materialize(
        obs=[4, 1, 4], var=[3, 0], out=out, block_size=2
    )

    assert actual is out
    np.testing.assert_allclose(actual, expected[[4, 1, 4]][:, [3, 0]])
    assert transformed.materialize(obs=2, var=1).shape == (1, 1)


def test_materialize_accepts_anndata_names(adata):
    """Transforms built from AnnData accept observation and variable names."""
    transformed = scp.transform(adata, scp.ShiftedLog(count_shift=1.0))
    expected = np.log1p(adata.X.toarray())

    actual = transformed.materialize(
        obs=[adata.obs_names[3], adata.obs_names[0]],
        var=[adata.var_names[2], adata.var_names[1]],
    )

    np.testing.assert_allclose(actual, expected[[3, 0]][:, [2, 1]])


def test_full_variable_access_warns_for_csr_backend(counts):
    """Selecting variables across every row warns about CSR access cost."""
    transformed = scp.transform(counts, scp.ShiftedLog(count_shift=1.0))

    with pytest.warns(sparse.SparseEfficiencyWarning, match="Selecting variables"):
        column = transformed.materialize(var=1)

    assert column.shape == (counts.shape[0], 1)


def test_materialize_validates_out_and_block_size(counts):
    """Materialization validates destination shape, dtype, and block size."""
    transformed = scp.transform(counts, scp.ShiftedLog(count_shift=1.0))

    with pytest.raises(ValueError, match="out must have shape"):
        transformed.materialize(obs=[0, 1], out=np.empty((1, counts.shape[1])))
    with pytest.raises(TypeError, match="floating-point"):
        transformed.materialize(out=np.empty(counts.shape, dtype=np.int64))
    with pytest.raises(ValueError, match="positive integer"):
        transformed.materialize(block_size=0)


def test_residual_transform_matches_dense_oracle(counts):
    """The two-step residual transform matches the independent dense oracle."""
    transformed = scp.transform(counts, scp.Residual())

    np.testing.assert_allclose(
        transformed.materialize(),
        _materialize_dense_residual(counts),
        atol=1e-12,
    )


def test_transformed_pca_matches_one_step_matrix_api(counts):
    """PCA of a fitted transform matches its existing one-step wrapper."""
    transformed = scp.transform(counts, scp.ShiftedCLR(count_shift=0.7))
    actual = transformed.pca(n_comps=2)
    expected = scp.shifted_clr_pca_matrix(
        counts, n_comps=2, count_shift=0.7
    )

    np.testing.assert_allclose(actual.singular_values, expected.singular_values)
    np.testing.assert_allclose(actual.components, expected.components)


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
        actual.singular_values, expected.uns["pca"]["singular_values"]
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
