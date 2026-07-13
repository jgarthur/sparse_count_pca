"""Tests for residual PCA at the AnnData boundary."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from sparse_count_pca import __version__, residual_pca, residual_pca_matrix


def test_default_outputs_and_copy(adata):
    """AnnData residual PCA honors default output keys and copy semantics."""
    copied = residual_pca(adata, n_comps=2, copy=True, dtype="float64")
    assert copied is not adata
    assert "X_pca" in copied.obsm
    assert "PCs" in copied.varm
    assert "pca" in copied.uns
    assert "X_pca" not in adata.obsm

    returned = residual_pca(adata, n_comps=2, dtype="float64")
    assert returned is None
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 2)
    assert adata.varm["PCs"].shape == (adata.n_vars, 2)


def test_default_anndata_dtype_is_float64(adata):
    """AnnData residual PCA stores float64 outputs by default."""
    result = residual_pca(adata, n_comps=2, copy=True)
    assert result.obsm["X_pca"].dtype == np.float64
    assert result.varm["PCs"].dtype == np.float64
    assert result.uns["pca"]["params"]["dtype"] == "float64"


def test_key_added_and_layer(adata):
    """Custom output keys and count layers are respected."""
    expected = residual_pca_matrix(adata.layers["counts"], 2, dtype="float64")
    residual_pca(
        adata,
        2,
        layer="counts",
        key_added="resid_pca",
        dtype="float64",
    )
    np.testing.assert_allclose(adata.obsm["resid_pca"], expected.scores)
    assert "X_resid_pca" not in adata.obsm
    assert "resid_pca" in adata.varm
    assert adata.uns["resid_pca"]["params"]["layer"] == "counts"


def test_clipping_params_are_recorded(adata):
    """AnnData PCA metadata records clipping parameters."""
    result = residual_pca(
        adata,
        2,
        clip=1.0,
        clip_mode="upper",
        clip_max_nnz_ratio=None,
        copy=True,
        dtype="float64",
    )
    params = result.uns["pca"]["params"]
    assert params["clip"] == 1.0
    assert params["clip_mode"] == "upper"
    assert params["clip_max_nnz_ratio"] is None


def test_default_and_explicit_masks(adata):
    """Default and explicit variable masks select the intended PCA genes."""
    adata.var["highly_variable"] = [True, False, True, True]
    residual_pca(adata, 2, dtype="float64")
    assert np.isnan(adata.varm["PCs"][1]).all()
    assert adata.uns["pca"]["params"]["mask_var"] == "highly_variable"
    assert adata.uns["pca"]["params"]["use_highly_variable"] is True
    assert adata.uns["pca"]["params"]["mask_var_details"] == {
        "kind": "default",
        "key": "highly_variable",
        "n_vars_used": 3,
    }

    all_genes = residual_pca(adata, 2, mask_var=None, copy=True, dtype="float64")
    assert np.isfinite(all_genes.varm["PCs"]).all()
    assert all_genes.uns["pca"]["params"]["mask_var"] is None
    assert all_genes.uns["pca"]["params"]["mask_var_details"]["kind"] == "none"

    explicit = residual_pca(
        adata,
        2,
        mask_var=np.array([True, True, True, False]),
        copy=True,
        dtype="float64",
    )
    assert np.isnan(explicit.varm["PCs"][3]).all()
    np.testing.assert_array_equal(
        explicit.uns["pca"]["params"]["mask_var"],
        np.array([True, True, True, False]),
    )
    assert explicit.uns["pca"]["params"]["mask_var_details"]["kind"] == "array"


def test_reproducibility_metadata_matches_matrix_result(adata):
    """AnnData and matrix APIs record matching reproducibility metadata."""
    matrix = residual_pca_matrix(
        adata.X,
        2,
        random_state=7,
        tol=1e-6,
        check_values=False,
    )
    annotated = residual_pca(
        adata,
        2,
        random_state=7,
        tol=1e-6,
        check_values=False,
        copy=True,
    )
    for params in (matrix.params, annotated.uns["pca"]["params"]):
        assert params["zero_center"] is True
        assert params["n_comps"] == 2
        assert params["random_state"] == 7
        assert params["tol"] == 1e-6
        assert params["check_values"] is False
        assert params["dtype"] == "float64"
        assert params["package_version"] == __version__


@pytest.mark.parametrize(
    "mask",
    [
        np.array([1, 1, 1, 0]),
        np.array([1.0, 1.0, 1.0, 0.0]),
        np.array([True, True, True, np.nan], dtype=object),
        np.array(["yes", "yes", "yes", ""]),
        pd.array([True, True, True, pd.NA], dtype="boolean"),
    ],
)
def test_mask_var_requires_genuinely_boolean_array(adata, mask):
    """Direct variable masks must have a genuinely boolean dtype."""
    with pytest.raises(ValueError, match="genuinely boolean"):
        residual_pca(adata, 2, mask_var=mask)


@pytest.mark.parametrize(
    "values",
    [
        [1, 1, 1, 0],
        [1.0, 1.0, 1.0, 0.0],
        ["yes", "yes", "yes", ""],
        pd.array([True, True, True, pd.NA], dtype="boolean"),
    ],
)
def test_var_mask_columns_require_genuinely_boolean_dtype(adata, values):
    """Variable-metadata mask columns must be genuinely boolean."""
    adata.var["selected"] = values
    with pytest.raises(ValueError, match="genuinely boolean"):
        residual_pca(adata, 2, mask_var="selected")


def test_deprecated_highly_variable_alias(adata):
    """The deprecated highly-variable alias warns and preserves behavior."""
    adata.var["highly_variable"] = [True, True, True, False]
    with pytest.warns(FutureWarning):
        result = residual_pca(
            adata,
            2,
            use_highly_variable=True,
            copy=True,
            dtype="float64",
        )
    assert np.isnan(result.varm["PCs"][3]).all()

    with (
        pytest.warns(FutureWarning),
        pytest.raises(ValueError, match="Cannot specify both"),
    ):
        residual_pca(
            adata,
            2,
            mask_var=None,
            use_highly_variable=False,
        )


def test_use_raw_aligns_current_var_names(adata):
    """Raw count selection is reordered and subset to current variables."""
    # Raw contains an extra leading gene; selection must discard it and align the
    # remaining columns to the current adata.var_names before PCA.
    raw = AnnData(
        sparse.csr_matrix(
            [
                [9, 5, 1, 0, 2],
                [9, 1, 4, 2, 0],
                [9, 0, 2, 5, 1],
                [9, 3, 0, 1, 4],
                [9, 2, 3, 0, 2],
                [9, 1, 1, 3, 2],
            ]
        )
    )
    raw.var_names = ["extra", "a", "b", "c", "d"]
    adata.raw = raw
    expected = residual_pca_matrix(raw[:, adata.var_names].X, 2, dtype="float64")
    result = residual_pca(adata, 2, use_raw=True, copy=True, dtype="float64")
    np.testing.assert_allclose(
        result.uns["pca"]["singular_values"], expected.singular_values
    )
    assert result.varm["PCs"].shape == (adata.n_vars, 2)


def test_use_raw_validation(adata):
    """Raw count selection rejects conflicts and missing variable names."""
    with pytest.raises(ValueError, match="adata.raw is None"):
        residual_pca(adata, 2, use_raw=True)
    with pytest.raises(ValueError, match="Specify only one"):
        residual_pca(adata, 2, use_raw=True, layer="counts")

    raw = AnnData(adata.X[:, :3].copy())
    raw.var_names = adata.var_names[:3]
    adata.raw = raw
    with pytest.raises(ValueError, match="all current"):
        residual_pca(adata, 2, use_raw=True)


def test_backed_sparse_x(adata, tmp_path):
    """Backed sparse counts produce the same result as in-memory counts."""
    path = tmp_path / "counts.h5ad"
    adata.write_h5ad(path)
    expected = residual_pca_matrix(adata.X, 2, dtype="float64")

    backed = ad.read_h5ad(path, backed="r")
    try:
        matrix_result = residual_pca_matrix(backed.X, 2, dtype="float64")
        returned = residual_pca(backed, 2, copy=False, dtype="float64")
        inplace_singular_values = backed.uns["pca"]["singular_values"].copy()
        assert returned is None
        assert backed.isbacked
        result = residual_pca(backed, 2, copy=True, dtype="float64")
    finally:
        backed.file.close()

    assert not result.isbacked
    np.testing.assert_allclose(matrix_result.singular_values, expected.singular_values)
    np.testing.assert_allclose(inplace_singular_values, expected.singular_values)
    np.testing.assert_allclose(
        result.uns["pca"]["singular_values"],
        expected.singular_values,
    )


def test_scaled_nb_alpha_from_var_and_array_shape(adata):
    """AnnData scaled-NB alpha resolves from metadata and validates shape."""
    adata.var["dispersion"] = [0.0, 0.1, 0.2, 0.3]
    result = residual_pca(
        adata,
        2,
        model="scaled_nb",
        alpha="dispersion",
        copy=True,
        dtype="float64",
    )
    assert result.uns["pca"]["params"]["alpha"] == "dispersion"

    with pytest.raises(ValueError, match="before masking"):
        residual_pca(
            adata,
            2,
            model="scaled_nb",
            alpha=np.array([0.1, 0.2, 0.3]),
            mask_var=np.array([True, True, True, False]),
        )


def test_scaled_nb_accepts_zero_dimensional_numpy_alpha_in_anndata(adata):
    """AnnData scaled-NB PCA accepts zero-dimensional NumPy alpha values."""
    result = residual_pca(
        adata,
        2,
        model="scaled_nb",
        alpha=np.array(0.1),
        copy=True,
        dtype="float64",
    )
    assert result.uns["pca"]["params"]["alpha"] == 0.1
