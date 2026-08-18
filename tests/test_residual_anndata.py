"""Tests for residual PCA at the AnnData boundary."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sparse_count_pca import __version__, residual_pca, residual_pca_matrix


def test_default_outputs_and_copy(adata):
    """AnnData residual PCA honors default output keys, dtype, and copy semantics."""
    copied = residual_pca(adata, n_comps=2, copy=True)
    assert copied is not adata
    assert "X_pca" in copied.obsm
    assert "PCs" in copied.varm
    assert "pca" in copied.uns
    assert "X_pca" not in adata.obsm
    assert copied.obsm["X_pca"].dtype == np.float64
    assert copied.varm["PCs"].dtype == np.float64
    params = copied.uns["pca"]["params"]
    assert params["dtype"] == "float64"
    assert params["normalization_n_vars"] == adata.n_vars
    assert params["pca_n_vars"] == adata.n_vars
    assert params["mask_var_details"] == {
        "kind": "default",
        "key": None,
        "n_vars_used": adata.n_vars,
    }

    returned = residual_pca(adata, n_comps=2, dtype="float64")
    assert returned is None
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 2)
    assert adata.varm["PCs"].shape == (adata.n_vars, 2)


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
    np.testing.assert_allclose(
        adata.obsm["resid_pca"], expected.scores, rtol=0.0, atol=1e-12
    )
    assert "X_resid_pca" not in adata.obsm
    assert "resid_pca" in adata.varm
    assert adata.uns["resid_pca"]["params"]["layer"] == "counts"


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
    assert adata.uns["pca"]["params"]["normalization_n_vars"] == adata.n_vars
    assert adata.uns["pca"]["params"]["pca_n_vars"] == 3

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
    keywords = dict(
        random_state=7,
        tol=1e-6,
        check_values=False,
        clip="scanpy",
        clip_mode="upper",
        clip_max_nnz_ratio=None,
    )
    matrix = residual_pca_matrix(adata.X, 2, **keywords)
    annotated = residual_pca(adata, 2, copy=True, **keywords)

    for params in (matrix.params, annotated.uns["pca"]["params"]):
        assert params["zero_center"] is True
        assert params["n_comps"] == 2
        assert params["random_state"] == 7
        assert params["tol"] == 1e-6
        assert params["check_values"] is False
        assert params["dtype"] == "float64"
        assert params["package_version"] == __version__
        assert params["clip"] == "scanpy"
        assert params["clip_threshold"] == pytest.approx(np.sqrt(adata.n_obs))
        assert params["clip_mode"] == "upper"
        assert params["clip_max_nnz_ratio"] is None


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
    ("mask", "error", "message"),
    [
        ("missing", KeyError, "not found"),
        (np.zeros(4, dtype=bool), ValueError, "selected zero genes"),
        (np.ones(3, dtype=bool), ValueError, "length 4"),
    ],
)
def test_mask_var_reports_selector_errors(adata, mask, error, message):
    """Missing, empty, and wrong-length masks produce specific errors."""
    with pytest.raises(error, match=message):
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
        assert "X_pca" in backed.obsm
        assert "PCs" in backed.varm
        assert "pca" in backed.uns
        np.testing.assert_allclose(
            backed.obsm["X_pca"], matrix_result.scores, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            backed.varm["PCs"], matrix_result.components.T, rtol=0.0, atol=1e-12
        )
        result = residual_pca(backed, 2, copy=True, dtype="float64")
    finally:
        backed.file.close()

    reopened = ad.read_h5ad(path, backed="r")
    try:
        assert "X_pca" not in reopened.obsm
        assert "PCs" not in reopened.varm
        assert "pca" not in reopened.uns
    finally:
        reopened.file.close()

    assert not result.isbacked
    np.testing.assert_allclose(
        matrix_result.singular_values,
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        inplace_singular_values,
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.uns["pca"]["singular_values"],
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
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
