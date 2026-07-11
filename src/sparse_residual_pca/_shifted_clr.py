from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike
from scipy import sparse

from ._anndata import (
    _empty,
    _resolve_mask_var,
    _scanpy_mask_params,
    _serialize_mask_var,
)
from ._matrix import (
    BoolArray,
    CountMatrix,
    PCAResult,
    _canonicalize_counts,
    _validate_boolean_mask,
)
from ._operator import SparseLowRankLinearOperator, _normalize_operator_dtype
from ._representation import SparseLowRankMatrix
from ._svd import Solver, compute_truncated_svd
from ._version import __version__


def build_shifted_clr_representation(
    X: sparse.csr_matrix,
    *,
    pseudocount: float,
) -> SparseLowRankMatrix:
    """Build shifted CLR as a sparse-plus-rank-one matrix."""
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and positive")
    row_totals = np.asarray(X.astype(np.float64).sum(axis=1)).ravel()
    if (row_totals == 0).any():
        raise ValueError("Cells with zero total counts are not supported")
    rows = np.repeat(np.arange(X.shape[0], dtype=np.intp), np.diff(X.indptr))
    proportions = X.data.astype(np.float64, copy=False) / row_totals[rows]
    transformed_data = np.log1p(proportions / pseudocount)
    shifted_log = sparse.csr_matrix(
        (transformed_data, X.indices.copy(), X.indptr.copy()), shape=X.shape
    )
    row_mean = np.asarray(shifted_log.sum(axis=1)).ravel() / X.shape[1]
    return SparseLowRankMatrix(
        shifted_log,
        -row_mean,
        np.ones(X.shape[1], dtype=np.float64),
    )


def _compute_shifted_clr_pca(
    X: CountMatrix,
    n_comps: int,
    *,
    mask: BoolArray | None,
    pseudocount: float,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> PCAResult:
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    dtype = _normalize_operator_dtype(dtype)
    counts = _canonicalize_counts(X, check_values=check_values)
    n_obs, n_vars = counts.shape
    representation = build_shifted_clr_representation(counts, pseudocount=pseudocount)
    if mask is not None:
        mask = _validate_boolean_mask(mask, n_vars, name="mask")
        if not mask.any():
            raise ValueError("mask selected zero genes")
        representation = representation.select_columns(mask)
    n_vars_used = representation.shape[1]
    if not 1 <= n_comps < min(n_obs, n_vars_used):
        raise ValueError("n_comps must satisfy 1 <= n_comps < min(n_obs, n_vars_used)")
    operator = SparseLowRankLinearOperator(representation, center=True, dtype=dtype)
    centered_squared = operator.frobenius_squared_centered()
    if operator.centered_variance_is_numerically_zero():
        raise ValueError(
            "Shifted CLR matrix has numerically zero centered variance; "
            "PCA directions are undefined"
        )
    decomposition = compute_truncated_svd(
        operator,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    singular_values = decomposition.singular_values
    scores = np.asarray(decomposition.left_vectors * singular_values, dtype=dtype)
    components = np.asarray(decomposition.right_vectors, dtype=dtype)
    total_variance = centered_squared / (n_obs - 1)
    explained_variance = singular_values**2 / (n_obs - 1)
    params = {
        "transform": "shifted_clr",
        "pseudocount": float(pseudocount),
        "normalization_n_vars": n_vars,
        "zero_center": True,
        "n_comps": n_comps,
        "solver": solver,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": str(dtype),
        "package_version": __version__,
    }
    assert operator.mean is not None
    return PCAResult(
        scores=scores,
        components=components,
        loadings=components.T,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance / total_variance,
        mean=operator.mean.copy(),
        total_variance=float(total_variance),
        params=params,
        operator=operator if return_operator else None,
    )


def shifted_clr_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    pseudocount: float = 1.0,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of the implicit shifted CLR transform."""
    return _compute_shifted_clr_pca(
        X,
        n_comps,
        mask=None,
        pseudocount=pseudocount,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def shifted_clr_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    pseudocount: float = 1.0,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute implicit shifted CLR PCA and write Scanpy-style outputs."""
    if copy:
        adata = adata.to_memory() if adata.isbacked else adata.copy()
    if use_raw and layer is not None:
        raise ValueError("Specify only one of use_raw=True or layer=...")
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True, but adata.raw is None")
        missing = adata.var_names.difference(adata.raw.var_names)
        if len(missing) > 0:
            raise ValueError(
                "use_raw=True requires all current adata.var_names to be present "
                "in adata.raw.var_names"
            )
        X = adata.raw[:, adata.var_names].X
    else:
        X = adata.layers[layer] if layer is not None else adata.X
    mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    result = _compute_shifted_clr_pca(
        X,
        n_comps,
        mask=mask,
        pseudocount=pseudocount,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
    if key_added is None:
        obsm_key, varm_key, uns_key = "X_pca", "PCs", "pca"
    else:
        obsm_key = varm_key = uns_key = key_added
    loadings = np.full((adata.n_vars, n_comps), np.nan, dtype=result.loadings.dtype)
    loadings[mask] = result.loadings
    adata.obsm[obsm_key] = result.scores
    adata.varm[varm_key] = loadings
    standard_hv, standard_mask = _scanpy_mask_params(
        mask_var, use_highly_variable, mask, adata.var
    )
    params = dict(result.params)
    params.update(
        {
            "layer": layer,
            "use_raw": use_raw,
            "mask_var": standard_mask,
            "use_highly_variable": standard_hv,
            "mask_var_details": _serialize_mask_var(mask_var, mask, adata.var),
        }
    )
    adata.uns[uns_key] = {
        "variance": result.explained_variance,
        "variance_ratio": result.explained_variance_ratio,
        "singular_values": result.singular_values,
        "params": params,
    }
    return adata if copy else None
