from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike, NDArray
from scipy import sparse

from ._anndata import (
    _empty,
    _get_count_matrix,
    _resolve_mask_var,
    _scanpy_mask_params,
    _serialize_mask_var,
)
from ._counts import (
    BoolArray,
    CountMatrix,
    _canonicalize_counts,
    _validate_boolean_mask,
)
from ._operator import SparseLowRankLinearOperator, _normalize_operator_dtype
from ._representation import SparseLowRankMatrix
from ._residual_pca import _resolve_alpha, _serialize_alpha
from ._residuals import (
    AlphaLike,
    _validate_model,
    build_pearson_residual_representation,
)
from ._svd import Solver, compute_truncated_svd
from ._version import __version__

Float64Array = NDArray[np.float64]
CorrespondenceModel: TypeAlias = Literal["poisson", "scaled_nb"]


@dataclass
class CorrespondenceAnalysisResult:
    """Principal-coordinate results from classical or experimental analysis.

    Standard coordinates are derivable by dividing principal coordinates by
    singular values and are intentionally not stored.
    """

    row_principal_coordinates: Float64Array
    column_principal_coordinates: Float64Array
    singular_values: Float64Array
    principal_inertias: Float64Array
    inertia_ratio: Float64Array
    total_inertia: float
    row_masses: Float64Array
    column_masses: Float64Array
    params: dict[str, Any]
    operator: SparseLowRankLinearOperator | None = None


def build_correspondence_representation(
    X: sparse.csr_matrix,
    *,
    model: CorrespondenceModel = "poisson",
    alpha: Float64Array | None = None,
) -> tuple[SparseLowRankMatrix, Float64Array, Float64Array]:
    """Build a total-scaled Pearson-residual representation."""
    row_totals = np.asarray(X.astype(np.float64).sum(axis=1)).ravel()
    column_totals = np.asarray(X.astype(np.float64).sum(axis=0)).ravel()
    total = float(np.sum(row_totals))
    row_masses = row_totals / total
    column_masses = column_totals / total
    pearson = build_pearson_residual_representation(
        X,
        row_totals,
        column_masses,
        model=model,
        alpha=alpha,
    )
    return pearson.scaled(1.0 / np.sqrt(total)), row_masses, column_masses


def _compute_correspondence_analysis(
    X: CountMatrix,
    n_comps: int,
    *,
    mask: BoolArray | None,
    model: CorrespondenceModel,
    alpha: AlphaLike,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> CorrespondenceAnalysisResult:
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    dtype = _normalize_operator_dtype(dtype)
    counts = _canonicalize_counts(X, check_values=check_values)
    n_vars = counts.shape[1]
    if model not in {"poisson", "scaled_nb"}:
        raise ValueError("model must be 'poisson' or 'scaled_nb'")
    alpha_full = _validate_model(model, "pearson", alpha, n_vars)
    if mask is not None:
        mask = _validate_boolean_mask(mask, n_vars, name="mask")
        if not mask.any():
            raise ValueError("mask selected zero columns")
        counts = counts[:, mask].tocsr()
        if alpha_full is not None:
            alpha_full = alpha_full[mask]
    if not 1 <= n_comps < min(counts.shape):
        raise ValueError(
            "n_comps must satisfy 1 <= n_comps < min(n_rows, n_columns_used)"
        )
    row_totals = np.asarray(counts.astype(np.float64).sum(axis=1)).ravel()
    column_totals = np.asarray(counts.astype(np.float64).sum(axis=0)).ravel()
    if (row_totals == 0).any():
        raise ValueError("Rows with zero mass are not supported")
    if (column_totals == 0).any():
        raise ValueError("Columns with zero mass are not supported")
    if model == "scaled_nb":
        warnings.warn(
            "model='scaled_nb' correspondence analysis is experimental; its "
            "inertia has no classical Pearson chi-square interpretation",
            UserWarning,
            stacklevel=3,
        )

    representation, row_masses, column_masses = build_correspondence_representation(
        counts,
        model=model,
        alpha=alpha_full,
    )
    operator = SparseLowRankLinearOperator(representation, center=False, dtype=dtype)
    total_inertia = operator.frobenius_squared_uncentered()
    eps = np.finfo(dtype).eps
    if total_inertia <= eps * eps * counts.shape[0] * counts.shape[1]:
        raise ValueError(
            "Contingency table has numerically zero inertia; "
            "correspondence axes are undefined"
        )
    decomposition = compute_truncated_svd(
        operator,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    left = decomposition.left_vectors.astype(np.float64, copy=False)
    right = decomposition.right_vectors.T.astype(np.float64, copy=False)
    singular_values = decomposition.singular_values
    row_principal = left * singular_values[None, :] / np.sqrt(row_masses)[:, None]
    column_principal = (
        right * singular_values[None, :] / np.sqrt(column_masses)[:, None]
    )
    inertias = singular_values**2
    params = {
        "analysis": "correspondence_analysis",
        "model": model,
        "alpha": _serialize_alpha(alpha),
        "experimental": model == "scaled_nb",
        "inertia_interpretation": (
            "pearson_chi_squared_over_grand_total"
            if model == "poisson"
            else "scaled_nb_pearson_residual_inertia"
        ),
        "zero_center": False,
        "n_comps": n_comps,
        "solver": solver,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": str(dtype),
        "package_version": __version__,
    }
    return CorrespondenceAnalysisResult(
        row_principal_coordinates=row_principal,
        column_principal_coordinates=column_principal,
        singular_values=singular_values,
        principal_inertias=inertias,
        inertia_ratio=inertias / total_inertia,
        total_inertia=float(total_inertia),
        row_masses=row_masses,
        column_masses=column_masses,
        params=params,
        operator=operator if return_operator else None,
    )


def correspondence_analysis_matrix(
    X: CountMatrix,
    n_comps: int = 2,
    *,
    model: CorrespondenceModel = "poisson",
    alpha: AlphaLike = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> CorrespondenceAnalysisResult:
    """Compute classical or experimental scaled-NB correspondence analysis."""
    return _compute_correspondence_analysis(
        X,
        n_comps,
        mask=None,
        model=model,
        alpha=alpha,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def correspondence_analysis(
    adata: AnnData,
    n_comps: int = 2,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    model: CorrespondenceModel = "poisson",
    alpha: AlphaLike | str = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute classical or experimental scaled-NB correspondence analysis."""
    if copy:
        adata = adata.to_memory() if adata.isbacked else adata.copy()
    X = _get_count_matrix(adata, layer=layer, use_raw=use_raw)
    mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    alpha_values = _resolve_alpha(alpha, adata, model)
    result = _compute_correspondence_analysis(
        X,
        n_comps,
        mask=mask,
        model=model,
        alpha=alpha_values,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
    result.params["alpha"] = _serialize_alpha(alpha)
    if key_added is None:
        obsm_key, varm_key, uns_key = "X_ca", "CA", "ca"
    else:
        obsm_key = varm_key = uns_key = key_added
    columns = np.full((adata.n_vars, n_comps), np.nan, dtype=np.float64)
    columns[mask] = result.column_principal_coordinates
    adata.obsm[obsm_key] = result.row_principal_coordinates
    adata.varm[varm_key] = columns
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
        "singular_values": result.singular_values,
        "principal_inertias": result.principal_inertias,
        "inertia_ratio": result.inertia_ratio,
        "total_inertia": result.total_inertia,
        "row_masses": result.row_masses,
        "column_masses": result.column_masses,
        "params": params,
    }
    return adata if copy else None
