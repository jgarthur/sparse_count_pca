from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike, NDArray
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
    _canonicalize_counts,
    _validate_boolean_mask,
)
from ._operator import SparseLowRankLinearOperator, _normalize_operator_dtype
from ._representation import SparseLowRankMatrix
from ._svd import Solver, compute_truncated_svd
from ._version import __version__

Float64Array = NDArray[np.float64]


@dataclass
class CorrespondenceAnalysisResult:
    """Principal-coordinate results from correspondence analysis."""

    row_standard_coordinates: Float64Array
    row_principal_coordinates: Float64Array
    column_standard_coordinates: Float64Array
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
) -> tuple[SparseLowRankMatrix, Float64Array, Float64Array]:
    """Build the canonical CA standardized-residual matrix."""
    row_totals = np.asarray(X.astype(np.float64).sum(axis=1)).ravel()
    column_totals = np.asarray(X.astype(np.float64).sum(axis=0)).ravel()
    total = float(np.sum(row_totals))
    row_masses = row_totals / total
    column_masses = column_totals / total
    support_rows = np.repeat(
        np.arange(X.shape[0], dtype=np.intp), np.diff(X.indptr)
    )
    denominator = np.sqrt(
        total * row_totals[support_rows] * column_masses[X.indices]
    )
    sparse_part = sparse.csr_matrix(
        (
            X.data.astype(np.float64, copy=False) / denominator,
            X.indices.copy(),
            X.indptr.copy(),
        ),
        shape=X.shape,
    )
    left = -np.sqrt(row_masses)
    right = np.sqrt(column_masses)
    return SparseLowRankMatrix(sparse_part, left, right), row_masses, column_masses


def _compute_correspondence_analysis(
    X: CountMatrix,
    n_comps: int,
    *,
    mask: BoolArray | None,
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
    n_obs, n_vars = counts.shape
    if mask is not None:
        mask = _validate_boolean_mask(mask, n_vars, name="mask")
        if not mask.any():
            raise ValueError("mask selected zero columns")
        counts = counts[:, mask].tocsr()
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

    representation, row_masses, column_masses = build_correspondence_representation(
        counts
    )
    operator = SparseLowRankLinearOperator(
        representation, center=False, dtype=dtype
    )
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
    row_standard = left / np.sqrt(row_masses)[:, None]
    column_standard = right / np.sqrt(column_masses)[:, None]
    row_principal = row_standard * singular_values
    column_principal = column_standard * singular_values
    inertias = singular_values**2
    params = {
        "analysis": "correspondence_analysis",
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
        row_standard_coordinates=row_standard,
        row_principal_coordinates=row_principal,
        column_standard_coordinates=column_standard,
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
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> CorrespondenceAnalysisResult:
    """Compute correspondence analysis of a contingency table."""
    return _compute_correspondence_analysis(
        X,
        n_comps,
        mask=None,
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
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute correspondence analysis and store principal coordinates."""
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
    result = _compute_correspondence_analysis(
        X,
        n_comps,
        mask=mask,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
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
