from __future__ import annotations

from typing import Any, Literal, TypeAlias

import numpy as np
from anndata import AnnData
from numpy.typing import ArrayLike, DTypeLike, NDArray
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

DirichletTransform = Literal["dirichlet_log", "dirichlet_clr"]
PriorProportions: TypeAlias = ArrayLike | None
Float64Array = NDArray[np.float64]


def _validate_dirichlet_prior(
    concentration: float,
    prior_proportions: PriorProportions,
    n_vars: int,
) -> tuple[float, Float64Array]:
    concentration = float(concentration)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    if prior_proportions is None:
        proportions = np.full(n_vars, 1.0 / n_vars, dtype=np.float64)
    else:
        proportions = np.asarray(prior_proportions, dtype=np.float64)
        if proportions.ndim != 1 or proportions.shape[0] != n_vars:
            raise ValueError("prior_proportions must have shape (n_vars,)")
        if not np.isfinite(proportions).all():
            raise ValueError("prior_proportions must be finite")
        if (proportions <= 0).any():
            raise ValueError("prior_proportions must be strictly positive")
        total = float(np.sum(proportions))
        if not np.isclose(total, 1.0, rtol=1e-8, atol=1e-12):
            raise ValueError("prior_proportions must sum to one")
        proportions = proportions / total
    return concentration, proportions


def _dirichlet_sparse_part(
    X: sparse.csr_matrix,
    prior_counts: Float64Array,
) -> sparse.csr_matrix:
    data = np.log1p(X.data.astype(np.float64, copy=False) / prior_counts[X.indices])
    return sparse.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)


def build_dirichlet_log_representation(
    X: sparse.csr_matrix,
    *,
    concentration: float,
    prior_proportions: PriorProportions = None,
) -> SparseLowRankMatrix:
    """Represent log Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = _validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    sparse_part = _dirichlet_sparse_part(X, prior_counts)
    row_totals = np.asarray(X.astype(np.float64).sum(axis=1)).ravel()
    log_prior_counts = np.log(prior_counts)
    log_denominator = np.log(row_totals + concentration)
    left = np.column_stack(
        (
            np.ones(X.shape[0], dtype=np.float64),
            -log_denominator,
        )
    )
    right = np.column_stack(
        (
            log_prior_counts,
            np.ones(X.shape[1], dtype=np.float64),
        )
    )
    return SparseLowRankMatrix(sparse_part, left, right)


def build_dirichlet_clr_representation(
    X: sparse.csr_matrix,
    *,
    concentration: float,
    prior_proportions: PriorProportions = None,
) -> SparseLowRankMatrix:
    """Represent CLR of Dirichlet posterior-mean proportions implicitly."""
    concentration, proportions = _validate_dirichlet_prior(
        concentration, prior_proportions, X.shape[1]
    )
    prior_counts = concentration * proportions
    sparse_part = _dirichlet_sparse_part(X, prior_counts)
    sparse_row_mean = np.asarray(sparse_part.sum(axis=1)).ravel() / X.shape[1]
    log_prior_counts = np.log(prior_counts)
    centered_log_prior = log_prior_counts - np.mean(log_prior_counts)
    left = np.column_stack(
        (
            -sparse_row_mean,
            np.ones(X.shape[0], dtype=np.float64),
        )
    )
    right = np.column_stack(
        (
            np.ones(X.shape[1], dtype=np.float64),
            centered_log_prior,
        )
    )
    return SparseLowRankMatrix(sparse_part, left, right)


def _compute_dirichlet_pca(
    X: CountMatrix,
    n_comps: int,
    *,
    transform: DirichletTransform,
    mask: BoolArray | None,
    concentration: float,
    prior_proportions: PriorProportions,
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
    concentration, proportions = _validate_dirichlet_prior(
        concentration, prior_proportions, n_vars
    )
    if transform == "dirichlet_log":
        representation = build_dirichlet_log_representation(
            counts,
            concentration=concentration,
            prior_proportions=proportions,
        )
    else:
        representation = build_dirichlet_clr_representation(
            counts,
            concentration=concentration,
            prior_proportions=proportions,
        )
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
        label = "Dirichlet log" if transform == "dirichlet_log" else "Dirichlet CLR"
        raise ValueError(
            f"{label} matrix has numerically zero centered variance; "
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
        "transform": transform,
        "concentration": concentration,
        "prior_proportions": (
            "uniform" if prior_proportions is None else proportions.copy()
        ),
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


def dirichlet_log_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    concentration: float = 1.0,
    prior_proportions: PriorProportions = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of implicit log Dirichlet posterior-mean proportions."""
    return _compute_dirichlet_pca(
        X,
        n_comps,
        transform="dirichlet_log",
        mask=None,
        concentration=concentration,
        prior_proportions=prior_proportions,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def dirichlet_clr_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    concentration: float = 1.0,
    prior_proportions: PriorProportions = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of implicit CLR Dirichlet posterior-mean proportions."""
    return _compute_dirichlet_pca(
        X,
        n_comps,
        transform="dirichlet_clr",
        mask=None,
        concentration=concentration,
        prior_proportions=prior_proportions,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def _resolve_prior_proportions(
    prior_proportions: PriorProportions | str,
    adata: AnnData,
) -> PriorProportions:
    if prior_proportions is None:
        return None
    if isinstance(prior_proportions, str):
        if prior_proportions not in adata.var:
            raise KeyError(f"{prior_proportions!r} not found in adata.var")
        return np.asarray(adata.var[prior_proportions], dtype=np.float64)
    values = np.asarray(prior_proportions)
    if values.ndim != 1 or values.shape[0] != adata.n_vars:
        raise ValueError(
            "AnnData prior_proportions arrays must have shape (adata.n_vars,)"
        )
    return values


def _serialize_prior_proportions(
    prior_proportions: PriorProportions | str,
) -> Any:
    if prior_proportions is None:
        return "uniform"
    if isinstance(prior_proportions, str):
        return prior_proportions
    return np.asarray(prior_proportions)


def _dirichlet_pca_anndata(
    adata: AnnData,
    n_comps: int,
    *,
    transform: DirichletTransform,
    layer: str | None,
    use_raw: bool,
    mask_var: Any,
    use_highly_variable: bool | None,
    key_added: str | None,
    concentration: float,
    prior_proportions: PriorProportions | str,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    copy: bool,
) -> AnnData | None:
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
    resolved_prior = _resolve_prior_proportions(prior_proportions, adata)
    result = _compute_dirichlet_pca(
        X,
        n_comps,
        transform=transform,
        mask=mask,
        concentration=concentration,
        prior_proportions=resolved_prior,
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
            "prior_proportions": _serialize_prior_proportions(prior_proportions),
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


def dirichlet_log_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    concentration: float = 1.0,
    prior_proportions: PriorProportions | str = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute implicit Dirichlet-log PCA and write Scanpy-style outputs."""
    return _dirichlet_pca_anndata(
        adata,
        n_comps,
        transform="dirichlet_log",
        layer=layer,
        use_raw=use_raw,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        concentration=concentration,
        prior_proportions=prior_proportions,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        copy=copy,
    )


def dirichlet_clr_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    concentration: float = 1.0,
    prior_proportions: PriorProportions | str = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute implicit Dirichlet-CLR PCA and write Scanpy-style outputs."""
    return _dirichlet_pca_anndata(
        adata,
        n_comps,
        transform="dirichlet_clr",
        layer=layer,
        use_raw=use_raw,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        concentration=concentration,
        prior_proportions=prior_proportions,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        copy=copy,
    )
