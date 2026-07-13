"""Dirichlet log and CLR PCA entry points."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike

from ._anndata import (
    _empty,
    _get_count_matrix,
    _resolve_mask_var,
    _write_pca_result,
)
from ._counts import (
    BoolArray,
    CountMatrix,
)
from ._log_transforms import PriorProportions
from ._pca import PCAResult
from ._svd import Solver
from ._transform import (
    DirichletCLR,
    DirichletLog,
)
from ._transform import _transform as transform_counts

DirichletTransform = Literal["dirichlet_log", "dirichlet_clr"]


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
    method = (
        DirichletLog(
            concentration=concentration,
            prior_proportions=prior_proportions,
        )
        if transform == "dirichlet_log"
        else DirichletCLR(
            concentration=concentration,
            prior_proportions=prior_proportions,
        )
    )
    transformed = transform_counts(
        X,
        method,
        check_values=check_values,
        dtype=dtype,
        _columns=mask,
    )
    return transformed.pca(
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
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
    X = _get_count_matrix(adata, layer=layer, use_raw=use_raw)
    resolved_mask = _resolve_mask_var(
        adata.var, mask_var, use_highly_variable
    )
    resolved_prior = _resolve_prior_proportions(prior_proportions, adata)
    result = _compute_dirichlet_pca(
        X,
        n_comps,
        transform=transform,
        mask=resolved_mask.values,
        concentration=concentration,
        prior_proportions=resolved_prior,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
    result.params["prior_proportions"] = _serialize_prior_proportions(prior_proportions)
    _write_pca_result(
        adata,
        result,
        mask=resolved_mask,
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
    )
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
    """Compute implicit Dirichlet-log PCA and write AnnData outputs."""
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
    """Compute implicit Dirichlet-CLR PCA and write AnnData outputs."""
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
