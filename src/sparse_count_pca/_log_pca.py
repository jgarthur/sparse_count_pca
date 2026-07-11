from __future__ import annotations

from typing import Any, Literal

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
    _canonicalize_counts,
    _validate_boolean_mask,
)
from ._log_transforms import (
    build_proportion_shifted_clr_representation,
    build_shifted_clr_representation,
    build_shifted_log_representation,
    validate_positive_scalar,
)
from ._pca import PCAResult, compute_pca_from_representation
from ._svd import Solver

LogTransform = Literal[
    "shifted_log",
    "shifted_clr",
    "proportion_shifted_clr",
]


def _compute_log_pca(
    X: CountMatrix,
    n_comps: int,
    *,
    transform: LogTransform,
    mask: BoolArray | None,
    shift: float,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> PCAResult:
    counts = _canonicalize_counts(X, check_values=check_values)
    n_vars = counts.shape[1]
    if transform == "shifted_log":
        count_shift = validate_positive_scalar(shift, name="count_shift")
        representation = build_shifted_log_representation(
            counts, count_shift=count_shift
        )
        params = {
            "transform": transform,
            "shift_domain": "count",
            "count_shift": count_shift,
            "normalization_n_vars": n_vars,
        }
        label = "Shifted log"
    elif transform == "shifted_clr":
        count_shift = validate_positive_scalar(shift, name="count_shift")
        representation = build_shifted_clr_representation(
            counts, count_shift=count_shift
        )
        params = {
            "transform": transform,
            "shift_domain": "count",
            "count_shift": count_shift,
            "normalization_n_vars": n_vars,
        }
        label = "Shifted CLR"
    else:
        composition_shift = validate_positive_scalar(shift, name="composition_shift")
        representation = build_proportion_shifted_clr_representation(
            counts, composition_shift=composition_shift
        )
        params = {
            "transform": transform,
            "shift_domain": "composition",
            "composition_shift": composition_shift,
            "effective_count_shift": "cell_total * composition_shift",
            "normalization_n_vars": n_vars,
        }
        label = "Proportion-shifted CLR"

    if mask is not None:
        mask = _validate_boolean_mask(mask, n_vars, name="mask")
        if not mask.any():
            raise ValueError("mask selected zero genes")
        representation = representation.select_columns(mask)

    return compute_pca_from_representation(
        representation,
        n_comps,
        transform_label=label,
        params=params,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def shifted_log_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    count_shift: float,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of sparse ``log1p(X / count_shift)`` values."""
    return _compute_log_pca(
        X,
        n_comps,
        transform="shifted_log",
        mask=None,
        shift=count_shift,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def shifted_clr_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    count_shift: float,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of CLR coordinates after a fixed raw-count shift."""
    return _compute_log_pca(
        X,
        n_comps,
        transform="shifted_clr",
        mask=None,
        shift=count_shift,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def proportion_shifted_clr_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    composition_shift: float,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of CLR coordinates after a fixed composition shift."""
    return _compute_log_pca(
        X,
        n_comps,
        transform="proportion_shifted_clr",
        mask=None,
        shift=composition_shift,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def _log_pca_anndata(
    adata: AnnData,
    n_comps: int,
    *,
    transform: LogTransform,
    shift: float,
    layer: str | None,
    use_raw: bool,
    mask_var: Any,
    use_highly_variable: bool | None,
    key_added: str | None,
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
    mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    result = _compute_log_pca(
        X,
        n_comps,
        transform=transform,
        mask=mask,
        shift=shift,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
    _write_pca_result(
        adata,
        result,
        mask=mask,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
    )
    return adata if copy else None


def shifted_log_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    count_shift: float,
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
    """Compute fixed-count shifted-log PCA and write AnnData outputs."""
    return _log_pca_anndata(
        adata,
        n_comps,
        transform="shifted_log",
        shift=count_shift,
        layer=layer,
        use_raw=use_raw,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        copy=copy,
    )


def shifted_clr_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    count_shift: float,
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
    """Compute fixed-count shifted-CLR PCA and write AnnData outputs."""
    return _log_pca_anndata(
        adata,
        n_comps,
        transform="shifted_clr",
        shift=count_shift,
        layer=layer,
        use_raw=use_raw,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        copy=copy,
    )


def proportion_shifted_clr_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    composition_shift: float,
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
    """Compute fixed-composition shifted-CLR PCA and write AnnData outputs."""
    return _log_pca_anndata(
        adata,
        n_comps,
        transform="proportion_shifted_clr",
        shift=composition_shift,
        layer=layer,
        use_raw=use_raw,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        key_added=key_added,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        copy=copy,
    )
