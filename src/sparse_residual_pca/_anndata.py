from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike, NDArray

from ._clip import ClipMode
from ._matrix import _compute_residual_pca, _validate_boolean_mask
from ._residuals import AlphaLike, Model, ResidualType
from ._svd import Solver
from ._version import __version__

_empty = object()


def _resolve_mask_var(
    var: Any,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
) -> NDArray[np.bool_]:
    """Resolve Scanpy-compatible variable-selection arguments.

    Args:
        var: AnnData variable metadata.
        mask_var: Variable metadata key, boolean vector, ``None`` for all
            genes, or the private default sentinel.
        use_highly_variable: Deprecated alias controlling use of
            ``var["highly_variable"]``.

    Returns:
        A one-dimensional boolean array aligned with ``var``.

    Raises:
        ValueError: If arguments conflict or the resolved mask is invalid.
        KeyError: If a requested variable metadata key is absent.
    """
    if use_highly_variable is not None:
        warnings.warn(
            "use_highly_variable is deprecated; use mask_var instead. "
            "use_highly_variable=True is equivalent to "
            "mask_var='highly_variable'; use_highly_variable=False is "
            "equivalent to mask_var=None.",
            FutureWarning,
            stacklevel=3,
        )
        if mask_var is not _empty:
            raise ValueError("Cannot specify both mask_var and use_highly_variable")
        mask_var = "highly_variable" if use_highly_variable else None

    if mask_var is _empty:
        if "highly_variable" in var:
            mask = _validate_boolean_mask(
                var["highly_variable"], var.shape[0], name="mask_var"
            )
        else:
            mask = np.ones(var.shape[0], dtype=bool)
    elif mask_var is None:
        mask = np.ones(var.shape[0], dtype=bool)
    elif isinstance(mask_var, str):
        if mask_var not in var:
            raise KeyError(f"{mask_var!r} not found in adata.var")
        mask = _validate_boolean_mask(var[mask_var], var.shape[0], name="mask_var")
    else:
        mask = _validate_boolean_mask(mask_var, var.shape[0], name="mask_var")

    if mask.sum() == 0:
        raise ValueError("mask_var selected zero genes")
    return mask


def _serialize_mask_var(
    mask_var: Any,
    mask: NDArray[np.bool_],
    var: Any,
) -> dict[str, str | int | None]:
    """Describe a resolved variable mask using AnnData-safe values."""
    if mask_var is _empty:
        key = "highly_variable" if "highly_variable" in var else None
        kind = "default"
    elif mask_var is None:
        key = None
        kind = "none"
    elif isinstance(mask_var, str):
        key = mask_var
        kind = "var_key"
    else:
        key = None
        kind = "array"
    return {"kind": kind, "key": key, "n_vars_used": int(mask.sum())}


def _scanpy_mask_params(
    mask_var: Any,
    use_highly_variable: bool | None,
    mask: NDArray[np.bool_],
    var: Any,
) -> tuple[bool, str | NDArray[np.bool_] | None]:
    """Return Scanpy-compatible mask parameter values."""
    if use_highly_variable is not None:
        selector: Any = "highly_variable" if use_highly_variable else None
    elif mask_var is _empty:
        selector = "highly_variable" if "highly_variable" in var else None
    else:
        selector = mask_var
    if selector is None or isinstance(selector, str):
        standard_mask = selector
    else:
        standard_mask = mask.copy()
    uses_highly_variable = (
        isinstance(selector, str) and selector == "highly_variable"
    )
    return uses_highly_variable, standard_mask


def _serialize_alpha(alpha: AlphaLike | str) -> Any:
    """Convert overdispersion input to an AnnData-safe value."""
    if alpha is None or isinstance(alpha, (int, float, str)):
        return alpha
    values = np.asarray(alpha)
    return values.item() if values.ndim == 0 else values


def _resolve_alpha(
    alpha: AlphaLike | str,
    adata: AnnData,
    mask: NDArray[np.bool_],
    model: Model,
) -> AlphaLike:
    """Resolve AnnData overdispersion input before variable masking.

    Args:
        alpha: Scalar, variable metadata key, full-length array, or ``None``.
        adata: AnnData object supplying variable metadata.
        mask: Resolved variable mask. The array is intentionally not applied
            here so alignment remains with ``adata.var``.
        model: Requested null model.

    Returns:
        A scalar, a full-length overdispersion array, or ``None``.

    Raises:
        ValueError: If overdispersion is missing, unexpected, or misaligned.
        KeyError: If an ``alpha`` metadata key is absent.
    """
    if model == "scaled_nb":
        if alpha is None:
            raise ValueError("alpha is required for model='scaled_nb'")
        if isinstance(alpha, str):
            if alpha not in adata.var:
                raise KeyError(f"{alpha!r} not found in adata.var")
            values = np.asarray(adata.var[alpha], dtype=np.float64)
            return values
        values = np.asarray(alpha)
        if values.ndim == 0:
            return values.item()
        if values.ndim != 1 or values.shape[0] != adata.n_vars:
            raise ValueError(
                "AnnData alpha arrays must have shape (adata.n_vars,) before masking"
            )
        return values
    if alpha is not None:
        raise ValueError("alpha is only used for model='scaled_nb'")
    return None


def residual_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    model: Model = "poisson",
    residual: ResidualType = "pearson",
    alpha: AlphaLike | str = None,
    clip: float | None = None,
    clip_mode: ClipMode = "symmetric",
    clip_max_nnz_ratio: float | None = 2.0,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute residual PCA and write Scanpy-compatible AnnData outputs.

    Cell totals and gene proportions are estimated from the selected count
    matrix before applying ``mask_var``. PCA is then computed on an implicit,
    column-centered residual matrix.

    Args:
        adata: AnnData object containing count data.
        n_comps: Number of principal components.
        layer: Count layer to use instead of ``adata.X``.
        use_raw: Whether to use ``adata.raw`` aligned to current variable names.
        mask_var: Variable metadata key, boolean vector, ``None`` for all
            genes, or omitted for Scanpy's highly-variable default.
        use_highly_variable: Deprecated alias for ``mask_var``.
        key_added: Optional output key suffix.
        model: Null model: ``"poisson"``, ``"binomial"``, or ``"scaled_nb"``.
        residual: Residual type: ``"pearson"`` or ``"deviance"``.
        alpha: Scaled-NB overdispersion as a scalar, variable metadata key, or
            full-length array.
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.
        check_values: Whether to require floating-point entries to be
            integer-like.
        dtype: Storage and operator floating-point dtype.
        solver: SVD solver. Version 1 supports only ``"arpack"``.
        random_state: Seed for the ARPACK starting vector.
        tol: ARPACK convergence tolerance.
        copy: Whether to modify and return a copy. Backed inputs are loaded
            into memory.

    Returns:
        The modified copy when ``copy=True``; otherwise ``None`` after
        modifying ``adata`` in place.

    Raises:
        ValueError: If inputs, model parameters, or dimensions are invalid.
    """
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
    alpha_values = _resolve_alpha(alpha, adata, mask, model)
    result = _compute_residual_pca(
        X,
        n_comps,
        mask=mask,
        model=model,
        residual=residual,
        alpha=alpha_values,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
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

    loadings_full = np.full(
        (adata.n_vars, n_comps), np.nan, dtype=result.loadings.dtype
    )
    loadings_full[mask] = result.loadings
    adata.obsm[obsm_key] = result.scores
    adata.varm[varm_key] = loadings_full
    standard_hv, standard_mask = _scanpy_mask_params(
        mask_var, use_highly_variable, mask, adata.var
    )
    adata.uns[uns_key] = {
        "variance": result.explained_variance,
        "variance_ratio": result.explained_variance_ratio,
        "singular_values": result.singular_values,
        "params": {
            "model": model,
            "residual": residual,
            "alpha": _serialize_alpha(alpha),
            "clip": clip,
            "clip_mode": clip_mode,
            "clip_max_nnz_ratio": clip_max_nnz_ratio,
            "zero_center": True,
            "layer": layer,
            "use_raw": use_raw,
            "mask_var": standard_mask,
            "use_highly_variable": standard_hv,
            "mask_var_details": _serialize_mask_var(mask_var, mask, adata.var),
            "solver": solver,
            "n_comps": n_comps,
            "random_state": random_state,
            "tol": tol,
            "check_values": check_values,
            "dtype": str(np.dtype(dtype)),
            "package_version": __version__,
        },
    }
    return adata if copy else None
