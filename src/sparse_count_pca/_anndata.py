from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray

from ._counts import _validate_boolean_mask
from ._pca import PCAResult

_empty = object()


def _get_count_matrix(
    adata: AnnData,
    *,
    layer: str | None,
    use_raw: bool,
) -> Any:
    """Select an AnnData count matrix with current-variable alignment."""
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
        return adata.raw[:, adata.var_names].X
    return adata.layers[layer] if layer is not None else adata.X


def _write_pca_result(
    adata: AnnData,
    result: PCAResult,
    *,
    mask: NDArray[np.bool_],
    mask_var: Any,
    use_highly_variable: bool | None,
    key_added: str | None,
    layer: str | None,
    use_raw: bool,
) -> None:
    """Write a PCA result using the package's Scanpy-compatible layout."""
    if key_added is None:
        obsm_key, varm_key, uns_key = "X_pca", "PCs", "pca"
    else:
        obsm_key = varm_key = uns_key = key_added
    loadings = np.full(
        (adata.n_vars, result.components.shape[0]),
        np.nan,
        dtype=result.loadings.dtype,
    )
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
    uses_highly_variable = isinstance(selector, str) and selector == "highly_variable"
    return uses_highly_variable, standard_mask
