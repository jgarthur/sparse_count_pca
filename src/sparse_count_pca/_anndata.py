"""AnnData input selection, variable masking, and result-writing helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray

from ._counts import _validate_boolean_mask
from ._pca import PCAResult

_empty = object()


@dataclass(frozen=True)
class _ResolvedMask:
    """Resolved variable selection and its AnnData-safe metadata."""

    values: NDArray[np.bool_]
    mask_var: str | NDArray[np.bool_] | None
    use_highly_variable: bool
    details: dict[str, str | int | None]


def _get_count_matrix(
    adata: AnnData,
    *,
    layer: str | None,
) -> Any:
    """Select an AnnData count matrix from ``.X`` or a layer."""
    return adata.layers[layer] if layer is not None else adata.X


def _write_pca_result(
    adata: AnnData,
    result: PCAResult,
    *,
    mask: _ResolvedMask,
    key_added: str | None,
    layer: str | None,
) -> None:
    """Write a PCA result using the package's Scanpy-compatible layout."""
    if key_added is None:
        obsm_key, varm_key, uns_key = "X_pca", "PCs", "pca"
    else:
        obsm_key = varm_key = uns_key = key_added
    loadings = np.full(
        (adata.n_vars, result.components.shape[0]),
        np.nan,
        dtype=result.components.dtype,
    )
    params = dict(result.params)
    # Decomposed columns can be narrower than the requested mask when a family
    # drops empty variables. ``NaN`` marks both reasons a row was not
    # decomposed; ``n_empty_vars_excluded`` distinguishes them.
    used = result._used_columns
    decomposed = mask.values if used is None else used
    # Scanpy calls these loadings, but they are component coefficients
    # (components.T), not variance-weighted statistical loadings.
    loadings[decomposed] = result.components.T
    adata.obsm[obsm_key] = result.scores
    adata.varm[varm_key] = loadings
    params.update(
        {
            "layer": layer,
            "mask_var": mask.mask_var,
            "use_highly_variable": mask.use_highly_variable,
            "mask_var_details": mask.details,
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
) -> _ResolvedMask:
    """Resolve Scanpy-compatible variable-selection arguments.

    Args:
        var: AnnData variable metadata.
        mask_var: Variable metadata key, boolean vector, ``None`` for all
            genes, or the private default sentinel.
        use_highly_variable: Deprecated alias controlling use of
            ``var["highly_variable"]``.

    Returns:
        Boolean values together with their standard and detailed metadata.

    Raises:
        ValueError: If arguments conflict or the resolved mask is invalid.
        KeyError: If a requested variable metadata key is absent.
    """
    was_default = mask_var is _empty
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
        selector = "highly_variable" if "highly_variable" in var else None
    else:
        selector = mask_var
    if selector is None:
        mask = np.ones(var.shape[0], dtype=bool)
    elif isinstance(selector, str):
        if selector not in var:
            raise KeyError(f"{selector!r} not found in adata.var")
        mask = _validate_boolean_mask(var[selector], var.shape[0], name="mask_var")
    else:
        mask = _validate_boolean_mask(selector, var.shape[0], name="mask_var")

    n_vars_used = int(mask.sum())
    if n_vars_used == 0:
        raise ValueError("mask_var selected zero genes")

    if was_default and use_highly_variable is None:
        kind = "default"
    elif selector is None:
        kind = "none"
    elif isinstance(selector, str):
        kind = "var_key"
    else:
        kind = "array"
    key = selector if isinstance(selector, str) else None
    if selector is None or isinstance(selector, str):
        standard_mask = selector
    else:
        standard_mask = mask.copy()
    uses_highly_variable = isinstance(selector, str) and selector == "highly_variable"
    return _ResolvedMask(
        values=mask,
        mask_var=standard_mask,
        use_highly_variable=uses_highly_variable,
        details={"kind": kind, "key": key, "n_vars_used": n_vars_used},
    )
