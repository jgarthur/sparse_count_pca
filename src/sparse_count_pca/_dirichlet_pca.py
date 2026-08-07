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
    if transform == "dirichlet_log":
        method = DirichletLog(
            concentration=concentration,
            prior_proportions=prior_proportions,
        )
    elif transform == "dirichlet_clr":
        method = DirichletCLR(
            concentration=concentration,
            prior_proportions=prior_proportions,
        )
    else:
        raise ValueError(f"Unsupported Dirichlet transform: {transform!r}")
    transformed = transform_counts(
        X,
        method,
        check_values=check_values,
        dtype=dtype,
        _columns=mask,
        _isolate_returned_operator=False,
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
    """Compute PCA of log Dirichlet posterior-mean compositions.

    For prior concentration ``A`` and proportions ``p``, the analyzed
    composition is ``(X + A * p) / (row_total + A)``.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        concentration: Positive total Dirichlet prior concentration, in units of counts.
            This is the total number of prior pseudo-counts spread across all variables.
            Larger values shrink each observation harder toward the prior composition.
        prior_proportions: Dirichlet prior proportions, as a length-``n_vars`` array, an
            AnnData variable key (AnnData entry points only), or ``None`` for a uniform
            prior. Values must be strictly positive and sum to one.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Normalization is always fitted in
            float64 and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK. ``None``
            breaks bit-for-bit reproducibility.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests machine
            precision.
        return_operator: Whether to retain the centered operator in the result.

    Returns:
        PCA scores, components, variance statistics, metadata, and optionally
        the centered operator.

    Raises:
        ValueError: If counts, dimensions, concentration, prior, or dtype are
            invalid.

    Examples:
        >>> result = dirichlet_log_pca_matrix(
        ...     counts, n_comps=20, concentration=1.0
        ... )
    """
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
    """Compute PCA of CLR Dirichlet posterior-mean compositions.

    For prior concentration ``A`` and proportions ``p``, the posterior-mean
    composition is ``(X + A * p) / (row_total + A)``; CLR then subtracts each
    row's mean log composition.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        concentration: Positive total Dirichlet prior concentration, in units of counts.
            This is the total number of prior pseudo-counts spread across all variables.
            Larger values shrink each observation harder toward the prior composition.
        prior_proportions: Dirichlet prior proportions, as a length-``n_vars`` array, an
            AnnData variable key (AnnData entry points only), or ``None`` for a uniform
            prior. Values must be strictly positive and sum to one.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Normalization is always fitted in
            float64 and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK. ``None``
            breaks bit-for-bit reproducibility.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests machine
            precision.
        return_operator: Whether to retain the centered operator in the result.

    Returns:
        PCA scores, components, variance statistics, metadata, and optionally
        the centered operator.

    Raises:
        ValueError: If counts, dimensions, concentration, prior, or dtype are
            invalid.

    Examples:
        >>> result = dirichlet_clr_pca_matrix(
        ...     counts, n_comps=20, concentration=1.0
        ... )
    """
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
    return np.asarray(prior_proportions).copy()


def _dirichlet_pca_anndata(
    adata: AnnData,
    n_comps: int,
    *,
    transform: DirichletTransform,
    layer: str | None,
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
    X = _get_count_matrix(adata, layer=layer)
    resolved_mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
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
    )
    return adata if copy else None


def dirichlet_log_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    layer: str | None = None,
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
    """Compute Dirichlet-log PCA and write AnnData outputs.

    With prior counts ``a_j = concentration * prior_proportions_j``, the
    analyzed values are ``log((X_ij + a_j) / (n_i + concentration))``, where
    ``n_i`` is cell ``i``'s count total. The prior is defined over the full
    variable universe before ``mask_var`` selects and centers PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional ``"X_pca"``, ``"PCs"``, and ``"pca"`` keys.
        concentration: Positive total Dirichlet prior concentration, in units of counts.
            This is the total number of prior pseudo-counts spread across all variables.
            Larger values shrink each observation harder toward the prior composition.
        prior_proportions: Dirichlet prior proportions, as a length-``n_vars`` array, an
            AnnData variable key (AnnData entry points only), or ``None`` for a uniform
            prior. Values must be strictly positive and sum to one.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Normalization is always fitted in
            float64 and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK. ``None``
            breaks bit-for-bit reproducibility.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests machine
            precision.
        copy: If ``True``, return a modified copy. Otherwise mutate ``adata`` and return
            ``None``.

    Returns:
        A modified AnnData object when ``copy=True``; otherwise ``None``.

    Raises:
        ValueError: If counts, dimensions, mask, prior, or dtype are invalid.
        KeyError: If a requested layer, mask, or prior key is absent.

    Examples:
        >>> dirichlet_log_pca(
        ...     adata, layer="counts", n_comps=20, concentration=1.0
        ... )
    """
    return _dirichlet_pca_anndata(
        adata,
        n_comps,
        transform="dirichlet_log",
        layer=layer,
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
    """Compute Dirichlet-CLR PCA and write AnnData outputs.

    With prior counts ``a_j = concentration * prior_proportions_j``, the
    analyzed values are ``clr(X_i + a)``, which equals the CLR of the
    posterior-mean composition ``(X_i + a) / (n_i + concentration)`` because CLR
    removes the per-cell denominator. The prior and CLR row mean use the full
    variable universe before ``mask_var`` selects and centers PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional ``"X_pca"``, ``"PCs"``, and ``"pca"`` keys.
        concentration: Positive total Dirichlet prior concentration, in units of counts.
            This is the total number of prior pseudo-counts spread across all variables.
            Larger values shrink each observation harder toward the prior composition.
        prior_proportions: Dirichlet prior proportions, as a length-``n_vars`` array, an
            AnnData variable key (AnnData entry points only), or ``None`` for a uniform
            prior. Values must be strictly positive and sum to one.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Normalization is always fitted in
            float64 and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK. ``None``
            breaks bit-for-bit reproducibility.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests machine
            precision.
        copy: If ``True``, return a modified copy. Otherwise mutate ``adata`` and return
            ``None``.

    Returns:
        A modified AnnData object when ``copy=True``; otherwise ``None``.

    Raises:
        ValueError: If counts, dimensions, mask, prior, or dtype are invalid.
        KeyError: If a requested layer, mask, or prior key is absent.

    Examples:
        >>> dirichlet_clr_pca(
        ...     adata, layer="counts", n_comps=20, concentration=1.0
        ... )
    """
    return _dirichlet_pca_anndata(
        adata,
        n_comps,
        transform="dirichlet_clr",
        layer=layer,
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
