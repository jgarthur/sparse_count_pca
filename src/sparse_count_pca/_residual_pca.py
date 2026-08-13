"""Residual PCA entry points for count matrices and AnnData objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike

from ._anndata import (
    _empty,
    _get_count_matrix,
    _resolve_mask_var,
    _write_pca_result,
)
from ._clip import ClipMode
from ._counts import BoolArray, CountMatrix
from ._pca import PCAResult
from ._residuals import AlphaLike, Model, ResidualType
from ._svd import Solver
from ._transform import Residual
from ._transform import _transform as transform_counts


def _compute_residual_pca(
    X: CountMatrix,
    n_comps: int,
    *,
    mask: BoolArray | None,
    model: Model,
    residual: ResidualType,
    alpha: AlphaLike,
    clip: float | None,
    clip_mode: ClipMode,
    clip_max_nnz_ratio: float | None,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> PCAResult:
    """Run the shared validated residual PCA implementation.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix.
        n_comps: Number of principal components.
        mask: Boolean variable mask, or ``None`` to use every column.
        model: Residual null model.
        residual: Residual type.
        alpha: Scaled-NB overdispersion, or ``None``.
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Storage and operator floating-point dtype.
        solver: SVD solver name.
        random_state: Seed for the ARPACK starting vector.
        tol: ARPACK convergence tolerance.
        return_operator: Whether to retain the centered residual operator.

    Returns:
        A populated ``PCAResult``.

    Raises:
        ValueError: If counts, dimensions, or model parameters are invalid.
    """
    transformed = transform_counts(
        X,
        Residual(
            model=model,
            residual=residual,
            alpha=alpha,
            clip=clip,
            clip_mode=clip_mode,
            clip_max_nnz_ratio=clip_max_nnz_ratio,
        ),
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


def residual_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    model: Model = "poisson",
    residual: ResidualType = "pearson",
    alpha: AlphaLike = None,
    clip: float | None = None,
    clip_mode: ClipMode = "symmetric",
    clip_max_nnz_ratio: float | None = 2.0,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA of implicitly represented residuals from a count matrix.

    The entire matrix defines cell totals and gene proportions. Residuals are
    represented without materializing the dense residual matrix, centered by
    column, and decomposed with ARPACK.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        model: Null model supplying the variance ``V_ij``. ``"poisson"`` uses ``mu_ij``,
            ``"binomial"`` uses ``mu_ij * (1 - p_j)``, and ``"scaled_nb"`` uses ``mu_ij
            * (1 + alpha_j * mean_n * p_j)``, where ``mean_n`` is the mean observation
            count total.
        residual: ``"pearson"`` for standardized deviations from ``mu_ij``, or
            ``"deviance"`` for signed square-root deviance contributions.
        alpha: Overdispersion of the ``scaled_nb`` model, as a nonnegative scalar, a
            length-``n_vars`` array, or an AnnData variable key (AnnData entry points
            only). Larger values mean more variance, and values below ``1e-8`` use the
            Poisson limit. Required for ``model="scaled_nb"`` and rejected for the other
            models. Values for variables with no counts are replaced with zero.
        clip: Positive threshold applied to the uncentered residual values
            before PCA centering, or ``None`` for no clipping.
        clip_mode: ``"symmetric"`` clips residuals into ``[-clip, clip]``; ``"upper"``
            clips only from above, into ``(-inf, clip]``, which leaves negative
            residuals (including all zero counts) untouched.
        clip_max_nnz_ratio: Upper bound on how far symmetric clipping may grow the
            stored sparse support, as a multiple of the input count matrix's number of
            stored nonzeros. Reaching it raises ``RuntimeError``; ``None`` removes the
            limit. Only symmetric clipping can grow support, so the limit never binds
            when ``clip`` is ``None`` or ``clip_mode="upper"``.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors and bounded support intermediates
            are evaluated in float64; sparse corrections, low-rank factors, and
            arrays passed to ARPACK use ``dtype``.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK. ``None``
            breaks bit-for-bit reproducibility.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests machine
            precision.
        return_operator: Whether to include the centered residual operator in
            the result.

    Returns:
        Residual PCA scores, components, variance statistics, and metadata.

    Raises:
        ValueError: If counts, dimensions, or model parameters are invalid.

    Examples:
        >>> result = residual_pca_matrix(
        ...     counts,
        ...     n_comps=20,
        ...     model="poisson",
        ...     residual="pearson",
        ... )
        >>> result.scores.shape
        (counts.shape[0], 20)

    See Also:
        residual_pca: AnnData entry point.
        transform: Two-step transformation and PCA workflow.
    """
    return _compute_residual_pca(
        X,
        n_comps,
        mask=None,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def _serialize_alpha(alpha: AlphaLike | str) -> Any:
    """Convert overdispersion input to an AnnData-safe value."""
    if alpha is None or isinstance(alpha, (int, float, str)):
        return alpha
    values = np.asarray(alpha)
    return values.item() if values.ndim == 0 else values.copy()


def _resolve_alpha(
    alpha: AlphaLike | str,
    adata: AnnData,
    model: Model,
) -> AlphaLike:
    """Resolve AnnData overdispersion input before variable masking."""
    if model == "scaled_nb":
        if alpha is None:
            raise ValueError("alpha is required for model='scaled_nb'")
        if isinstance(alpha, str):
            if alpha not in adata.var:
                raise KeyError(f"{alpha!r} not found in adata.var")
            return np.asarray(adata.var[alpha], dtype=np.float64)
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

    The chosen count matrix fits cell totals and gene proportions before
    ``mask_var`` selects PCA variables. Residual clipping, when enabled, occurs
    before centering the selected transformed columns.

    By default, scores are written to ``adata.obsm["X_pca"]``, component
    vectors to ``adata.varm["PCs"]``, and variance statistics and parameters to
    ``adata.uns["pca"]``. Masked component rows contain ``NaN``.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``, where
            ``n_nonempty_vars`` counts selected variables that are not
            identically zero.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional PCA keys.
        model: Null model supplying the variance ``V_ij``. ``"poisson"`` uses ``mu_ij``,
            ``"binomial"`` uses ``mu_ij * (1 - p_j)``, and ``"scaled_nb"`` uses ``mu_ij
            * (1 + alpha_j * mean_n * p_j)``, where ``mean_n`` is the mean observation
            count total.
        residual: ``"pearson"`` for standardized deviations from ``mu_ij``, or
            ``"deviance"`` for signed square-root deviance contributions.
        alpha: Overdispersion of the ``scaled_nb`` model, as a nonnegative scalar, a
            length-``n_vars`` array, or an AnnData variable key (AnnData entry points
            only). Larger values mean more variance, and values below ``1e-8`` use the
            Poisson limit. Required for ``model="scaled_nb"`` and rejected for the other
            models. Values for variables with no counts are replaced with zero.
        clip: Positive threshold applied to the uncentered residual values
            before PCA centering, or ``None`` for no clipping.
        clip_mode: ``"symmetric"`` clips residuals into ``[-clip, clip]``; ``"upper"``
            clips only from above, into ``(-inf, clip]``, which leaves negative
            residuals (including all zero counts) untouched.
        clip_max_nnz_ratio: Upper bound on how far symmetric clipping may grow the
            stored sparse support, as a multiple of the input count matrix's number of
            stored nonzeros. Reaching it raises ``RuntimeError``; ``None`` removes the
            limit. Only symmetric clipping can grow support, so the limit never binds
            when ``clip`` is ``None`` or ``clip_mode="upper"``.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors and bounded support intermediates
            are evaluated in float64; sparse corrections, low-rank factors, and
            arrays passed to ARPACK use ``dtype``.
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
        ValueError: If counts, dimensions, mask, clipping, or model parameters
            are invalid.
        KeyError: If a requested layer, mask, or overdispersion key is absent.
        RuntimeError: If exact symmetric clipping would meet or exceed the
            configured sparse support-growth limit.

    Examples:
        >>> residual_pca(
        ...     adata,
        ...     layer="counts",
        ...     n_comps=50,
        ...     model="poisson",
        ...     residual="pearson",
        ... )
        >>> adata.obsm["X_pca"].shape
        (adata.n_obs, 50)

    See Also:
        residual_pca_matrix: Matrix entry point.
        transform: Two-step transformation and PCA workflow.
    """
    if copy:
        adata = adata.to_memory() if adata.isbacked else adata.copy()
    X = _get_count_matrix(adata, layer=layer)
    resolved_mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    alpha_values = _resolve_alpha(alpha, adata, model)
    result = _compute_residual_pca(
        X,
        n_comps,
        mask=resolved_mask.values,
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
    result.params["alpha"] = _serialize_alpha(alpha)
    _write_pca_result(
        adata,
        result,
        mask=resolved_mask,
        key_added=key_added,
        layer=layer,
    )
    return adata if copy else None
