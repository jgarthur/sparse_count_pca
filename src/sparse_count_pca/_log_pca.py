"""Log-normalized and shifted-CLR PCA entry points."""

from __future__ import annotations

from typing import Any

from anndata import AnnData
from numpy.typing import ArrayLike, DTypeLike

from ._anndata import (
    _empty,
    _get_count_matrix,
    _resolve_mask_var,
    _write_pca_result,
)
from ._counts import BoolArray, CountMatrix
from ._pca import PCAResult
from ._svd import Solver
from ._transform import (
    Log1pNormalized,
    ProportionShiftedCLR,
    ShiftedCLR,
    Transform,
)
from ._transform import _transform as transform_counts


def _compute_log_pca(
    X: CountMatrix,
    n_comps: int,
    *,
    method: Transform,
    mask: BoolArray | None,
    obs: Any | None,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> PCAResult:
    transformed = transform_counts(
        X,
        method,
        check_values=check_values,
        dtype=dtype,
        _columns=mask,
        _isolate_returned_operator=False,
        _obs=obs,
    )
    return transformed.pca(
        n_comps,
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
    """Compute PCA of count-scale shifted CLR coordinates.

    The transform is ``clr(X + count_shift)``, that is
    ``log(x_ij + count_shift) - mean_k log(x_ik + count_shift)`` over all
    ``n_vars`` variables. The PFlog formulation in Booeshaghi et al. preprint v4
    is obtained with ``count_shift = 1 / (4 * alpha)``, for a dataset-wide
    ``alpha`` under a negative-binomial size-factor model rather than the
    per-gene ``scaled_nb`` ``alpha`` of ``residual_pca``.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        count_shift: Positive constant added to every raw count before taking logs. It
            is the same for every observation regardless of total count. The PFlog
            formulation in Booeshaghi et al. preprint v4 uses ``count_shift = 1 / (4 *
            alpha)``, where ``alpha`` is the overdispersion of the negative-binomial
            size-factor model.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        ValueError: If counts, dimensions, shift, or dtype are invalid.

    Examples:
        >>> result = shifted_clr_pca_matrix(
        ...     counts, n_comps=20, count_shift=1.0
        ... )
    """
    return _compute_log_pca(
        X,
        n_comps,
        method=ShiftedCLR(count_shift=count_shift),
        mask=None,
        obs=None,
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
    """Compute PCA of composition-scale shifted CLR coordinates.

    The transform is ``clr(X / row_total + composition_shift)``. Its effective
    raw-count shift varies with row depth and it is distinct from current
    count-scale PFlog.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        composition_shift: Positive constant added to every proportion after dividing
            each observation by its count total, before taking logs. Because ``clr(x_i /
            n_i + composition_shift) = clr(x_i + n_i * composition_shift)``, the
            equivalent raw-count shift is ``n_i * composition_shift`` and therefore
            differs per observation.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        ValueError: If counts, dimensions, shift, or dtype are invalid.

    Examples:
        >>> result = proportion_shifted_clr_pca_matrix(
        ...     counts, n_comps=20, composition_shift=1.0
        ... )
    """
    return _compute_log_pca(
        X,
        n_comps,
        method=ProportionShiftedCLR(composition_shift=composition_shift),
        mask=None,
        obs=None,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def log1p_norm_pca_matrix(
    X: CountMatrix,
    n_comps: int = 50,
    *,
    target_sum: float | None = None,
    size_factors: ArrayLike | None = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> PCAResult:
    """Compute PCA after size-factor normalization and ``log1p``.

    The transform is ``log1p(x_ij / s_i)``. With ``target_sum=t``, divisors are
    ``s_i = n_i / t``. By default, ``t`` is the median observation total, so
    size factors have median one. Alternatively, pass positive size factors
    directly; they are used without rescaling.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations in rows
            and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        target_sum: Positive total to which each observation is normalized. ``None``
            uses the median observation total. Mutually exclusive with
            ``size_factors``.
        size_factors: Positive length-``n_obs`` divisors used as-is. Their scale sets
            the effective raw-count pseudocount. Mutually exclusive with an explicit
            ``target_sum``.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        TypeError: If a named size-factor key is used with matrix input.
        ValueError: If counts, dimensions, normalization parameters, or dtype
            are invalid.

    Examples:
        >>> result = log1p_norm_pca_matrix(counts, n_comps=20, target_sum=1e4)
    """
    return _compute_log_pca(
        X,
        n_comps,
        method=Log1pNormalized(
            target_sum=target_sum,
            size_factors=size_factors,
        ),
        mask=None,
        obs=None,
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
    method: Transform,
    layer: str | None,
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
    X = _get_count_matrix(adata, layer=layer)
    resolved_mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    result = _compute_log_pca(
        X,
        n_comps,
        method=method,
        mask=resolved_mask.values,
        obs=adata.obs,
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
        mask=resolved_mask,
        key_added=key_added,
        layer=layer,
    )
    return adata if copy else None


def log1p_norm_pca(
    adata: AnnData,
    n_comps: int = 50,
    *,
    target_sum: float | None = None,
    size_factors: ArrayLike | str | None = None,
    layer: str | None = None,
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
    """Compute size-factor-normalized log1p PCA and write AnnData outputs.

    The transform is ``log1p(x_ij / s_i)``. With ``target_sum=t``, divisors are
    ``s_i = n_i / t``. By default, ``t`` is the median observation total, so a
    typical observation has a size factor and effective pseudocount near one.
    Supplied size factors are used without rescaling.

    Args:
        adata: AnnData object with observations in rows and variables in columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        target_sum: Positive total to which each observation is normalized. ``None``
            uses the median observation total. Mutually exclusive with
            ``size_factors``.
        size_factors: Positive length-``n_obs`` divisors or an ``adata.obs`` key. Values
            are used as-is, and their scale sets the effective raw-count pseudocount.
            Mutually exclusive with an explicit ``target_sum``.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables. When
            omitted, use ``"highly_variable"`` if present; explicit ``None`` selects
            every variable. Cell totals are computed before this PCA-only mask.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults to the
            conventional ``"X_pca"``, ``"PCs"``, and ``"pca"`` keys.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        ValueError: If counts, dimensions, mask, normalization parameters, or dtype
            are invalid.
        KeyError: If a requested layer, mask key, or size-factor key is absent.

    Examples:
        >>> log1p_norm_pca(
        ...     adata, layer="counts", n_comps=20, target_sum=1e4
        ... )
    """
    return _log_pca_anndata(
        adata,
        n_comps,
        method=Log1pNormalized(
            target_sum=target_sum,
            size_factors=size_factors,
        ),
        layer=layer,
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
    """Compute count-scale shifted-CLR PCA and write AnnData outputs.

    The transform is ``clr(X + count_shift)``. CLR row means use the full
    variable universe before ``mask_var`` selects and centers PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        count_shift: Positive constant added to every raw count before taking logs. It
            is the same for every observation regardless of total count. The PFlog
            formulation in Booeshaghi et al. preprint v4 uses ``count_shift = 1 / (4 *
            alpha)``, where ``alpha`` is the overdispersion of the negative-binomial
            size-factor model.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional ``"X_pca"``, ``"PCs"``, and ``"pca"`` keys.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        ValueError: If counts, dimensions, mask, shift, or dtype are invalid.
        KeyError: If a requested layer or mask key is absent.

    Examples:
        >>> shifted_clr_pca(
        ...     adata, layer="counts", n_comps=20, count_shift=1.0
        ... )
    """
    return _log_pca_anndata(
        adata,
        n_comps,
        method=ShiftedCLR(count_shift=count_shift),
        layer=layer,
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
    """Compute composition-scale shifted-CLR PCA and write AnnData outputs.

    The transform is ``clr(X / row_total + composition_shift)``. CLR row means
    use the full variable universe before ``mask_var`` selects PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_vars)``.
        composition_shift: Positive constant added to every proportion after dividing
            each observation by its count total, before taking logs. Because ``clr(x_i /
            n_i + composition_shift) = clr(x_i + n_i * composition_shift)``, the
            equivalent raw-count shift is ``n_i * composition_shift`` and therefore
            differs per observation.
        layer: AnnData count layer to use. If ``layer=None``, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional ``"X_pca"``, ``"PCs"``, and ``"pca"`` keys.
        check_values: When ``True``, reject floating-point input whose values are not
            within ``1e-8`` of integers.
        dtype: Representation and ARPACK calculation dtype, either ``"float64"`` or
            ``"float32"``. Normalization factors are always fitted in float64; the
            representation built from them is cast to ``dtype`` afterwards.
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
        ValueError: If counts, dimensions, mask, shift, or dtype are invalid.
        KeyError: If a requested layer or mask key is absent.

    Examples:
        >>> proportion_shifted_clr_pca(
        ...     adata, layer="counts", n_comps=20, composition_shift=1.0
        ... )
    """
    return _log_pca_anndata(
        adata,
        n_comps,
        method=ProportionShiftedCLR(composition_shift=composition_shift),
        layer=layer,
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
