"""Shifted-log and shifted-CLR PCA entry points."""

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
from ._counts import BoolArray, CountMatrix
from ._pca import PCAResult
from ._svd import Solver
from ._transform import (
    ProportionShiftedCLR,
    ShiftedCLR,
    ShiftedLog,
)
from ._transform import _transform as transform_counts

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
    if transform == "shifted_log":
        method = ShiftedLog(count_shift=shift)
    elif transform == "shifted_clr":
        method = ShiftedCLR(count_shift=shift)
    elif transform == "proportion_shifted_clr":
        method = ProportionShiftedCLR(composition_shift=shift)
    else:
        raise ValueError(f"Unsupported log transform: {transform!r}")
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
    """Compute PCA of ``log1p(X / count_shift)`` without densifying.

    This count-scale transform does not perform library-size normalization.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations
            in rows and variables in columns.
        n_comps: Number of principal components.
        count_shift: Positive raw-count shift.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
        return_operator: Whether to retain the centered operator in the result.

    Returns:
        PCA scores, components, variance statistics, metadata, and optionally
        the centered operator.

    Raises:
        ValueError: If counts, dimensions, shift, or dtype are invalid.

    Examples:
        >>> result = shifted_log_pca_matrix(
        ...     counts, n_comps=20, count_shift=1.0
        ... )
    """
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
    """Compute PCA of count-scale shifted CLR coordinates.

    The transform is ``clr(X + count_shift)``. The PFlog formulation in
    Booeshaghi et al. preprint v4 is obtained with
    ``count_shift = 1 / (4 * alpha)``.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations
            in rows and variables in columns.
        n_comps: Number of principal components.
        count_shift: Positive raw-count shift.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
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
    """Compute PCA of composition-scale shifted CLR coordinates.

    The transform is ``clr(X / row_total + composition_shift)``. Its effective
    raw-count shift varies with row depth and it is distinct from current
    count-scale PFlog.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations
            in rows and variables in columns.
        n_comps: Number of principal components.
        composition_shift: Positive shift on the row-composition scale.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
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
    resolved_mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    result = _compute_log_pca(
        X,
        n_comps,
        transform=transform,
        mask=resolved_mask.values,
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
        mask=resolved_mask,
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
    """Compute count-scale shifted-log PCA and write AnnData outputs.

    The transform is ``log1p(X / count_shift)`` and does not perform
    library-size normalization. Normalization uses the full variable universe
    before ``mask_var`` selects and centers PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components.
        count_shift: Positive raw-count shift.
        layer: Count layer to use. By default, use ``adata.X``.
        use_raw: Whether to use ``adata.raw.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
        copy: If ``True``, return a modified copy; otherwise mutate ``adata``.

    Returns:
        A modified AnnData object when ``copy=True``; otherwise ``None``.

    Raises:
        ValueError: If counts, dimensions, mask, shift, or dtype are invalid.
        KeyError: If a requested layer or mask key is absent.

    Examples:
        >>> shifted_log_pca(
        ...     adata, layer="counts", n_comps=20, count_shift=1.0
        ... )
    """
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
    """Compute count-scale shifted-CLR PCA and write AnnData outputs.

    CLR row means use the full variable universe before ``mask_var`` selects
    and centers PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components.
        count_shift: Positive raw-count shift. Use ``1 / (4 * alpha)`` for the
            PFlog formulation in Booeshaghi et al. preprint v4.
        layer: Count layer to use. By default, use ``adata.X``.
        use_raw: Whether to use ``adata.raw.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
        copy: If ``True``, return a modified copy; otherwise mutate ``adata``.

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
    """Compute composition-scale shifted-CLR PCA and write AnnData outputs.

    The transform is ``clr(X / row_total + composition_shift)``. CLR row means
    use the full variable universe before ``mask_var`` selects PCA columns.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of principal components.
        composition_shift: Positive shift on the row-composition scale.
        layer: Count layer to use. By default, use ``adata.X``.
        use_raw: Whether to use ``adata.raw.X``.
        mask_var: Boolean array or ``adata.var`` key selecting PCA variables.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``.
        check_values: Whether floating-point counts must be integer-like.
        dtype: Representation and ARPACK calculation dtype.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed used to construct ARPACK's starting vector.
        tol: Convergence tolerance passed to SciPy.
        copy: If ``True``, return a modified copy; otherwise mutate ``adata``.

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
