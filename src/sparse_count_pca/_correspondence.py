"""Correspondence analysis for sparse count matrices and AnnData objects."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from anndata import AnnData
from numpy.typing import DTypeLike, NDArray
from scipy import sparse

from ._anndata import (
    _empty,
    _get_count_matrix,
    _resolve_mask_var,
)
from ._counts import (
    BoolArray,
    CountMatrix,
    _canonicalize_counts,
    _empty_columns,
    _sum_counts,
    _validate_boolean_mask,
)
from ._operator import (
    SparseLowRankLinearOperator,
    _normalize_operator_dtype,
    _squared_norm_is_numerically_zero,
)
from ._representation import SparseLowRankMatrix
from ._residual_pca import _resolve_alpha, _serialize_alpha
from ._residuals import (
    AlphaLike,
    _validate_model,
    build_pearson_residual_representation,
)
from ._svd import Solver, compute_truncated_svd
from ._version import __version__

Float64Array = NDArray[np.float64]
CorrespondenceModel: TypeAlias = Literal["poisson", "scaled_nb"]


@dataclass
class CorrespondenceAnalysisResult:
    """Principal-coordinate results from classical or experimental analysis.

    Standard coordinates are derivable by dividing principal coordinates by
    singular values and are intentionally not stored.

    Attributes:
        row_principal_coordinates: Row principal coordinates with shape
            ``(n_rows, n_comps)``, computed as each left singular vector divided
            by the square root of its row mass and then scaled by the singular
            value.
        column_principal_coordinates: Column principal coordinates with shape
            ``(n_columns_used, n_comps)``, computed the same way from the right
            singular vectors and column masses. Rows for analyzed columns with
            zero mass are ``NaN``, since dividing by zero mass leaves their
            coordinate undefined.
        singular_values: Singular values in descending order.
        principal_inertias: Inertia carried by each returned axis, equal to its
            squared singular value.
        inertia_ratio: ``principal_inertias / total_inertia``, the fraction of
            total inertia carried by each returned axis. Entries sum to one only
            if every axis is returned.
        total_inertia: Inertia of the whole table, not only the returned axes.
            For classical CA this is the Pearson chi-squared statistic divided
            by the grand total; in experimental scaled-NB mode it is the
            corresponding residual inertia, with no chi-squared interpretation.
        row_masses: Row totals of the analyzed table divided by its grand total,
            so they sum to one.
        column_masses: Column totals of the analyzed table divided by its grand
            total, so they sum to one.
        params: Model, solver, dtype, and reproducibility metadata.
        operator: Uncentered operator passed to ARPACK when
            ``return_operator=True``; otherwise ``None``.
    """

    row_principal_coordinates: Float64Array
    column_principal_coordinates: Float64Array
    singular_values: Float64Array
    principal_inertias: Float64Array
    inertia_ratio: Float64Array
    total_inertia: float
    row_masses: Float64Array
    column_masses: Float64Array
    params: dict[str, Any]
    operator: SparseLowRankLinearOperator | None = None


def build_correspondence_representation(
    X: sparse.csr_matrix,
    *,
    model: CorrespondenceModel = "poisson",
    alpha: Float64Array | None = None,
    row_totals: Float64Array | None = None,
    column_totals: Float64Array | None = None,
) -> tuple[SparseLowRankMatrix, Float64Array, Float64Array]:
    """Build a total-scaled Pearson-residual representation.

    The analyzed entries are ``(x_ij - mu_ij) / (sqrt(N) * sqrt(V_ij))``, the
    Pearson residuals of the same null model divided by the square root of the
    grand total ``N``. Row and column masses are the margins divided by ``N``.
    """
    if row_totals is None:
        row_totals = _sum_counts(X, axis=1)
    if column_totals is None:
        column_totals = _sum_counts(X, axis=0)
    with np.errstate(over="ignore"):
        total = float(np.sum(row_totals, dtype=np.float64))
    if not (
        np.isfinite(row_totals).all()
        and np.isfinite(column_totals).all()
        and np.isfinite(total)
    ):
        raise ValueError("Count margins overflow float64")
    row_masses = row_totals / total
    column_masses = column_totals / total
    pearson = build_pearson_residual_representation(
        X,
        row_totals,
        column_masses,
        model=model,
        alpha=alpha,
    )
    return pearson.scaled(1.0 / np.sqrt(total)), row_masses, column_masses


def _compute_correspondence_analysis(
    X: CountMatrix,
    n_comps: int,
    *,
    mask: BoolArray | None,
    model: CorrespondenceModel,
    alpha: AlphaLike,
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
) -> CorrespondenceAnalysisResult:
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    dtype = _normalize_operator_dtype(dtype)
    counts = _canonicalize_counts(X, check_values=check_values)
    n_vars = counts.shape[1]
    if model not in {"poisson", "scaled_nb"}:
        raise ValueError("model must be 'poisson' or 'scaled_nb'")
    selected = np.ones(n_vars, dtype=bool)
    if mask is not None:
        selected = _validate_boolean_mask(mask, n_vars, name="mask")
        if not selected.any():
            raise ValueError("mask selected zero columns")
    # A zero-mass column contributes nothing to the fitted table: its
    # standardized deviation is zero by continuity, so its overdispersion
    # cannot reach any output. Emptiness is a property of the input rather than
    # of the mask, so this matches the residual families and does not depend on
    # which columns were selected.
    alpha_full = _validate_model(
        model, "pearson", alpha, n_vars, ignore=_empty_columns(counts)
    )
    if not selected.all():
        counts = counts[:, selected].tocsr(copy=False)
        if alpha_full is not None:
            alpha_full = alpha_full[selected]
    row_totals = _sum_counts(counts, axis=1)
    column_totals = _sum_counts(counts, axis=0)
    if not (np.isfinite(row_totals).all() and np.isfinite(column_totals).all()):
        raise ValueError("Count margins overflow float64")
    if (row_totals == 0).any():
        raise ValueError(
            "Rows with zero mass are not supported; filter empty rows out of "
            "the count table first"
        )
    empty_columns = column_totals == 0
    n_empty_vars = int(empty_columns.sum())
    n_rows, n_columns = counts.shape
    # Named apart from ``n_columns_used``, which is the analyzed table width and
    # still counts retained zero-mass columns.
    n_nonempty_columns = n_columns - n_empty_vars
    if not 1 <= n_comps < min(n_rows, n_nonempty_columns):
        detail = (
            ""
            if not n_empty_vars
            else (
                f"; n_nonempty_columns excludes {n_empty_vars} of the "
                f"{n_columns} analyzed columns that have zero mass"
            )
        )
        raise ValueError(
            "n_comps must satisfy 1 <= n_comps < "
            f"min(n_rows={n_rows}, n_nonempty_columns={n_nonempty_columns}); "
            f"got n_comps={n_comps}{detail}"
        )
    if model == "scaled_nb":
        warnings.warn(
            "model='scaled_nb' correspondence analysis is experimental; its "
            "inertia has no classical Pearson chi-square interpretation",
            UserWarning,
            stacklevel=3,
        )

    representation, row_masses, column_masses = build_correspondence_representation(
        counts,
        model=model,
        alpha=alpha_full,
        row_totals=row_totals,
        column_totals=column_totals,
    )
    operator = SparseLowRankLinearOperator(
        representation,
        center=False,
        dtype=dtype,
        # This one-step path owns the representation. If retained, the operator
        # takes over those arrays rather than copying them for isolation.
        copy=False,
    )
    total_inertia = operator.frobenius_squared_uncentered()
    if _squared_norm_is_numerically_zero(
        total_inertia,
        scale=1.0,
        shape=counts.shape,
        dtype=dtype,
    ):
        raise ValueError(
            "Contingency table has numerically zero inertia, so correspondence "
            "axes are undefined. Every row has the same profile across the "
            "selected columns, leaving no departure from independence."
        )
    decomposition = compute_truncated_svd(
        operator,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    left = decomposition.left_vectors.astype(np.float64, copy=False)
    right = decomposition.right_vectors.T.astype(np.float64, copy=False)
    singular_values = decomposition.singular_values
    row_principal = left * singular_values[None, :] / np.sqrt(row_masses)[:, None]
    # A zero-mass column has a zero singular-vector entry, but the principal
    # coordinate divides by the square root of its mass, so it alone is
    # genuinely undefined and is reported as NaN rather than dropped.
    with np.errstate(divide="ignore", invalid="ignore"):
        column_principal = (
            right * singular_values[None, :] / np.sqrt(column_masses)[:, None]
        )
    if n_empty_vars:
        column_principal[empty_columns] = np.nan
    inertias = singular_values**2
    params = {
        "analysis": "correspondence_analysis",
        "model": model,
        "alpha": _serialize_alpha(alpha),
        "experimental": model == "scaled_nb",
        "inertia_interpretation": (
            "pearson_chi_squared_over_grand_total"
            if model == "poisson"
            else "scaled_nb_pearson_residual_inertia"
        ),
        "zero_center": False,
        "n_comps": n_comps,
        "n_empty_vars": n_empty_vars,
        "solver": solver,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": str(dtype),
        "package_version": __version__,
    }
    return CorrespondenceAnalysisResult(
        row_principal_coordinates=row_principal,
        column_principal_coordinates=column_principal,
        singular_values=singular_values,
        principal_inertias=inertias,
        inertia_ratio=inertias / total_inertia,
        total_inertia=float(total_inertia),
        row_masses=row_masses,
        column_masses=column_masses,
        params=params,
        operator=operator if return_operator else None,
    )


def correspondence_analysis_matrix(
    X: CountMatrix,
    n_comps: int = 2,
    *,
    model: CorrespondenceModel = "poisson",
    alpha: AlphaLike = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    return_operator: bool = False,
) -> CorrespondenceAnalysisResult:
    """Compute correspondence analysis from a nonnegative count matrix.

    Classical mode decomposes the standardized independence-residual matrix
    ``(x_ij - mu_ij) / (sqrt(N) * sqrt(mu_ij))`` without ordinary PCA column
    centering, where ``mu_ij = n_i * p_j`` and ``N`` is the grand total. The
    experimental scaled-NB mode substitutes an exposure-scaled
    negative-binomial variance and emits ``UserWarning``.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with observations
            in rows and variables in columns.
        n_comps: Number of correspondence axes to return. Must satisfy
            ``1 <= n_comps < min(n_rows, n_nonempty_columns)``, where
            ``n_nonempty_columns`` counts analyzed columns with nonzero mass.
        model: ``"poisson"`` for classical CA, standardizing by the Poisson
            variance ``mu_ij``, or ``"scaled_nb"`` for the experimental residual
            ordination, standardizing by ``mu_ij * (1 + alpha_j * mean_n *
            p_j)`` with ``mean_n`` the mean row total.
        alpha: Per-variable overdispersion of the ``scaled_nb`` model, as a
            scalar broadcast to every variable or a length-``n_vars`` array. It
            is a dispersion, not a size: larger values mean more variance and
            ``alpha=0`` is the Poisson limit. An estimate reported as a size
            (``theta``, ``r``) must be inverted first. Must be nonnegative;
            values below ``1e-8`` use the Poisson limit. Values for analyzed
            columns with zero mass are replaced with zero instead of being
            validated.
            Required only when ``model="scaled_nb"``, and rejected otherwise.
        check_values: When ``True``, reject floating-point input whose values
            are not within ``1e-8`` of integers. Set it to ``False`` to accept
            genuinely fractional input.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Margins are always fitted in float64
            and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK.
            ``None`` draws an unseeded vector, so runs are no longer bit-for-bit
            reproducible.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests
            machine precision; larger values stop sooner and less accurately.
        return_operator: Whether to retain the uncentered operator in the
            result.

    Returns:
        Row and column principal coordinates, inertias, masses, metadata, and
        optionally the fitted operator.

    Raises:
        ValueError: If counts, dimensions, masses, model parameters, or dtype
            are invalid.
        NotImplementedError: If a solver other than ARPACK is requested.

    Examples:
        >>> result = correspondence_analysis_matrix(counts, n_comps=2)
        >>> result.row_principal_coordinates.shape
        (counts.shape[0], 2)
    """
    return _compute_correspondence_analysis(
        X,
        n_comps,
        mask=None,
        model=model,
        alpha=alpha,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=return_operator,
    )


def correspondence_analysis(
    adata: AnnData,
    n_comps: int = 2,
    *,
    layer: str | None = None,
    mask_var: Any = _empty,
    use_highly_variable: bool | None = None,
    key_added: str | None = None,
    model: CorrespondenceModel = "poisson",
    alpha: AlphaLike | str = None,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    solver: Solver = "arpack",
    random_state: int | None = 0,
    tol: float = 0.0,
    copy: bool = False,
) -> AnnData | None:
    """Compute correspondence analysis and write results to AnnData.

    The default output keys are ``adata.obsm["X_ca"]``,
    ``adata.varm["CA"]``, and ``adata.uns["ca"]``. A variable mask defines the
    contingency table before row and column margins are fitted.

    Args:
        adata: AnnData object with observations in rows and variables in
            columns.
        n_comps: Number of correspondence axes to return. Must satisfy
            ``1 <= n_comps < min(n_obs, n_nonempty_columns)``, where
            ``n_nonempty_columns`` counts selected columns with nonzero mass.
        layer: Count layer to use. By default, use ``adata.X``.
        mask_var: Boolean array or ``adata.var`` key defining table columns.
            When omitted, use ``"highly_variable"`` if present; explicit
            ``None`` selects every variable.
        use_highly_variable: Deprecated Scanpy-compatible mask selector.
        key_added: Exact key used in ``obsm``, ``varm``, and ``uns``. Defaults
            to the conventional ``"X_ca"``, ``"CA"``, and ``"ca"`` keys.
        model: ``"poisson"`` for classical CA, standardizing by the Poisson
            variance ``mu_ij``, or ``"scaled_nb"`` for the experimental residual
            ordination, standardizing by ``mu_ij * (1 + alpha_j * mean_n *
            p_j)`` with ``mean_n`` the mean cell count total.
        alpha: Per-gene overdispersion of the ``scaled_nb`` model, as a scalar
            broadcast to every gene, a length-``adata.n_vars`` array, or an
            ``adata.var`` key. It is a dispersion, not a size: larger values
            mean more variance and ``alpha=0`` is the Poisson limit. An estimate
            reported as a size (``theta``, ``r``) must be inverted first. Must
            be nonnegative; values below ``1e-8`` use the Poisson limit.
            Required only when ``model="scaled_nb"``, and rejected otherwise.
        check_values: When ``True``, reject floating-point input whose values
            are not within ``1e-8`` of integers. Set it to ``False`` to accept
            genuinely fractional input.
        dtype: Representation and ARPACK calculation dtype, either
            ``"float64"`` or ``"float32"``. Margins are always fitted in float64
            and cast afterwards.
        solver: SVD solver. Only ``"arpack"`` is supported.
        random_state: Seed for the random starting vector handed to ARPACK.
            ``None`` draws an unseeded vector, so runs are no longer bit-for-bit
            reproducible.
        tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0`` requests
            machine precision; larger values stop sooner and less accurately.
        copy: If ``True``, return a modified copy. Otherwise mutate ``adata``
            and return ``None``.

    Returns:
        A modified AnnData object when ``copy=True``; otherwise ``None``.

    Raises:
        ValueError: If counts, dimensions, mask, masses, or model parameters
            are invalid.
        KeyError: If a requested layer, mask, or overdispersion key is absent.

    Examples:
        >>> correspondence_analysis(adata, layer="counts", n_comps=2)
        >>> adata.obsm["X_ca"].shape
        (adata.n_obs, 2)
    """
    if copy:
        adata = adata.to_memory() if adata.isbacked else adata.copy()
    X = _get_count_matrix(adata, layer=layer)
    resolved_mask = _resolve_mask_var(adata.var, mask_var, use_highly_variable)
    alpha_values = _resolve_alpha(alpha, adata, model)
    result = _compute_correspondence_analysis(
        X,
        n_comps,
        mask=resolved_mask.values,
        model=model,
        alpha=alpha_values,
        check_values=check_values,
        dtype=dtype,
        solver=solver,
        random_state=random_state,
        tol=tol,
        return_operator=False,
    )
    result.params["alpha"] = _serialize_alpha(alpha)
    if key_added is None:
        obsm_key, varm_key, uns_key = "X_ca", "CA", "ca"
    else:
        obsm_key = varm_key = uns_key = key_added
    columns = np.full((adata.n_vars, n_comps), np.nan, dtype=np.float64)
    columns[resolved_mask.values] = result.column_principal_coordinates
    adata.obsm[obsm_key] = result.row_principal_coordinates
    adata.varm[varm_key] = columns
    params = dict(result.params)
    params.update(
        {
            "layer": layer,
            "mask_var": resolved_mask.mask_var,
            "use_highly_variable": resolved_mask.use_highly_variable,
            "mask_var_details": resolved_mask.details,
        }
    )
    adata.uns[uns_key] = {
        "singular_values": result.singular_values,
        "principal_inertias": result.principal_inertias,
        "inertia_ratio": result.inertia_ratio,
        "total_inertia": result.total_inertia,
        "row_masses": result.row_masses,
        "column_masses": result.column_masses,
        "params": params,
    }
    return adata if copy else None
