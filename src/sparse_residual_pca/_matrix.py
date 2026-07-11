from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from anndata.abc import CSCDataset, CSRDataset
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse

from ._clip import ClipMode, validate_clip
from ._operator import ResidualLinearOperator, _normalize_operator_dtype
from ._residuals import (
    AlphaLike,
    Model,
    ResidualType,
    _validate_model,
    build_residual_representation,
)
from ._svd import Solver, compute_truncated_svd
from ._version import __version__

CountMatrix: TypeAlias = (
    ArrayLike | sparse.spmatrix | sparse.sparray | CSRDataset | CSCDataset
)
FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
COUNT_INTEGER_ATOL = 1e-8


def _validate_boolean_mask(
    mask: ArrayLike, n_vars: int, *, name: str
) -> BoolArray:
    """Require a one-dimensional, genuinely boolean variable mask."""
    values = np.asarray(mask)
    if values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be genuinely boolean")
    if values.ndim != 1 or values.shape[0] != n_vars:
        raise ValueError(f"{name} must be a boolean vector with length n_vars")
    return values


@dataclass
class ResidualPCAResult:
    """Outputs from implicit residual PCA.

    Attributes:
        scores: Cell scores with shape ``(n_obs, n_comps)``.
        components: Principal axes with shape ``(n_comps, n_vars_used)``.
        loadings: Transposed components with shape
            ``(n_vars_used, n_comps)``.
        singular_values: Singular values in descending order.
        explained_variance: Per-component sample variance.
        explained_variance_ratio: Fraction of total centered variance.
        mean: Column means of the uncentered residual matrix.
        total_variance: Total sample variance of the centered residual matrix.
        params: Parameters used for the computation.
        operator: Centered residual operator when requested, otherwise
            ``None``.
    """

    scores: FloatArray
    components: FloatArray
    loadings: FloatArray
    singular_values: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    mean: FloatArray
    total_variance: float
    params: dict[str, Any]
    operator: ResidualLinearOperator | None = None


PCAResult = ResidualPCAResult


def _canonicalize_counts(X: CountMatrix, *, check_values: bool) -> sparse.csr_matrix:
    """Convert a count matrix to canonical CSR form and validate its data.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix.
        check_values: Whether floating-point entries must be integer-like.

    Returns:
        A sorted CSR matrix with duplicate and explicit-zero entries removed.

    Raises:
        ValueError: If entries are nonfinite, negative, or non-integer-like.
    """
    if isinstance(X, (CSRDataset, CSCDataset)):
        X = X.to_memory()

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
    else:
        X = np.asarray(X)
        if not (
            np.issubdtype(X.dtype, np.integer)
            or np.issubdtype(X.dtype, np.floating)
        ):
            raise ValueError("Input counts must have a real numeric dtype")
        warnings.warn(
            "Dense input was converted to CSR. This may require substantial memory.",
            UserWarning,
            stacklevel=3,
        )
        X = sparse.csr_matrix(X)

    if not (
        np.issubdtype(X.dtype, np.integer)
        or np.issubdtype(X.dtype, np.floating)
    ):
        raise ValueError("Input counts must have a real numeric dtype")
    X.sum_duplicates()
    X.sort_indices()
    if not np.isfinite(X.data).all():
        raise ValueError("Input contains NaN or inf")
    if (X.data < 0).any():
        raise ValueError("Input contains negative values")
    data_float64 = X.data.astype(np.float64)
    if X.dtype.itemsize > np.dtype(np.float64).itemsize or np.issubdtype(
        X.dtype, np.integer
    ):
        if not np.array_equal(data_float64.astype(X.dtype), X.data):
            raise ValueError("Input counts cannot be represented exactly as float64")
    if (
        check_values
        and np.issubdtype(X.dtype, np.floating)
        and not np.allclose(
            X.data, np.rint(X.data), rtol=0.0, atol=COUNT_INTEGER_ATOL
        )
    ):
        raise ValueError(
            "Input contains non-integer values; pass check_values=False to skip"
        )
    X.eliminate_zeros()
    return X


def _serialize_matrix_alpha(alpha: AlphaLike) -> Any:
    """Convert matrix overdispersion input to a stable result value."""
    if alpha is None or np.isscalar(alpha):
        return alpha
    values = np.asarray(alpha)
    return values.item() if values.ndim == 0 else values


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
) -> ResidualPCAResult:
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
        A populated ``ResidualPCAResult``.

    Raises:
        ValueError: If counts, dimensions, or model parameters are invalid.
    """
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    dtype = _normalize_operator_dtype(dtype)
    validate_clip(clip, clip_mode, clip_max_nnz_ratio)
    X = _canonicalize_counts(X, check_values=check_values)
    n_obs, n_vars = X.shape
    alpha_full = _validate_model(model, residual, alpha, n_vars)

    n = np.asarray(X.astype(np.float64).sum(axis=1)).ravel()
    if (n == 0).any():
        raise ValueError("Cells with zero total counts are not supported")
    total = float(np.sum(n))
    p_full = np.asarray(X.astype(np.float64).sum(axis=0)).ravel() / total

    if mask is None:
        mask = np.ones(n_vars, dtype=bool)
    else:
        mask = _validate_boolean_mask(mask, n_vars, name="mask")
    if not mask.any():
        raise ValueError("mask selected zero genes")

    p = p_full[mask]
    if (p == 0).any():
        raise ValueError("Selected genes with zero total counts are not supported")
    if model == "binomial" and (p >= 1).any():
        raise ValueError("Binomial residuals require 0 < p_j < 1")

    alpha_used = alpha_full[mask] if alpha_full is not None else None

    X_used = X[:, mask].tocsr()
    if not 1 <= n_comps < min(n_obs, X_used.shape[1]):
        raise ValueError(
            "n_comps must satisfy 1 <= n_comps < min(n_obs, n_vars_used) "
            "when solver='arpack'"
        )

    representation = build_residual_representation(
        X_used,
        n,
        p,
        model=model,
        residual=residual,
        alpha=alpha_used,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
    )
    operator = ResidualLinearOperator(
        representation.sparse,
        representation.left[:, 0],
        representation.right[:, 0],
        center=True,
        dtype=dtype,
    )
    centered_squared = operator.frobenius_squared_centered()
    if operator.centered_variance_is_numerically_zero():
        raise ValueError(
            "Residual matrix has numerically zero centered variance; "
            "PCA directions are undefined"
        )
    total_variance = centered_squared / (n_obs - 1)
    decomposition = compute_truncated_svd(
        operator,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    U = decomposition.left_vectors
    singular_values = decomposition.singular_values
    components = decomposition.right_vectors
    scores = np.asarray(U * singular_values, dtype=np.dtype(dtype))
    components = np.asarray(components, dtype=np.dtype(dtype))
    loadings = components.T

    explained_variance = singular_values**2 / (n_obs - 1)
    explained_variance_ratio = explained_variance / total_variance
    params = {
        "model": model,
        "residual": residual,
        "alpha": _serialize_matrix_alpha(alpha),
        "clip": clip,
        "clip_mode": clip_mode,
        "clip_max_nnz_ratio": clip_max_nnz_ratio,
        "zero_center": True,
        "use_highly_variable": False,
        "mask_var": None,
        "layer": None,
        "solver": solver,
        "n_comps": n_comps,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": str(np.dtype(dtype)),
        "package_version": __version__,
    }
    assert operator.mean is not None
    return ResidualPCAResult(
        scores=scores,
        components=components,
        loadings=loadings,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=operator.mean.copy(),
        total_variance=float(total_variance),
        params=params,
        operator=operator if return_operator else None,
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
) -> ResidualPCAResult:
    """Compute PCA of implicitly represented residuals from a count matrix.

    The entire matrix defines cell totals and gene proportions. Residuals are
    represented without materializing the dense residual matrix, centered by
    column, and decomposed with ARPACK.

    Args:
        X: Dense, SciPy sparse, or backed sparse count matrix with cells in
            rows.
        n_comps: Number of principal components.
        model: Null model: ``"poisson"``, ``"binomial"``, or ``"scaled_nb"``.
        residual: Residual type: ``"pearson"`` or ``"deviance"``.
        alpha: Scalar or per-gene scaled-NB overdispersion.
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.
        check_values: Whether floating-point entries must be integer-like.
        dtype: Storage and operator floating-point dtype.
        solver: SVD solver. Version 1 supports only ``"arpack"``.
        random_state: Seed for the ARPACK starting vector.
        tol: ARPACK convergence tolerance.
        return_operator: Whether to include the centered residual operator in
            the result.

    Returns:
        Residual PCA scores, components, variance statistics, and metadata.

    Raises:
        ValueError: If counts, dimensions, or model parameters are invalid.
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
