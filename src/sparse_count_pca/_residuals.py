"""Pearson and deviance residual representation builders."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse
from scipy.special import xlogy

from ._clip import ClipMode, _guarded_clipped_zero_locations
from ._counts import BoolArray
from ._operator import _normalize_operator_dtype
from ._representation import SparseLowRankMatrix

ALPHA_EPS = 1e-8
RELATIVE_DEVIANCE_SERIES_THRESHOLD = 0.125
DEVIANCE_ROUNDING_TOLERANCE = 64.0 * np.finfo(np.float64).eps
_SERIES_TERMS = 24
_RESIDUAL_SUPPORT_BLOCK_SIZE = 1_000_000

Model: TypeAlias = Literal["poisson", "binomial", "scaled_nb"]
ResidualType: TypeAlias = Literal["pearson", "deviance"]
AlphaLike: TypeAlias = float | ArrayLike | None
Float64Array: TypeAlias = NDArray[np.float64]

SUPPORTED: set[tuple[Model, ResidualType]] = {
    ("poisson", "pearson"),
    ("poisson", "deviance"),
    ("binomial", "pearson"),
    ("binomial", "deviance"),
    ("scaled_nb", "pearson"),
    ("scaled_nb", "deviance"),
}


def _support_row_blocks(
    X: sparse.csr_matrix,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield whole-row blocks containing about the configured number of values."""
    row_start = 0
    while row_start < X.shape[0]:
        target = min(
            int(X.indptr[row_start]) + _RESIDUAL_SUPPORT_BLOCK_SIZE,
            X.nnz,
        )
        row_stop = int(np.searchsorted(X.indptr, target, side="right")) - 1
        row_stop = min(X.shape[0], max(row_start + 1, row_stop))
        value_start = int(X.indptr[row_start])
        value_stop = int(X.indptr[row_stop])
        yield row_start, row_stop, value_start, value_stop
        row_start = row_stop


def _clipped_zero_corrections(
    X: sparse.csr_matrix,
    left: Float64Array,
    right: Float64Array,
    *,
    clip: float | None,
    clip_mode: ClipMode,
    clip_max_nnz_ratio: float | None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return exact structural-zero corrections required by symmetric clipping."""
    empty = np.empty(0, dtype=np.intp)
    if clip is None or clip_mode != "symmetric" or X.nnz == 0:
        return empty, empty
    return _guarded_clipped_zero_locations(
        X,
        left,
        right,
        clip,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
    )


def _finalize_residual_representation(
    X: sparse.csr_matrix,
    data: NDArray[np.floating],
    left: Float64Array,
    right: Float64Array,
    correction_rows: NDArray[np.intp],
    correction_cols: NDArray[np.intp],
    *,
    clip: float | None,
    dtype: np.dtype[np.floating],
    eliminate_zeros: bool,
) -> SparseLowRankMatrix:
    """Own CSR support, add clipped zeros, and cast factors for final storage."""
    sparse_part = sparse.csr_matrix(
        (
            data,
            # The persistent representation owns its support so later caller
            # mutation of X cannot change the fitted transform.
            X.indices.copy(),
            X.indptr.copy(),
        ),
        shape=X.shape,
    )
    if eliminate_zeros:
        sparse_part.eliminate_zeros()
    if correction_rows.size:
        assert clip is not None
        correction_data = np.asarray(
            -clip - left[correction_rows] * right[correction_cols],
            dtype=dtype,
        )
        corrections = sparse.csr_matrix(
            (correction_data, (correction_rows, correction_cols)),
            shape=X.shape,
        )
        sparse_part = (sparse_part + corrections).tocsr()
        if eliminate_zeros:
            sparse_part.eliminate_zeros()
    return SparseLowRankMatrix(
        sparse_part,
        left.astype(dtype, copy=False),
        right.astype(dtype, copy=False),
    )


def build_pearson_residual_representation(
    X: sparse.csr_matrix,
    n: Float64Array,
    p: Float64Array,
    *,
    model: Literal["poisson", "scaled_nb"],
    alpha: Float64Array | None,
    clip: float | None = None,
    clip_mode: ClipMode = "symmetric",
    clip_max_nnz_ratio: float | None = None,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Build a bounded-memory Poisson or scaled-NB Pearson representation."""
    calculation_dtype = _normalize_operator_dtype(dtype)
    if model == "poisson":
        if alpha is not None:
            raise ValueError("alpha is only used for model='scaled_nb'")
        variance_scale = np.ones_like(p)
    elif model == "scaled_nb":
        if alpha is None:
            raise ValueError("alpha is required for model='scaled_nb'")
        alpha_array = np.asarray(alpha, dtype=np.float64)
        poisson = alpha_array < ALPHA_EPS
        effective_alpha = np.where(poisson, 0.0, alpha_array)
        variance_scale = 1.0 + effective_alpha * float(np.mean(n)) * p
    else:
        raise ValueError(f"Unsupported Pearson residual model: {model!r}")

    left = -np.sqrt(n)
    right = np.sqrt(p / variance_scale)
    if clip is not None and clip_mode == "upper":
        assert (left < 0).all() and (right >= 0).all()
    correction_rows, correction_cols = _clipped_zero_corrections(
        X,
        left,
        right,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
    )

    data = np.empty(X.nnz, dtype=calculation_dtype)
    for row_start, row_stop, value_start, value_stop in _support_row_blocks(X):
        rows = np.repeat(
            np.arange(row_start, row_stop, dtype=np.intp),
            np.diff(X.indptr[row_start : row_stop + 1]),
        )
        columns = X.indices[value_start:value_stop]
        x = X.data[value_start:value_stop].astype(np.float64, copy=False)
        denominator = np.sqrt(n[rows] * p[columns] * variance_scale[columns])
        if clip is None:
            values = x / denominator
        else:
            mu = n[rows] * p[columns]
            values = (x - mu) / denominator
            if clip_mode == "upper":
                np.minimum(values, clip, out=values)
            else:
                np.clip(values, -clip, clip, out=values)
            values -= left[rows] * right[columns]
        data[value_start:value_stop] = values

    return _finalize_residual_representation(
        X,
        data,
        left,
        right,
        correction_rows,
        correction_cols,
        clip=clip,
        dtype=calculation_dtype,
        eliminate_zeros=clip is not None,
    )


def _validate_model(
    model: Model,
    residual: ResidualType,
    alpha: AlphaLike,
    n_vars: int,
    *,
    ignore: BoolArray | None = None,
) -> Float64Array | None:
    """Validate a residual specification and normalize overdispersion.

    Args:
        model: Null model name.
        residual: Residual type.
        alpha: Scalar or per-gene overdispersion for the scaled-NB model.
        n_vars: Number of genes represented by ``alpha``.
        ignore: Genes whose overdispersion cannot affect any output. Their
            values are replaced with zero instead of being validated.

    Returns:
        A float64 overdispersion vector for the scaled-NB model, otherwise
        ``None``.

    Raises:
        ValueError: If the model specification or overdispersion is invalid.
    """
    if (model, residual) not in SUPPORTED:
        raise ValueError(
            f"Unsupported model/residual combination: {(model, residual)!r}"
        )
    if model == "scaled_nb" and alpha is None:
        raise ValueError("alpha is required for model='scaled_nb'")
    if model != "scaled_nb" and alpha is not None:
        raise ValueError("alpha is only used for model='scaled_nb'")
    if model != "scaled_nb":
        return None

    alpha_array = np.asarray(alpha, dtype=np.float64)
    if alpha_array.ndim == 0:
        alpha_array = np.full(n_vars, float(alpha_array), dtype=np.float64)
    if alpha_array.ndim != 1 or alpha_array.shape[0] != n_vars:
        raise ValueError("alpha must be a scalar or have shape (n_vars,)")
    if ignore is not None and ignore.any():
        # Overdispersion scales only its own gene's residual variance. A gene
        # with no counts contributes no residual, so its value cannot reach any
        # output; replace it with zero rather than validating it. Shape is
        # still checked above, so a wrong-length array still raises.
        alpha_array = np.where(ignore, 0.0, alpha_array)
    if not np.isfinite(alpha_array).all():
        raise ValueError("alpha must contain only finite values")
    if (alpha_array < 0).any():
        raise ValueError("alpha must be nonnegative")
    return alpha_array


def _relative_entropy_increment(
    base: Float64Array, delta: Float64Array
) -> Float64Array:
    """Evaluate ``(base + delta) log1p(delta / base) - delta`` stably."""
    relative = delta / base
    result = np.empty_like(relative)
    boundary = delta == -base
    result[boundary] = base[boundary]

    series = (~boundary) & (np.abs(relative) <= RELATIVE_DEVIANCE_SERIES_THRESHOLD)
    r = relative[series]
    power = r * r
    series_sum = power / 2.0
    # Iterating over the small fixed order avoids a terms-by-values temporary.
    for order in range(3, _SERIES_TERMS + 1):
        power *= -r
        series_sum += power / (order * (order - 1))
    result[series] = base[series] * series_sum

    direct = ~(boundary | series)
    result[direct] = (base[direct] + delta[direct]) * np.log1p(
        relative[direct]
    ) - delta[direct]
    return result


def _check_deviance_rounding(deviance: Float64Array) -> Float64Array:
    """Clamp final-rounding negatives and reject materially negative values."""
    if (deviance < -DEVIANCE_ROUNDING_TOLERANCE).any():
        minimum = float(np.min(deviance))
        raise FloatingPointError(
            f"Stable deviance calculation produced a negative value ({minimum})"
        )
    return np.maximum(deviance, 0.0)


def _poisson_deviance(x: Float64Array, mu: Float64Array) -> Float64Array:
    """Compute ``2 * (x log(x / mu) - (x - mu))`` stably."""
    result = 2.0 * _relative_entropy_increment(mu, x - mu)
    return _check_deviance_rounding(result)


def _binomial_deviance(
    x: Float64Array,
    n: Float64Array,
    mu: Float64Array,
) -> Float64Array:
    """Compute success-plus-failure binomial deviance stably."""
    delta = x - mu
    result = 2.0 * (
        _relative_entropy_increment(mu, delta)
        + _relative_entropy_increment(n - mu, -delta)
    )
    return _check_deviance_rounding(result)


def _scaled_nb_deviance_series(
    x: Float64Array,
    mu: Float64Array,
    alpha_tilde: Float64Array,
) -> Float64Array:
    """Evaluate near-mean scaled-NB deviance as a relative-error series."""
    relative = (x - mu) / mu
    alpha_mu = alpha_tilde * mu
    power = relative * relative
    series_sum = np.zeros_like(relative)
    small_alpha_mu = alpha_mu <= 1.0
    rho = np.empty_like(alpha_mu)
    rho[small_alpha_mu] = alpha_mu[small_alpha_mu] / (1.0 + alpha_mu[small_alpha_mu])
    one_minus_rho = np.empty_like(alpha_mu)
    one_minus_rho[~small_alpha_mu] = 1.0 / (1.0 + alpha_mu[~small_alpha_mu])
    factor = np.empty_like(relative)
    rho_power = rho[small_alpha_mu].copy()
    log_rho = np.log1p(-one_minus_rho[~small_alpha_mu])

    for order in range(2, _SERIES_TERMS + 1):
        exponent = order - 1
        factor[small_alpha_mu] = 1.0 - rho_power
        factor[~small_alpha_mu] = -np.expm1(exponent * log_rho)
        series_sum += power * factor / (order * exponent)
        power *= -relative
        rho_power *= rho[small_alpha_mu]
    return 2.0 * mu * series_sum


def _scaled_nb_deviance(
    x: Float64Array,
    mu: Float64Array,
    alpha_tilde: Float64Array,
    poisson: NDArray[np.bool_],
) -> Float64Array:
    """Compute scaled-NB deviance using a preselected probability family.

    Args:
        x: Counts on the sparse support.
        mu: Null-model means aligned with ``x``.
        alpha_tilde: Cell- and gene-adjusted overdispersion aligned with ``x``.
        poisson: Whether each value belongs to a gene using the Poisson limit.

    Returns:
        Elementwise deviance contributions as a float64 array.
    """
    result = np.empty_like(mu, dtype=np.float64)
    result[poisson] = _poisson_deviance(x[poisson], mu[poisson])
    nb = ~poisson
    a = alpha_tilde[nb]
    x_nb = x[nb]
    mu_nb = mu[nb]
    relative = (x_nb - mu_nb) / mu_nb
    use_series = np.abs(relative) <= RELATIVE_DEVIANCE_SERIES_THRESHOLD
    nb_result = np.empty_like(mu_nb, dtype=np.float64)
    nb_result[use_series] = _scaled_nb_deviance_series(
        x_nb[use_series], mu_nb[use_series], a[use_series]
    )

    direct = ~use_series
    if direct.any():
        x_direct = x_nb[direct]
        mu_direct = mu_nb[direct]
        a_direct = a[direct]
        delta = x_direct - mu_direct
        first = xlogy(x_direct, x_direct / mu_direct)
        positive_alpha = a_direct > 0
        second = np.empty_like(first)
        z = np.zeros_like(first)
        z[positive_alpha] = (
            a_direct[positive_alpha]
            * delta[positive_alpha]
            / (1.0 + a_direct[positive_alpha] * mu_direct[positive_alpha])
        )
        second[positive_alpha] = (
            (1.0 + a_direct[positive_alpha] * x_direct[positive_alpha])
            * np.log1p(z[positive_alpha])
            / a_direct[positive_alpha]
        )
        second[~positive_alpha] = delta[~positive_alpha]
        nb_result[direct] = 2.0 * (first - second)
    result[nb] = _check_deviance_rounding(nb_result)
    return result


def build_residual_representation(
    X: sparse.csr_matrix,
    n: Float64Array,
    p: Float64Array,
    *,
    model: Model,
    residual: ResidualType,
    alpha: Float64Array | None,
    clip: float | None,
    clip_mode: ClipMode,
    clip_max_nnz_ratio: float | None,
    dtype: DTypeLike = "float64",
) -> SparseLowRankMatrix:
    """Build a sparse-plus-rank-one residual representation.

    The uncentered residual matrix is represented as ``R = S + u v.T``.
    Residual values are evaluated only on the nonzero support of ``X``;
    ``u v.T`` supplies the zero-count residuals. Symmetric clipping may add
    sparse corrections at zero-count locations.

    Args:
        X: CSR count matrix with shape ``(n_obs, n_vars)``.
        n: Cell totals computed before variable masking.
        p: Gene proportions for the columns of ``X``.
        model: One of ``"poisson"``, ``"binomial"``, or ``"scaled_nb"``.
        residual: Either ``"pearson"`` or ``"deviance"``.
        alpha: Validated per-gene scaled-NB overdispersion, or ``None``.
        clip: Positive clipping threshold, or ``None``.
        clip_mode: Whether to clip symmetrically or only the upper tail.
        clip_max_nnz_ratio: Maximum sparse support-growth ratio for exact
            symmetric clipping, or ``None`` for no limit.
        dtype: Floating-point storage dtype for the returned representation.

    Returns:
        A rank-one sparse-plus-low-rank representation.
    """
    alpha_array = None if alpha is None else np.asarray(alpha, dtype=np.float64)
    if residual == "pearson" and model in {"poisson", "scaled_nb"}:
        return build_pearson_residual_representation(
            X,
            n,
            p,
            model=model,
            alpha=alpha_array,
            clip=clip,
            clip_mode=clip_mode,
            clip_max_nnz_ratio=clip_max_nnz_ratio,
            dtype=dtype,
        )

    calculation_dtype = _normalize_operator_dtype(dtype)
    mean_n = float(np.mean(n))
    left = -np.sqrt(n)
    poisson = None
    scale_i = None
    if residual == "pearson":
        assert model == "binomial"
        right = np.sqrt(p / (1.0 - p))
    elif model == "poisson":
        right = np.sqrt(2.0 * p)
    elif model == "binomial":
        right = np.sqrt(2.0 * (-np.log1p(-p)))
    else:
        assert alpha_array is not None
        poisson = alpha_array < ALPHA_EPS
        right = np.empty_like(p)
        right[poisson] = np.sqrt(2.0 * p[poisson])
        nb = ~poisson
        z = alpha_array[nb] * mean_n
        right[nb] = np.sqrt(2.0 * np.log1p(z * p[nb]) / z)
        scale_i = n / mean_n

    if clip is not None and clip_mode == "upper":
        # ``v_j`` is zero exactly for a gene with no counts, whose structural
        # zeros are already zero rather than negative. Upper clipping still
        # leaves them alone, so the invariant only needs ``v >= 0``.
        assert (left < 0).all() and (right >= 0).all()
    correction_rows, correction_cols = _clipped_zero_corrections(
        X,
        left,
        right,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
    )

    data = np.empty(X.nnz, dtype=calculation_dtype)
    for row_start, row_stop, value_start, value_stop in _support_row_blocks(X):
        rows = np.repeat(
            np.arange(row_start, row_stop, dtype=np.intp),
            np.diff(X.indptr[row_start : row_stop + 1]),
        )
        columns = X.indices[value_start:value_stop]
        x = X.data[value_start:value_stop].astype(np.float64, copy=False)
        n_support = n[rows]
        p_support = p[columns]
        mu = n_support * p_support

        if residual == "pearson":
            variance = mu * (1.0 - p_support)
            residual_nonzero = (x - mu) / np.sqrt(variance)
        else:
            if model == "poisson":
                deviance = _poisson_deviance(x, mu)
            elif model == "binomial":
                deviance = _binomial_deviance(x, n_support, mu)
            else:
                assert alpha_array is not None
                assert poisson is not None
                assert scale_i is not None
                alpha_tilde = alpha_array[columns] / scale_i[rows]
                deviance = _scaled_nb_deviance(
                    x,
                    mu,
                    alpha_tilde,
                    poisson[columns],
                )
            residual_nonzero = np.sign(x - mu) * np.sqrt(deviance)

        baseline = left[rows] * right[columns]
        if clip is None:
            values = residual_nonzero - baseline
        elif clip_mode == "upper":
            values = np.minimum(residual_nonzero, clip) - baseline
        else:
            values = np.clip(residual_nonzero, -clip, clip) - baseline
        data[value_start:value_stop] = values

    return _finalize_residual_representation(
        X,
        data,
        left,
        right,
        correction_rows,
        correction_cols,
        clip=clip,
        dtype=calculation_dtype,
        eliminate_zeros=True,
    )
