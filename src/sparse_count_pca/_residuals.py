"""Pearson and deviance residual representation builders."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.special import xlogy

from ._clip import ClipMode, apply_clipping
from ._counts import BoolArray
from ._representation import SparseLowRankMatrix

ALPHA_EPS = 1e-8
RELATIVE_DEVIANCE_SERIES_THRESHOLD = 0.125
DEVIANCE_ROUNDING_TOLERANCE = 64.0 * np.finfo(np.float64).eps
_SERIES_TERMS = 24

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


def build_pearson_residual_representation(
    X: sparse.csr_matrix,
    n: Float64Array,
    p: Float64Array,
    *,
    model: Literal["poisson", "scaled_nb"],
    alpha: Float64Array | None,
) -> SparseLowRankMatrix:
    """Build an unclipped Poisson or scaled-NB Pearson representation."""
    rows = np.repeat(np.arange(X.shape[0], dtype=np.intp), np.diff(X.indptr))
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

    denominator = np.sqrt(n[rows] * p[X.indices] * variance_scale[X.indices])
    sparse_part = sparse.csr_matrix(
        (
            X.data.astype(np.float64, copy=False) / denominator,
            # The persistent representation owns its support so later caller
            # mutation of X cannot change the fitted transform.
            X.indices.copy(),
            X.indptr.copy(),
        ),
        shape=X.shape,
    )
    left = -np.sqrt(n)
    right = np.sqrt(p / variance_scale)
    return SparseLowRankMatrix(sparse_part, left, right)


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

    Returns:
        A rank-one sparse-plus-low-rank representation.
    """
    alpha_array = None if alpha is None else np.asarray(alpha, dtype=np.float64)
    if residual == "pearson" and clip is None and model in {"poisson", "scaled_nb"}:
        return build_pearson_residual_representation(
            X,
            n,
            p,
            model=model,
            alpha=alpha_array,
        )

    rows = np.repeat(np.arange(X.shape[0], dtype=np.intp), np.diff(X.indptr))
    cols = X.indices
    x = X.data.astype(np.float64, copy=False)
    n_support = n[rows]
    p_support = p[cols]
    mu = n_support * p_support
    mean_n = float(np.mean(n))

    u = -np.sqrt(n)
    if residual == "pearson":
        if model == "poisson":
            v = np.sqrt(p)
            variance = mu
        elif model == "binomial":
            v = np.sqrt(p / (1.0 - p))
            variance = mu * (1.0 - p_support)
        else:
            assert alpha_array is not None
            poisson = alpha_array < ALPHA_EPS
            effective_alpha = np.where(poisson, 0.0, alpha_array)
            scale = 1.0 + effective_alpha * mean_n * p
            v = np.sqrt(p / scale)
            variance = mu * scale[cols]
        residual_nonzero = (x - mu) / np.sqrt(variance)
    else:
        if model == "poisson":
            v = np.sqrt(2.0 * p)
            deviance = _poisson_deviance(x, mu)
        elif model == "binomial":
            v = np.sqrt(2.0 * (-np.log1p(-p)))
            deviance = _binomial_deviance(x, n_support, mu)
        else:
            assert alpha_array is not None
            poisson = alpha_array < ALPHA_EPS
            v = np.empty_like(p)
            v[poisson] = np.sqrt(2.0 * p[poisson])
            nb = ~poisson
            z = alpha_array[nb] * mean_n
            v[nb] = np.sqrt(2.0 * np.log1p(z * p[nb]) / z)
            scale_i = n / mean_n
            alpha_tilde = alpha_array[cols] / scale_i[rows]
            deviance = _scaled_nb_deviance(x, mu, alpha_tilde, poisson[cols])
        residual_nonzero = np.sign(x - mu) * np.sqrt(deviance)

    if clip is not None and clip_mode == "upper":
        # ``v_j`` is zero exactly for a gene with no counts, whose structural
        # zeros are already zero rather than negative. Upper clipping still
        # leaves them alone, so the invariant only needs ``v >= 0``.
        assert (u < 0).all() and (v >= 0).all()
    S = apply_clipping(
        X,
        residual_nonzero,
        u,
        v,
        rows,
        clip=clip,
        clip_mode=clip_mode,
        clip_max_nnz_ratio=clip_max_nnz_ratio,
    )
    return SparseLowRankMatrix(S, u, v)
