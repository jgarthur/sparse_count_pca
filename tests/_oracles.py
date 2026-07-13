"""Independent dense residual formulas used as test oracles."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.special import xlogy

from sparse_count_pca._residuals import ALPHA_EPS

DenseOrSparse: TypeAlias = ArrayLike | sparse.spmatrix | sparse.sparray
Float64Array: TypeAlias = NDArray[np.float64]


def _dense_relative_entropy(base: Float64Array, delta: Float64Array) -> Float64Array:
    relative = delta / base
    result = np.empty_like(relative)
    boundary = delta == -base
    result[boundary] = base[boundary]
    series = (~boundary) & (np.abs(relative) <= 0.125)
    r = relative[series]
    power = r * r
    total = power / 2.0
    for order in range(3, 25):
        power *= -r
        total += power / (order * (order - 1))
    result[series] = base[series] * total
    direct = ~(boundary | series)
    result[direct] = (
        (base[direct] + delta[direct]) * np.log1p(relative[direct])
        - delta[direct]
    )
    return result


def _dense_poisson_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Float64Array:
    mu = (
        np.asarray(n, dtype=np.float64)[:, None]
        * np.asarray(p, dtype=np.float64)[None, :]
    )
    X = np.asarray(X, dtype=np.float64)
    d = 2.0 * _dense_relative_entropy(mu, X - mu)
    return np.sign(X - mu) * np.sqrt(np.maximum(d, 0.0))


def _dense_binomial_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Float64Array:
    n = np.asarray(n, dtype=np.float64)[:, None]
    p = np.asarray(p, dtype=np.float64)[None, :]
    mu = n * p
    X = np.asarray(X, dtype=np.float64)
    delta = X - mu
    d = 2.0 * (
        _dense_relative_entropy(mu, delta)
        + _dense_relative_entropy(n - mu, -delta)
    )
    return np.sign(X - mu) * np.sqrt(np.maximum(d, 0.0))


def _dense_scaled_nb_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    alpha: ArrayLike,
) -> Float64Array:
    X = np.asarray(X, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), p.shape)
    mu = n[:, None] * p[None, :]
    scale = n / np.mean(n)
    alpha_tilde = alpha[None, :] / scale[:, None]
    poisson_genes = alpha < ALPHA_EPS
    poisson = np.broadcast_to(poisson_genes[None, :], mu.shape)
    d = np.empty_like(mu)
    poisson_d = 2.0 * _dense_relative_entropy(mu, X - mu)
    d[poisson] = poisson_d[poisson]
    a = alpha_tilde[~poisson]
    d[~poisson] = 2.0 * (
        xlogy(X[~poisson], X[~poisson] / mu[~poisson])
        - (X[~poisson] + 1.0 / a)
        * (np.log1p(a * X[~poisson]) - np.log1p(a * mu[~poisson]))
    )
    return np.sign(X - mu) * np.sqrt(np.maximum(d, 0.0))


def _materialize_dense_residual(
    X: DenseOrSparse,
    *,
    model: str = "poisson",
    residual: str = "pearson",
    alpha: ArrayLike | None = None,
    clip: float | None = None,
    clip_mode: str = "symmetric",
    center: bool = False,
) -> Float64Array:
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X = np.asarray(X, dtype=np.float64)
    n = X.sum(axis=1)
    p = X.sum(axis=0) / X.sum()
    mu = n[:, None] * p[None, :]

    if residual == "pearson":
        if model == "poisson":
            variance = mu
        elif model == "binomial":
            variance = mu * (1.0 - p[None, :])
        elif model == "scaled_nb":
            alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), p.shape)
            effective_alpha = np.where(alpha < ALPHA_EPS, 0.0, alpha)
            variance = mu * (
                1.0 + effective_alpha[None, :] * np.mean(n) * p[None, :]
            )
        else:
            raise ValueError(f"Unknown model: {model!r}")
        result = (X - mu) / np.sqrt(variance)
    elif model == "poisson":
        result = _dense_poisson_deviance(X, n, p)
    elif model == "binomial":
        result = _dense_binomial_deviance(X, n, p)
    elif model == "scaled_nb":
        result = _dense_scaled_nb_deviance(X, n, p, alpha)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    if clip is not None:
        if clip_mode == "symmetric":
            result = np.clip(result, -clip, clip)
        elif clip_mode == "upper":
            result = np.minimum(result, clip)
        else:
            raise ValueError(f"Unknown clip mode: {clip_mode!r}")
    if center:
        result = result - result.mean(axis=0)
    return result


def _compare_subspaces(
    left: ArrayLike,
    right: ArrayLike,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    left_q = np.linalg.qr(left)[0]
    right_q = np.linalg.qr(right)[0]
    return np.allclose(
        left_q @ left_q.T,
        right_q @ right_q.T,
        atol=atol,
        rtol=rtol,
    )
