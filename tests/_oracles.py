"""Shared independent dense formulas used by simulated and real-data tests."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse, stats

from sparse_count_pca._residuals import ALPHA_EPS
from tests.proportion_shifted_clr_reference.oracle import proportion_shifted_clr
from tests.shifted_clr_reference.oracle import count_shifted_clr

DenseOrSparse: TypeAlias = ArrayLike | sparse.spmatrix | sparse.sparray
Float64Array: TypeAlias = NDArray[np.float64]


def _as_dense_float64(X: DenseOrSparse) -> Float64Array:
    """Return dense float64 values for a dense or SciPy sparse input."""
    values = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return np.asarray(values, dtype=np.float64)


def _dense_clr(logged: ArrayLike) -> Float64Array:
    """Center logged values independently within every row."""
    logged = np.asarray(logged, dtype=np.float64)
    return logged - logged.mean(axis=1, keepdims=True)


def _dense_count_shifted_clr(X: DenseOrSparse, count_shift: float) -> Float64Array:
    """Evaluate the count-scale shifted-CLR reference definition densely."""
    return count_shifted_clr(X, count_shift)


def _dense_proportion_shifted_clr(
    X: DenseOrSparse, composition_shift: float
) -> Float64Array:
    """Evaluate the historical composition-scale shifted-CLR definition densely."""
    return proportion_shifted_clr(X, composition_shift)


def _dense_log1p_norm(
    X: DenseOrSparse,
    size_factors: ArrayLike,
) -> Float64Array:
    """Evaluate size-factor normalization followed by log1p densely."""
    values = _as_dense_float64(X)
    divisors = np.asarray(size_factors, dtype=np.float64)
    return np.log1p(values / divisors[:, None])


def _dense_dirichlet(
    X: DenseOrSparse,
    concentration: float,
    prior_proportions: ArrayLike | None,
    *,
    clr: bool,
) -> Float64Array:
    """Evaluate a Dirichlet posterior-mean log transform densely."""
    values = _as_dense_float64(X)
    if prior_proportions is None:
        prior = np.full(values.shape[1], 1.0 / values.shape[1])
    else:
        prior = np.asarray(prior_proportions, dtype=np.float64)
    posterior = (values + concentration * prior) / (
        values.sum(axis=1, keepdims=True) + concentration
    )
    logged = np.log(posterior)
    return _dense_clr(logged) if clr else logged


def _dense_correspondence(
    X: DenseOrSparse,
    *,
    model: str = "poisson",
    alpha: ArrayLike | None = None,
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """Evaluate classical or scaled-NB correspondence matrices densely."""
    observed = _as_dense_float64(X)
    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    grand_total = row_totals.sum(dtype=np.float64)
    row_masses = row_totals / grand_total
    column_masses = column_totals / grand_total
    expected = np.outer(row_totals, column_masses)

    if model == "poisson":
        probabilities = observed / grand_total
        independence = np.outer(row_masses, column_masses)
        standardized = (probabilities - independence) / np.sqrt(independence)
    elif model == "scaled_nb":
        if alpha is None:
            raise ValueError("alpha is required for the scaled_nb oracle")
        alpha_values = np.broadcast_to(
            np.asarray(alpha, dtype=np.float64), column_masses.shape
        )
        effective_alpha = np.where(alpha_values < ALPHA_EPS, 0.0, alpha_values)
        variance_scale = 1.0 + effective_alpha * row_totals.mean() * column_masses
        residuals = (observed - expected) / np.sqrt(expected * variance_scale[None, :])
        standardized = residuals / np.sqrt(grand_total)
    else:
        raise ValueError(f"Unknown correspondence model: {model!r}")

    return standardized, row_masses, column_masses


# Deviance oracles must not restate the algebra in ``_residuals``, in any grouping:
# an oracle sharing production's derivation agrees with it even when both are wrong.
def _poisson_deviance_values(X: Float64Array, mu: Float64Array) -> Float64Array:
    """Return Poisson deviance from SciPy log-likelihoods for aligned means."""
    # The saturated Poisson mean is the observation itself.
    return 2.0 * (stats.poisson.logpmf(X, X) - stats.poisson.logpmf(X, mu))


def _dense_poisson_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Float64Array:
    """Evaluate Poisson deviance residuals from SciPy log-likelihoods."""
    X = np.asarray(X, dtype=np.float64)
    mu = (
        np.asarray(n, dtype=np.float64)[:, None]
        * np.asarray(p, dtype=np.float64)[None, :]
    )
    deviance = _poisson_deviance_values(X, mu)
    return np.sign(X - mu) * np.sqrt(np.maximum(deviance, 0.0))


def _dense_binomial_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Float64Array:
    """Evaluate binomial deviance residuals from SciPy log-likelihoods."""
    X = np.asarray(X, dtype=np.float64)
    trials = np.asarray(n, dtype=np.float64)[:, None]
    proportions = np.asarray(p, dtype=np.float64)[None, :]
    mu = trials * proportions
    # The saturated binomial proportion is X / n.
    observed = X / trials
    deviance = 2.0 * (
        stats.binom.logpmf(X, trials, observed)
        - stats.binom.logpmf(X, trials, proportions)
    )
    return np.sign(X - mu) * np.sqrt(np.maximum(deviance, 0.0))


def _dense_scaled_nb_deviance(
    X: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    alpha: ArrayLike,
) -> Float64Array:
    """Evaluate scaled-NB deviance residuals from SciPy log-likelihoods."""
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
    d[poisson] = _poisson_deviance_values(X, mu)[poisson]

    # SciPy parameterizes nbinom by successes r and success probability q, not by
    # a mean: mean = r(1-q)/q and variance = mean + mean**2 / r. So a target mean m
    # needs q = r / (r + m), and matching variance mu + alpha_tilde * mu**2 gives
    # r = 1 / alpha_tilde. The saturated fit holds r and puts the mean on X.
    nb = ~poisson
    shape = 1.0 / np.where(nb, alpha_tilde, 1.0)
    log_likelihood_null = stats.nbinom.logpmf(X, shape, shape / (shape + mu))
    log_likelihood_saturated = stats.nbinom.logpmf(X, shape, shape / (shape + X))
    d[nb] = (2.0 * (log_likelihood_saturated - log_likelihood_null))[nb]
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
            variance = mu * (1.0 + effective_alpha[None, :] * np.mean(n) * p[None, :])
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
    rtol: float = 0.0,
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
