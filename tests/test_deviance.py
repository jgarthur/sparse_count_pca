from decimal import Decimal, localcontext

import numpy as np
import pytest

from sparse_residual_pca._residuals import (
    RELATIVE_DEVIANCE_SERIES_THRESHOLD,
    _binomial_deviance,
    _poisson_deviance,
    _scaled_nb_deviance,
)


def _d(value):
    return value if isinstance(value, Decimal) else Decimal(value)


def _xlog_ratio(x, denominator):
    x = _d(x)
    denominator = _d(denominator)
    return Decimal(0) if x == 0 else x * (x / denominator).ln()


def _poisson_oracle(x, mu):
    with localcontext() as context:
        context.prec = 100
        x = _d(x)
        mu = _d(mu)
        return float(2 * (_xlog_ratio(x, mu) - (x - mu)))


def _binomial_oracle(x, n, mu):
    with localcontext() as context:
        context.prec = 100
        x = _d(x)
        n = _d(n)
        mu = _d(mu)
        return float(
            2
            * (
                _xlog_ratio(x, mu)
                + _xlog_ratio(n - x, n - mu)
            )
        )


def _scaled_nb_oracle(x, mu, alpha):
    with localcontext() as context:
        context.prec = 100
        x = _d(x)
        mu = _d(mu)
        alpha = _d(alpha)
        first = _xlog_ratio(x, mu)
        log_difference = (1 + alpha * x).ln() - (1 + alpha * mu).ln()
        second = (x + 1 / alpha) * log_difference
        return float(2 * (first - second))


@pytest.mark.parametrize(
    ("x", "mu"),
    [
        (0, 3),
        (10**15, 10**15),
        (10**15 + 1, 10**15),
        (10**15 + 10, 10**15),
    ],
)
def test_poisson_deviance_matches_high_precision_near_mean(x, mu):
    actual = _poisson_deviance(
        np.array([x], dtype=np.float64), np.array([mu], dtype=np.float64)
    )[0]
    assert actual == pytest.approx(_poisson_oracle(x, mu), rel=2e-15, abs=1e-30)


@pytest.mark.parametrize(
    "x", [0, 10**15, 10**15 + 1, 10**15 + 10, 2 * 10**15]
)
def test_binomial_deviance_matches_high_precision_near_mean_and_boundaries(x):
    n = 2 * 10**15
    mu = 10**15
    actual = _binomial_deviance(
        np.array([x], dtype=np.float64),
        np.array([n], dtype=np.float64),
        np.array([mu], dtype=np.float64),
    )[0]
    assert actual == pytest.approx(
        _binomial_oracle(x, n, mu), rel=2e-15, abs=1e-30
    )


@pytest.mark.parametrize("difference", [1, 10])
@pytest.mark.parametrize("alpha", [Decimal("1e-12"), Decimal("0.1")])
def test_scaled_nb_deviance_matches_high_precision_near_mean(difference, alpha):
    mu = 10**15
    x = mu + difference
    actual = _scaled_nb_deviance(
        np.array([x], dtype=np.float64),
        np.array([mu], dtype=np.float64),
        np.array([float(alpha)]),
        np.array([False]),
    )[0]
    assert actual == pytest.approx(
        _scaled_nb_oracle(x, mu, alpha), rel=3e-14, abs=1e-35
    )


@pytest.mark.parametrize(
    ("x", "mu", "alpha"),
    [(0, 3, Decimal("0.1")), (10**6, 10**6, Decimal("0.1"))],
)
def test_scaled_nb_deviance_matches_high_precision_at_boundaries(x, mu, alpha):
    actual = _scaled_nb_deviance(
        np.array([x], dtype=np.float64),
        np.array([mu], dtype=np.float64),
        np.array([float(alpha)]),
        np.array([False]),
    )[0]
    assert actual == pytest.approx(
        _scaled_nb_oracle(x, mu, alpha), rel=3e-14, abs=1e-35
    )


@pytest.mark.parametrize(
    "side",
    [
        np.nextafter(RELATIVE_DEVIANCE_SERIES_THRESHOLD, 0.0),
        np.nextafter(RELATIVE_DEVIANCE_SERIES_THRESHOLD, 1.0),
    ],
)
def test_deviance_is_accurate_on_both_sides_of_series_switch(side):
    mu = 10**8
    x = mu + side * mu
    poisson = _poisson_deviance(np.array([x]), np.array([mu]))[0]
    nb = _scaled_nb_deviance(
        np.array([x]),
        np.array([mu]),
        np.array([1e-6]),
        np.array([False]),
    )[0]
    assert poisson == pytest.approx(
        _poisson_oracle(Decimal.from_float(x), Decimal.from_float(mu)), rel=2e-14
    )
    assert nb == pytest.approx(
        _scaled_nb_oracle(
            Decimal.from_float(x), Decimal.from_float(mu), Decimal("1e-6")
        ),
        rel=2e-14,
    )


def test_scaled_nb_small_positive_dispersion_approaches_poisson_without_switching():
    mu = 10**8
    x = mu + 10
    alpha = Decimal("1e-14")
    nb = _scaled_nb_deviance(
        np.array([x], dtype=np.float64),
        np.array([mu], dtype=np.float64),
        np.array([float(alpha)]),
        np.array([False]),
    )[0]
    poisson = _poisson_deviance(
        np.array([x], dtype=np.float64), np.array([mu], dtype=np.float64)
    )[0]
    assert nb == pytest.approx(_scaled_nb_oracle(x, mu, alpha), rel=3e-14)
    assert nb == pytest.approx(poisson, rel=2e-6)
