"""Numerical-accuracy tests for deviance calculations."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from sparse_count_pca._residuals import (
    DEVIANCE_ROUNDING_TOLERANCE,
    RELATIVE_DEVIANCE_SERIES_THRESHOLD,
    _binomial_deviance,
    _check_deviance_rounding,
    _poisson_deviance,
    _relative_entropy_increment,
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
        return float(2 * (_xlog_ratio(x, mu) + _xlog_ratio(n - x, n - mu)))


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


def test_relative_entropy_increment_handles_boundaries_and_both_paths():
    """Relative entropy is stable at zero, support, and series boundaries."""
    base = np.array([3.0, 8.0, 8.0, 8.0, 8.0, 1e15, 1e15])
    delta = np.array(
        [
            -3.0,
            0.0,
            -8.0 * RELATIVE_DEVIANCE_SERIES_THRESHOLD,
            8.0 * RELATIVE_DEVIANCE_SERIES_THRESHOLD,
            2.0,
            -1.0,
            1.0,
        ]
    )
    actual = _relative_entropy_increment(base, delta)
    expected = np.array([_poisson_oracle(b + d, b) / 2.0 for b, d in zip(base, delta)])

    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-30)


def test_negative_deviance_rounding_boundary_is_inclusive():
    """Tiny negative deviance is clamped through the boundary, then rejected."""
    at_boundary = np.array([-DEVIANCE_ROUNDING_TOLERANCE])
    outside = np.array([np.nextafter(-DEVIANCE_ROUNDING_TOLERANCE, -np.inf)])

    np.testing.assert_array_equal(_check_deviance_rounding(at_boundary), [0.0])
    with pytest.raises(FloatingPointError, match="negative value"):
        _check_deviance_rounding(outside)


def test_materially_negative_deviance_is_rejected_at_an_absolute_scale():
    """A materially negative deviance raises whatever the rounding tolerance is."""
    # Literal magnitudes, so this holds whatever the tolerance constant becomes.
    assert DEVIANCE_ROUNDING_TOLERANCE < 1e-12

    np.testing.assert_array_equal(_check_deviance_rounding(np.array([-1e-16])), [0.0])
    with pytest.raises(FloatingPointError, match="negative value"):
        _check_deviance_rounding(np.array([-1e-6]))


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
    """Poisson deviance matches high precision near its mean."""
    actual = _poisson_deviance(
        np.array([x], dtype=np.float64), np.array([mu], dtype=np.float64)
    )[0]
    assert actual == pytest.approx(_poisson_oracle(x, mu), rel=2e-15, abs=1e-30)


@pytest.mark.parametrize("x", [0, 10**15, 10**15 + 1, 10**15 + 10, 2 * 10**15])
def test_binomial_deviance_matches_high_precision_near_mean_and_boundaries(x):
    """Binomial deviance matches high precision near means and boundaries."""
    n = 2 * 10**15
    mu = 10**15
    actual = _binomial_deviance(
        np.array([x], dtype=np.float64),
        np.array([n], dtype=np.float64),
        np.array([mu], dtype=np.float64),
    )[0]
    assert actual == pytest.approx(_binomial_oracle(x, n, mu), rel=2e-15, abs=1e-30)


@pytest.mark.parametrize("difference", [1, 10])
@pytest.mark.parametrize("alpha", [Decimal("1e-12"), Decimal("0.1")])
def test_scaled_nb_deviance_matches_high_precision_near_mean(difference, alpha):
    """Scaled-NB deviance matches high precision near its mean."""
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
    """Scaled-NB deviance matches high precision at count boundaries."""
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
        -np.nextafter(RELATIVE_DEVIANCE_SERIES_THRESHOLD, 0.0),
        -np.nextafter(RELATIVE_DEVIANCE_SERIES_THRESHOLD, 1.0),
    ],
)
def test_deviance_is_accurate_on_both_sides_of_series_switch(side):
    """Deviance stays accurate on both sides of the series threshold."""
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
        _poisson_oracle(Decimal.from_float(x), Decimal.from_float(mu)),
        rel=2e-14,
        abs=0.0,
    )
    assert nb == pytest.approx(
        _scaled_nb_oracle(
            Decimal.from_float(x), Decimal.from_float(mu), Decimal("1e-6")
        ),
        rel=2e-14,
        abs=0.0,
    )


def test_scaled_nb_small_positive_dispersion_approaches_poisson_without_switching():
    """Small positive NB dispersion approaches Poisson without branch changes."""
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
    assert nb == pytest.approx(_scaled_nb_oracle(x, mu, alpha), rel=3e-14, abs=0.0)
    assert nb == pytest.approx(poisson, rel=2e-6, abs=0.0)
