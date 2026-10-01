"""Tests for bounded count validation and its error and ownership contracts."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import _counts as counts_module


@pytest.mark.parametrize("position", [1, 2, 5])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "NaN or inf"),
        (np.inf, "NaN or inf"),
        (-1.0, "negative"),
        (1.25, "non-integer"),
    ],
)
def test_count_checks_reject_values_across_blocks(
    monkeypatch, position, value, message
):
    """Invalid counts fail even at or beyond a validation-block boundary."""
    monkeypatch.setattr(counts_module, "_COUNT_CHECK_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.ones((3, 2)))
    counts.data[position] = value
    original = counts.data.copy()

    with pytest.raises(ValueError, match=message):
        counts_module._canonicalize_counts(counts, check_values=True)
    np.testing.assert_array_equal(counts.data, original)


@pytest.mark.parametrize(
    ("first", "last", "message"),
    [(-1.0, np.nan, "NaN or inf"), (1.25, -1.0, "negative")],
)
def test_blocked_count_checks_preserve_error_precedence(
    monkeypatch, first, last, message
):
    """Finite and nonnegative checks retain precedence across separate blocks."""
    monkeypatch.setattr(counts_module, "_COUNT_CHECK_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.ones((3, 2)))
    counts.data[0], counts.data[-1] = first, last

    with pytest.raises(ValueError, match=message):
        counts_module._canonicalize_counts(counts, check_values=True)


@pytest.mark.parametrize("value", [2**53, 2**53 + 1, 2**53 + 2])
def test_blocked_count_checks_preserve_integer_precision(monkeypatch, value):
    """Large integers are accepted exactly when float64 preserves their value."""
    monkeypatch.setattr(counts_module, "_COUNT_CHECK_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.ones((3, 2), dtype=np.uint64))
    counts.data[-1] = value

    if value == 2**53 + 1:
        with pytest.raises(ValueError, match="represented exactly as float64"):
            counts_module._canonicalize_counts(counts, check_values=True)
    else:
        assert counts_module._canonicalize_counts(counts, check_values=True) is counts


def test_blocked_count_checks_keep_fractional_escape_hatch(monkeypatch):
    """Disabling integrality admits fractional counts without copying their arrays."""
    monkeypatch.setattr(counts_module, "_COUNT_CHECK_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.full((3, 2), 1.25))
    counts.data.flags.writeable = False

    assert counts_module._canonicalize_counts(counts, check_values=False) is counts
