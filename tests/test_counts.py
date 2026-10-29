"""Tests for bounded count validation and its error and ownership contracts."""

import numpy as np
import pytest
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca import _counts as counts_module
from tests._oracles import _assert_pca_agreement


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
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", 2)
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
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.ones((3, 2)))
    counts.data[0], counts.data[-1] = first, last

    with pytest.raises(ValueError, match=message):
        counts_module._canonicalize_counts(counts, check_values=True)


@pytest.mark.parametrize("value", [2**53, 2**53 + 1, 2**53 + 2])
def test_blocked_count_checks_preserve_integer_precision(monkeypatch, value):
    """Large integers are accepted exactly when float64 preserves their value."""
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.ones((3, 2), dtype=np.uint64))
    counts.data[-1] = value

    if value == 2**53 + 1:
        with pytest.raises(ValueError, match="represented exactly as float64"):
            counts_module._canonicalize_counts(counts, check_values=True)
    else:
        assert counts_module._canonicalize_counts(counts, check_values=True) is counts


def test_blocked_count_checks_keep_fractional_escape_hatch(monkeypatch):
    """Disabling integrality admits fractional counts without copying their arrays."""
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(np.full((3, 2), 1.25))
    counts.data.flags.writeable = False

    assert counts_module._canonicalize_counts(counts, check_values=False) is counts


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csr_array])
@pytest.mark.parametrize("axis", [-2, -1, 0, 1])
@pytest.mark.parametrize("target", [2, 100_000])
@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64, np.uint64]
)
def test_count_margins_accumulate_before_output_casting(
    monkeypatch, matrix_type, axis, target, dtype
):
    """Both count margins accumulate small contributions in float64 across blocks."""
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", target)
    dense = np.array(
        [[100_000_000, 1, 0], [0, 100_000_000, 1], [1, 0, 100_000_000]], dtype=dtype
    )
    counts = matrix_type(dense)
    originals = [
        values.copy() for values in (counts.data, counts.indices, counts.indptr)
    ]
    for values in (counts.data, counts.indices, counts.indptr):
        values.flags.writeable = False

    actual = counts_module._sum_counts(counts, axis=axis)

    assert actual.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(actual, [100_000_001.0] * 3)
    for name, original in zip(("data", "indices", "indptr"), originals, strict=True):
        np.testing.assert_array_equal(getattr(counts, name), original)
        assert not getattr(counts, name).flags.writeable


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csr_array])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_count_margins_avoid_integer_accumulator_overflow(matrix_type, axis, dtype):
    """Valid large integer counts cannot overflow an integer margin accumulator."""
    counts = matrix_type(np.full((4, 4), 2**62, dtype=dtype))

    actual = counts_module._sum_counts(counts, axis=axis)

    np.testing.assert_array_equal(actual, np.full(4, float(2**64)))


@pytest.mark.parametrize("matrix_type", [sparse.csr_matrix, sparse.csr_array])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("target", [1, 2, 100_000])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (3, 4)])
def test_count_margins_handle_empty_support(
    monkeypatch, matrix_type, axis, target, shape
):
    """Empty sparse support produces correctly shaped zero margins on either axis."""
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", target)
    actual = counts_module._sum_counts(matrix_type(shape, dtype=np.float32), axis=axis)

    np.testing.assert_array_equal(actual, np.zeros(shape[1 - axis]))


@pytest.mark.parametrize("axis", [0, 1])
def test_count_margins_skip_empty_rows_and_keep_explicit_zeros(monkeypatch, axis):
    """Empty CSR rows and stored zeros contribute zero without changing the support."""
    monkeypatch.setattr(counts_module, "_COUNT_BLOCK_SIZE", 2)
    counts = sparse.csr_matrix(
        (
            np.array([100_000_000, 1, 0, 1], dtype=np.float32),
            [0, 1, 2, 0],
            [0, 0, 3, 3, 4, 4],
        ),
        shape=(5, 3),
    )
    expected = counts.astype(np.float64).toarray().sum(axis=axis)

    np.testing.assert_array_equal(
        counts_module._sum_counts(counts, axis=axis), expected
    )
    assert counts.nnz == 4


@pytest.mark.parametrize("axis", [0, 1])
def test_count_margins_preserve_float64_overflow_detection(axis):
    """A genuinely overflowing float64 margin remains nonfinite for caller checks."""
    counts = sparse.csr_matrix(np.full((2, 2), np.finfo(np.float64).max))

    assert np.isinf(counts_module._sum_counts(counts, axis=axis)).all()


@pytest.mark.parametrize("axis", [0, 1])
def test_count_margins_allow_totals_exceeding_float32_range(axis):
    """Float32 count totals stay finite when they fit the float64 accumulator."""
    maximum = np.finfo(np.float32).max
    counts = sparse.csr_matrix(np.full((2, 2), maximum, dtype=np.float32))

    actual = counts_module._sum_counts(counts, axis=axis)

    np.testing.assert_array_equal(actual, np.full(2, float(maximum) * 2))
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    "method",
    [
        scp.Log1pNormalized(),
        scp.ShiftedCLR(count_shift=0.7),
        scp.ProportionShiftedCLR(composition_shift=0.3),
        scp.DirichletLog(concentration=2.5),
        scp.DirichletCLR(concentration=2.5),
        scp.Residual(clip=None),
        scp.Residual(model="binomial", clip=None),
        scp.Residual(model="scaled_nb", alpha=0.2, clip=None),
        scp.Residual(residual="deviance", clip=None),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float32_counts_fit_the_same_transform_as_float64_counts(method, dtype):
    """Float32 input counts fit the same factors and PCA as their exact float64 copy."""
    counts = sparse.csr_matrix(
        np.array(
            [
                [100_000_000, 1, 2, 0],
                [0, 100_000_000, 1, 2],
                [2, 0, 100_000_000, 1],
                [1, 2, 0, 100_000_000],
                [3, 1, 4, 2],
            ],
            dtype=np.float32,
        )
    )
    actual = scp.transform(counts, method, dtype=dtype)
    expected = scp.transform(counts.astype(np.float64), method, dtype=dtype)

    np.testing.assert_array_equal(actual._sparse.data, expected._sparse.data)
    np.testing.assert_array_equal(actual._left, expected._left)
    np.testing.assert_array_equal(actual._right, expected._right)
    actual_pca = actual.pca(n_comps=2)
    expected_pca = expected.pca(n_comps=2)
    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(actual_pca.mean, expected_pca.mean, rtol=tolerance)
    _assert_pca_agreement(
        actual_pca,
        expected_pca.singular_values,
        expected_pca.scores,
        expected_pca.components,
        rtol=tolerance,
    )
    assert actual_pca.total_variance == pytest.approx(
        expected_pca.total_variance, rel=tolerance
    )
    np.testing.assert_allclose(
        actual_pca.explained_variance_ratio,
        expected_pca.explained_variance_ratio,
        rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float32_counts_fit_the_same_correspondence_analysis_as_float64_counts(dtype):
    """Float32 count margins preserve float64-fitted correspondence masses and axes."""
    counts = sparse.csr_matrix(
        np.array(
            [[100_000_000, 1, 2], [2, 100_000_000, 1], [1, 2, 100_000_000], [1, 2, 3]],
            dtype=np.float32,
        )
    )
    actual = scp.correspondence_analysis_matrix(counts, n_comps=2, dtype=dtype)
    expected = scp.correspondence_analysis_matrix(
        counts.astype(np.float64), n_comps=2, dtype=dtype
    )

    np.testing.assert_array_equal(actual.row_masses, expected.row_masses)
    np.testing.assert_array_equal(actual.column_masses, expected.column_masses)
    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        actual.singular_values, expected.singular_values, rtol=tolerance, atol=0.0
    )
    assert actual.total_inertia == pytest.approx(expected.total_inertia, rel=tolerance)
