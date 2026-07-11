import numpy as np
import pytest
import zarr
from scipy import sparse

from sparse_residual_pca import residual_pca_matrix
from sparse_residual_pca._operator import ResidualLinearOperator
from sparse_residual_pca._residuals import ALPHA_EPS
from tests._oracles import _compare_subspaces, _materialize_dense_residual


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("poisson", "deviance", None),
        ("binomial", "pearson", None),
        ("binomial", "deviance", None),
        ("scaled_nb", "pearson", np.array([0.0, 0.1, 0.3, 1.0])),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_operator_and_svd_match_dense(counts, model, residual, alpha):
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=alpha,
        dtype="float64",
        return_operator=True,
    )
    dense = _materialize_dense_residual(
        counts, model=model, residual=residual, alpha=alpha, center=True
    )
    rng = np.random.default_rng(3)
    z = rng.normal(size=counts.shape[1])
    y = rng.normal(size=counts.shape[0])
    Z = rng.normal(size=(counts.shape[1], 3))
    Y = rng.normal(size=(counts.shape[0], 3))

    np.testing.assert_allclose(result.operator @ z, dense @ z, atol=1e-10)
    np.testing.assert_allclose(result.operator.T @ y, dense.T @ y, atol=1e-10)
    np.testing.assert_allclose(result.operator @ Z, dense @ Z, atol=1e-10)
    np.testing.assert_allclose(result.operator.T @ Y, dense.T @ Y, atol=1e-10)

    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(result.singular_values, singular_values[:2], rtol=1e-8)
    assert _compare_subspaces(result.components.T, Vt[:2].T, atol=1e-7)
    expected_total_variance = np.sum(dense**2) / (counts.shape[0] - 1)
    assert result.total_variance == pytest.approx(expected_total_variance, rel=1e-10)
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        result.explained_variance / expected_total_variance,
    )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_scaled_nb_probability_family_is_selected_once_per_gene(residual):
    counts = sparse.csr_matrix(
        [
            [999_990, 5, 5],
            [1, 4, 5],
            [2, 3, 5],
            [3, 2, 5],
        ]
    )
    alpha = np.array([0.9 * ALPHA_EPS, 1.1 * ALPHA_EPS, 0.1])
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model="scaled_nb",
        residual=residual,
        alpha=alpha,
        dtype="float64",
        return_operator=True,
    )
    expected = _materialize_dense_residual(
        counts,
        model="scaled_nb",
        residual=residual,
        alpha=alpha,
        center=True,
    )
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), expected, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_scaled_nb_genes_below_alpha_threshold_equal_poisson(residual):
    counts = sparse.csr_matrix(
        [
            [999_990, 5, 5],
            [1, 4, 5],
            [2, 3, 5],
            [3, 2, 5],
        ]
    )
    scaled_nb = residual_pca_matrix(
        counts,
        n_comps=2,
        model="scaled_nb",
        residual=residual,
        alpha=np.full(counts.shape[1], 0.9 * ALPHA_EPS),
        dtype="float64",
        return_operator=True,
    )
    poisson = residual_pca_matrix(
        counts,
        n_comps=2,
        model="poisson",
        residual=residual,
        dtype="float64",
        return_operator=True,
    )
    identity = np.eye(counts.shape[1])
    np.testing.assert_allclose(
        scaled_nb.operator @ identity,
        poisson.operator @ identity,
        rtol=1e-13,
        atol=1e-13,
    )


def test_scaled_nb_accepts_zero_dimensional_numpy_alpha(counts):
    scalar = residual_pca_matrix(
        counts,
        n_comps=2,
        model="scaled_nb",
        alpha=0.1,
        dtype="float64",
    )
    zero_dimensional = residual_pca_matrix(
        counts,
        n_comps=2,
        model="scaled_nb",
        alpha=np.array(0.1),
        dtype="float64",
    )
    np.testing.assert_allclose(
        zero_dimensional.singular_values, scalar.singular_values
    )


def test_uncentered_operator_matches_dense(counts):
    result = residual_pca_matrix(
        counts, n_comps=2, dtype="float64", return_operator=True
    )
    centered = result.operator
    uncentered = ResidualLinearOperator(
        centered.S, centered.u, centered.v, center=False, dtype="float64"
    )
    dense = _materialize_dense_residual(counts)
    np.testing.assert_allclose(uncentered @ np.eye(counts.shape[1]), dense, atol=1e-10)
    assert uncentered.frobenius_squared_uncentered() == pytest.approx(
        np.sum(dense**2), rel=1e-10
    )


@pytest.mark.parametrize(
    ("dtype", "rtol"), [("float64", 1e-12), ("float32", 1e-7)]
)
def test_stable_variance_sparse_support_matches_stored_operator(dtype, rtol):
    n_obs, n_vars = 101, 4
    rows = np.array([0, 3, 10, 25, 50, 75, 90, 100])
    cols = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    data = np.array([0.3, -0.7, 1.1, -0.2, 0.4, 0.8, -0.5, 0.9])
    S = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_vars))
    u = np.linspace(-2.0, -0.25, n_obs)
    v = np.array([0.4, 0.8, 1.2, 1.6])
    operator = ResidualLinearOperator(S, u, v, center=True, dtype=dtype)

    stored_centered = (
        operator @ np.eye(n_vars, dtype=dtype)
    ).astype(np.float64)
    expected = np.sum(stored_centered**2)
    assert operator.frobenius_squared_centered() == pytest.approx(
        expected, rel=rtol
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_total_variance_high_count_near_constant_matches_stored_operator(dtype):
    baseline = 10**6
    counts = np.tile([baseline, 2 * baseline, 3 * baseline, 4 * baseline], (1000, 1))
    counts[0, 0] += 1
    counts[0, 1] -= 1

    with pytest.warns(UserWarning, match="Dense input was converted to CSR"):
        result = residual_pca_matrix(
            counts,
            n_comps=2,
            dtype=dtype,
            return_operator=True,
        )

    identity = np.eye(counts.shape[1], dtype=dtype)
    stored_centered = (result.operator @ identity).astype(np.float64)
    expected = np.sum(stored_centered**2) / (counts.shape[0] - 1)
    assert expected > 0
    assert result.total_variance == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        result.explained_variance / expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_identical_rows_raise_zero_centered_variance_before_arpack(dtype, monkeypatch):
    counts = np.tile([1, 2, 3, 4], (6, 1))

    def fail_if_called(*args, **kwargs):
        pytest.fail("ARPACK was called for a zero-variance matrix")

    monkeypatch.setattr(
        "sparse_residual_pca._matrix.compute_truncated_svd", fail_if_called
    )
    with pytest.warns(UserWarning, match="Dense input was converted to CSR"):
        with pytest.raises(ValueError, match="zero centered variance"):
            residual_pca_matrix(counts, n_comps=2, dtype=dtype)


def test_exact_symmetric_clipping_matches_dense(counts):
    # For Poisson Pearson residuals, the five zero entries have magnitudes
    # [1.308, 1.308, 1.398, 1.398, 1.461]. Thus clip=1.35 clips exactly three.
    clip = 1.35
    n_clipped_zeros = 3
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=clip,
        dtype="float64",
        return_operator=True,
    )
    dense = _materialize_dense_residual(counts, clip=clip, center=True)
    assert result.operator.S.nnz == counts.nnz + n_clipped_zeros
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-10
    )
    assert result.total_variance == pytest.approx(
        np.sum(dense**2) / (counts.shape[0] - 1), rel=1e-10
    )


@pytest.mark.parametrize(
    ("dtype", "rtol"), [("float64", 1e-12), ("float32", 1e-7)]
)
def test_clipped_total_variance_matches_stored_operator(counts, dtype, rtol):
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=0.5,
        dtype=dtype,
        return_operator=True,
    )
    stored_centered = (
        result.operator @ np.eye(counts.shape[1], dtype=dtype)
    ).astype(np.float64)
    expected = np.sum(stored_centered**2) / (counts.shape[0] - 1)
    assert result.total_variance == pytest.approx(expected, rel=rtol)


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("poisson", "deviance", None),
        ("binomial", "pearson", None),
        ("binomial", "deviance", None),
        ("scaled_nb", "pearson", np.array([0.0, 0.1, 0.3, 1.0])),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_symmetric_clipping_is_exact_for_every_residual(counts, model, residual, alpha):
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=0.5,
        dtype="float64",
        return_operator=True,
    )
    dense = _materialize_dense_residual(
        counts,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=0.5,
        center=True,
    )
    # This fixture has five structural zeros, all of which cross the threshold.
    assert result.operator.S.nnz == counts.nnz + 5
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-10
    )


def test_symmetric_clipping_support_growth_guard(counts):
    clip = 1.35
    exact_growth_ratio = (counts.nnz + 3) / counts.nnz

    with pytest.raises(RuntimeError, match="increase sparse support"):
        residual_pca_matrix(
            counts,
            n_comps=2,
            clip=clip,
            clip_max_nnz_ratio=exact_growth_ratio,
        )

    allowed = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=clip,
        clip_max_nnz_ratio=exact_growth_ratio + 0.01,
        dtype="float64",
        return_operator=True,
    )
    assert allowed.operator.S.nnz == counts.nnz + 3

    unlimited = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=clip,
        clip_max_nnz_ratio=None,
        dtype="float64",
        return_operator=True,
    )
    exact = _materialize_dense_residual(counts, clip=clip, center=True)
    assert unlimited.operator.S.nnz == counts.nnz + 3
    np.testing.assert_allclose(
        unlimited.operator @ np.eye(counts.shape[1]), exact, atol=1e-10
    )

    with pytest.raises(RuntimeError, match="increase sparse support"):
        residual_pca_matrix(
            counts,
            n_comps=2,
            clip=clip,
            clip_max_nnz_ratio=1.0,
        )


def test_symmetric_clipping_large_finite_growth_guard(counts):
    clip = 1.35
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=clip,
        clip_max_nnz_ratio=1e308,
        dtype="float64",
        return_operator=True,
    )
    dense = _materialize_dense_residual(counts, clip=clip, center=True)
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-10
    )


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("poisson", "deviance", None),
        ("binomial", "pearson", None),
        ("binomial", "deviance", None),
        ("scaled_nb", "pearson", np.array([0.0, 0.1, 0.3, 1.0])),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_upper_clipping_is_exact_without_support_growth(counts, model, residual, alpha):
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=0.5,
        clip_mode="upper",
        clip_max_nnz_ratio=1.0,
        dtype="float64",
        return_operator=True,
    )
    dense = _materialize_dense_residual(
        counts,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=0.5,
        clip_mode="upper",
        center=True,
    )
    sparse_support = set(zip(*result.operator.S.nonzero()))
    count_support = set(zip(*counts.nonzero()))
    assert sparse_support <= count_support
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), dense, atol=1e-10
    )


def test_upper_clipping_leaves_negative_tail_unchanged(counts):
    clip = 0.5
    unclipped = _materialize_dense_residual(counts)
    expected = _materialize_dense_residual(
        counts,
        clip=clip,
        clip_mode="upper",
    )
    assert np.min(unclipped) < -clip
    np.testing.assert_array_equal(
        expected[unclipped < -clip],
        unclipped[unclipped < -clip],
    )
    assert np.max(expected) <= clip


def test_matrix_clipping_params_are_recorded(counts):
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=1.0,
        clip_mode="upper",
        clip_max_nnz_ratio=None,
    )
    assert result.params["clip"] == 1.0
    assert result.params["clip_mode"] == "upper"
    assert result.params["clip_max_nnz_ratio"] is None


def test_dtype_contract(counts):
    result = residual_pca_matrix(
        counts, n_comps=2, dtype="float32", return_operator=True
    )
    assert result.scores.dtype == np.float32
    assert result.components.dtype == np.float32
    assert result.operator.dtype == np.dtype("float32")
    assert (result.operator @ np.ones(counts.shape[1])).dtype == np.float32
    assert result.singular_values.dtype == np.float64
    assert result.explained_variance.dtype == np.float64


def test_default_operator_dtype_is_float64(counts):
    result = residual_pca_matrix(counts, n_comps=2, return_operator=True)
    assert result.scores.dtype == np.float64
    assert result.components.dtype == np.float64
    assert result.operator.dtype == np.dtype("float64")
    assert result.params["dtype"] == "float64"
    direct = ResidualLinearOperator(
        sparse.csr_matrix(np.eye(3)), np.ones(3), np.ones(3)
    )
    assert direct.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.int64,
        np.complex128,
        pytest.param(
            getattr(np, "float128", None),
            marks=pytest.mark.skipif(
                not hasattr(np, "float128"), reason="float128 is unavailable"
            ),
        ),
    ],
)
def test_operator_dtype_rejects_types_other_than_float32_and_float64(counts, dtype):
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        residual_pca_matrix(counts, n_comps=2, dtype=dtype)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        ResidualLinearOperator(
            sparse.csr_matrix(np.eye(3)),
            np.ones(3),
            np.ones(3),
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "scaled_nb"}, "alpha is required"),
        ({"model": "poisson", "alpha": 0.1}, "alpha is only used"),
        ({"model": "scaled_nb", "alpha": -0.1}, "alpha must be nonnegative"),
        ({"clip": 0}, "clip must be finite and positive"),
        ({"clip": np.inf}, "clip must be finite and positive"),
        ({"clip": np.nan}, "clip must be finite and positive"),
        ({"clip": "sqrt_n_obs"}, "String clip aliases"),
        ({"clip_mode": "lower"}, "clip_mode must be one of"),
        ({"clip_max_nnz_ratio": 0.99}, "must be finite and at least 1"),
        ({"clip_max_nnz_ratio": np.inf}, "must be finite and at least 1"),
        ({"clip_max_nnz_ratio": np.nan}, "must be finite and at least 1"),
        ({"solver": "lobpcg"}, "Only solver='arpack'"),
    ],
)
def test_parameter_validation(counts, kwargs, message):
    error = (
        NotImplementedError
        if "solver" in kwargs or isinstance(kwargs.get("clip"), str)
        else ValueError
    )
    with pytest.raises(error, match=message):
        residual_pca_matrix(counts, n_comps=2, **kwargs)


def test_count_validation(counts):
    noninteger = counts.astype(float)
    noninteger.data[0] += 0.25
    with pytest.raises(ValueError, match="non-integer"):
        residual_pca_matrix(noninteger, n_comps=2)
    residual_pca_matrix(noninteger, n_comps=2, check_values=False)

    negative = counts.copy()
    negative.data[0] = -1
    with pytest.raises(ValueError, match="negative"):
        residual_pca_matrix(negative, n_comps=2)

    zero_cell = sparse.vstack([counts, sparse.csr_matrix((1, counts.shape[1]))])
    with pytest.raises(ValueError, match="zero total"):
        residual_pca_matrix(zero_cell, n_comps=2)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_complex_counts_are_rejected_before_cast(counts, sparse_input):
    values = counts.astype(np.complex128)
    values.data[0] += 1j
    X = values if sparse_input else values.toarray()
    with pytest.raises(ValueError, match="real numeric"):
        residual_pca_matrix(X, n_comps=2)


def test_large_fractional_counts_are_rejected(counts):
    fractional = counts.astype(np.float64)
    fractional.data[0] = 100_000.4
    with pytest.raises(ValueError, match="non-integer"):
        residual_pca_matrix(fractional, n_comps=2)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_sparse_counts_are_rejected(counts, value):
    invalid = counts.astype(np.float64)
    invalid.data[0] = value
    with pytest.raises(ValueError, match="NaN or inf"):
        residual_pca_matrix(invalid, n_comps=2)


def test_nonnumeric_dense_counts_are_rejected():
    invalid = np.full((4, 3), "1")
    with pytest.raises(ValueError, match="real numeric"):
        residual_pca_matrix(invalid, n_comps=2)


def test_integer_counts_must_be_exactly_representable_as_float64(counts):
    unsafe = counts.astype(np.uint64)
    unsafe.data[0] = 2**53 + 1
    with pytest.raises(ValueError, match="represented exactly as float64"):
        residual_pca_matrix(unsafe, n_comps=2)


def test_explicit_sparse_zeros_are_removed_without_mutating_input(counts):
    with_zero = counts.copy()
    insertion = with_zero.indptr[1]
    with_zero.data = np.insert(with_zero.data, insertion, 0)
    with_zero.indices = np.insert(with_zero.indices, insertion, 2)
    with_zero.indptr[1:] += 1
    original_nnz = with_zero.nnz
    expected = residual_pca_matrix(counts, n_comps=2)
    actual = residual_pca_matrix(with_zero, n_comps=2)
    assert with_zero.nnz == original_nnz
    np.testing.assert_allclose(actual.singular_values, expected.singular_values)


def test_dense_zarr_input_is_eagerly_converted(counts, tmp_path):
    X = zarr.create_array(tmp_path / "counts.zarr", data=counts.toarray())
    expected = residual_pca_matrix(counts, n_comps=2, dtype="float64")

    with pytest.warns(UserWarning, match="Dense input was converted to CSR"):
        result = residual_pca_matrix(X, n_comps=2, dtype="float64")

    np.testing.assert_allclose(result.singular_values, expected.singular_values)


def test_n_comps_is_not_clamped(counts):
    with pytest.raises(ValueError, match="n_comps must satisfy"):
        residual_pca_matrix(counts, n_comps=counts.shape[1])
