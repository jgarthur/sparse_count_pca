"""Tests for residual representations, operators, and PCA results."""

import numpy as np
import pytest
import zarr
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca import _residuals as residuals_module
from sparse_count_pca import residual_pca, residual_pca_matrix
from sparse_count_pca._counts import COUNT_INTEGER_ATOL, _canonicalize_counts
from sparse_count_pca._operator import SparseLowRankLinearOperator
from sparse_count_pca._representation import SparseLowRankMatrix
from sparse_count_pca._residuals import ALPHA_EPS
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
    """Implicit residual operators and SVDs match dense calculations."""
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

    np.testing.assert_allclose(result.operator @ z, dense @ z, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(result.operator.T @ y, dense.T @ y, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(result.operator @ Z, dense @ Z, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(result.operator.T @ Y, dense.T @ Y, rtol=0.0, atol=1e-10)

    _, singular_values, Vt = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(
        result.singular_values, singular_values[:2], rtol=1e-8, atol=0.0
    )
    assert _compare_subspaces(result.components.T, Vt[:2].T, rtol=0.0, atol=1e-7)
    expected_total_variance = np.sum(dense**2) / (counts.shape[0] - 1)
    assert result.total_variance == pytest.approx(
        expected_total_variance, rel=1e-10, abs=0.0
    )
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        result.explained_variance / expected_total_variance,
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_scaled_nb_probability_family_is_selected_once_per_gene(residual):
    """Scaled-NB probability-family selection is consistent within each gene."""
    counts = sparse.csr_matrix(
        [
            [999_990, 5, 5],
            [1, 4, 5],
            [2, 3, 5],
            [3, 2, 5],
        ]
    )
    alpha = np.array([0.9 * ALPHA_EPS, ALPHA_EPS, 1.1 * ALPHA_EPS])
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
    # The million-count row pushes r = 1 / alpha_tilde to 4e8, where production's
    # own conditioning costs ~6e-8. A wrong family moves these values by whole units.
    np.testing.assert_allclose(
        result.operator @ np.eye(counts.shape[1]), expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("residual", ["pearson", "deviance"])
def test_scaled_nb_genes_below_alpha_threshold_equal_poisson(residual):
    """Genes below the alpha threshold use the Poisson residual limit."""
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


@pytest.mark.parametrize("entry_point", ["matrix", "anndata"])
def test_scaled_nb_accepts_zero_dimensional_numpy_alpha(counts, entry_point):
    """Every entry point treats a zero-dimensional NumPy alpha as a scalar."""
    keywords = dict(model="scaled_nb", dtype="float64")
    if entry_point == "matrix":
        scalar = residual_pca_matrix(counts, n_comps=2, alpha=0.1, **keywords)
        zero_dimensional = residual_pca_matrix(
            counts, n_comps=2, alpha=np.array(0.1), **keywords
        )
        singular_values = zero_dimensional.singular_values
        recorded = zero_dimensional.params["alpha"]
    else:
        scalar = residual_pca(
            AnnData(counts.copy()), 2, alpha=0.1, copy=True, **keywords
        )
        zero_dimensional = residual_pca(
            AnnData(counts.copy()), 2, alpha=np.array(0.1), copy=True, **keywords
        )
        singular_values = zero_dimensional.uns["pca"]["singular_values"]
        scalar = scalar.uns["pca"]
        recorded = zero_dimensional.uns["pca"]["params"]["alpha"]

    expected = (
        scalar.singular_values if entry_point == "matrix" else scalar["singular_values"]
    )
    np.testing.assert_allclose(singular_values, expected, rtol=0.0, atol=1e-12)
    assert recorded == 0.1


def test_uncentered_operator_matches_dense(counts):
    """The uncentered implicit operator matches its dense representation."""
    result = residual_pca_matrix(
        counts, n_comps=2, dtype="float64", return_operator=True
    )
    centered = result.operator
    uncentered = SparseLowRankLinearOperator(
        SparseLowRankMatrix(centered.S, centered.left, centered.right),
        center=False,
        dtype="float64",
    )
    dense = _materialize_dense_residual(counts)
    np.testing.assert_allclose(
        uncentered @ np.eye(counts.shape[1]), dense, rtol=0.0, atol=1e-10
    )
    assert uncentered.frobenius_squared_uncentered() == pytest.approx(
        np.sum(dense**2), rel=1e-10, abs=0.0
    )


@pytest.mark.parametrize(("dtype", "rtol"), [("float64", 1e-12), ("float32", 1e-7)])
def test_stable_variance_sparse_support_matches_stored_operator(dtype, rtol):
    """Stable variance matches the represented operator on sparse support."""
    n_obs, n_vars = 101, 4
    rows = np.array([0, 3, 10, 25, 50, 75, 90, 100])
    cols = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    data = np.array([0.3, -0.7, 1.1, -0.2, 0.4, 0.8, -0.5, 0.9])
    S = sparse.csr_matrix((data, (rows, cols)), shape=(n_obs, n_vars))
    u = np.linspace(-2.0, -0.25, n_obs)
    v = np.array([0.4, 0.8, 1.2, 1.6])
    operator = SparseLowRankLinearOperator(
        SparseLowRankMatrix(S, u, v), center=True, dtype=dtype
    )

    stored_centered = (operator @ np.eye(n_vars, dtype=dtype)).astype(np.float64)
    expected = np.sum(stored_centered**2)
    assert operator.frobenius_squared_centered() == pytest.approx(
        expected, rel=rtol, abs=0.0
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_total_variance_high_count_near_constant_matches_stored_operator(dtype):
    """Variance remains accurate for high-count, nearly constant matrices."""
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
    assert result.total_variance == pytest.approx(expected, rel=1e-10, abs=0.0)
    np.testing.assert_allclose(
        result.explained_variance_ratio,
        result.explained_variance / expected,
        rtol=1e-10,
        atol=0.0,
    )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_identical_rows_raise_zero_centered_variance_before_arpack(dtype, monkeypatch):
    """Identical rows raise for zero variance before invoking ARPACK."""
    counts = np.tile([1, 2, 3, 4], (6, 1))

    def fail_if_called(*args, **kwargs):
        pytest.fail("ARPACK was called for a zero-variance matrix")

    monkeypatch.setattr("sparse_count_pca._pca.compute_truncated_svd", fail_if_called)
    with pytest.warns(UserWarning, match="Dense input was converted to CSR"):
        with pytest.raises(ValueError, match="zero centered variance"):
            residual_pca_matrix(counts, n_comps=2, dtype=dtype)


def test_exact_symmetric_clipping_matches_dense(counts):
    """Exact symmetric clipping matches a dense clipped residual matrix."""
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
        result.operator @ np.eye(counts.shape[1]),
        dense,
        rtol=0.0,
        atol=1e-10,
    )
    assert result.total_variance == pytest.approx(
        np.sum(dense**2) / (counts.shape[0] - 1), rel=1e-10, abs=0.0
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
@pytest.mark.parametrize(
    ("clip", "clip_mode"),
    [(None, "symmetric"), (0.5, "symmetric"), (0.5, "upper")],
    ids=["unclipped", "symmetric", "upper"],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_blocked_residual_build_matches_single_block_and_dense_oracle(
    counts,
    monkeypatch,
    model,
    residual,
    alpha,
    clip,
    clip_mode,
    dtype,
):
    """Every blocked residual family exactly matches one block and its oracle."""
    n = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    column_totals = np.asarray(counts.sum(axis=0, dtype=np.float64)).ravel()
    p = column_totals / column_totals.sum(dtype=np.float64)
    keywords = {
        "model": model,
        "residual": residual,
        "alpha": alpha,
        "clip": clip,
        "clip_mode": clip_mode,
        "clip_max_nnz_ratio": None,
        "dtype": dtype,
    }

    monkeypatch.setattr(
        residuals_module,
        "_RESIDUAL_SUPPORT_BLOCK_SIZE",
        counts.nnz + 1,
    )
    single = residuals_module.build_residual_representation(counts, n, p, **keywords)

    monkeypatch.setattr(residuals_module, "_RESIDUAL_SUPPORT_BLOCK_SIZE", 4)
    assert len(list(residuals_module._support_row_blocks(counts))) >= 3
    blocked = residuals_module.build_residual_representation(counts, n, p, **keywords)

    assert blocked.sparse.dtype == np.dtype(dtype)
    assert blocked.left.dtype == np.dtype(dtype)
    assert blocked.right.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(blocked.sparse.data, single.sparse.data)
    np.testing.assert_array_equal(blocked.sparse.indices, single.sparse.indices)
    np.testing.assert_array_equal(blocked.sparse.indptr, single.sparse.indptr)
    np.testing.assert_array_equal(blocked.left, single.left)
    np.testing.assert_array_equal(blocked.right, single.right)

    actual = blocked.sparse.toarray() + blocked.left @ blocked.right.T
    expected = _materialize_dense_residual(
        counts,
        model=model,
        residual=residual,
        alpha=alpha,
        clip=clip,
        clip_mode=clip_mode,
    )
    atol = 2e-6 if dtype == "float32" else (1e-10 if residual == "deviance" else 1e-12)
    np.testing.assert_allclose(
        actual.astype(np.float64),
        expected,
        rtol=0.0,
        atol=atol,
    )


def test_residual_builder_skips_zero_elimination_when_data_are_nonzero(
    counts, monkeypatch
):
    """A nonzero sparse correction avoids a redundant full-support prune pass."""

    def fail_if_called(self):
        pytest.fail("eliminate_zeros was called for entirely nonzero data")

    monkeypatch.setattr(sparse.csr_matrix, "eliminate_zeros", fail_if_called)

    transformed = scp.transform(counts, scp.Residual(), dtype="float32")

    assert transformed._sparse.nnz == counts.nnz


@pytest.mark.parametrize(("dtype", "rtol"), [("float64", 1e-12), ("float32", 1e-7)])
def test_clipped_total_variance_matches_stored_operator(counts, dtype, rtol):
    """Clipped total variance matches the stored implicit operator."""
    result = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=0.5,
        dtype=dtype,
        return_operator=True,
    )
    stored_centered = (result.operator @ np.eye(counts.shape[1], dtype=dtype)).astype(
        np.float64
    )
    expected = np.sum(stored_centered**2) / (counts.shape[0] - 1)
    assert result.total_variance == pytest.approx(expected, rel=rtol, abs=0.0)


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_symmetric_clipping_is_exact_for_every_residual(counts, model, residual, alpha):
    """Symmetric clipping is exact for Pearson and deviance residuals."""
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
        result.operator @ np.eye(counts.shape[1]),
        dense,
        rtol=0.0,
        atol=1e-10,
    )


def test_symmetric_clipping_support_growth_guard(counts):
    """Symmetric clipping enforces its sparse-support growth limit."""
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
        unlimited.operator @ np.eye(counts.shape[1]),
        exact,
        rtol=0.0,
        atol=1e-10,
    )

    with pytest.raises(RuntimeError, match="increase sparse support"):
        residual_pca_matrix(
            counts,
            n_comps=2,
            clip=clip,
            clip_max_nnz_ratio=1.0,
        )

    # A very large finite limit behaves like no limit rather than overflowing.
    large_finite = residual_pca_matrix(
        counts,
        n_comps=2,
        clip=clip,
        clip_max_nnz_ratio=1e308,
        dtype="float64",
        return_operator=True,
    )
    assert large_finite.operator.S.nnz == counts.nnz + 3
    np.testing.assert_allclose(
        large_finite.operator @ np.eye(counts.shape[1]),
        exact,
        rtol=0.0,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("model", "residual", "alpha"),
    [
        ("poisson", "pearson", None),
        ("binomial", "deviance", None),
        ("scaled_nb", "deviance", np.array([0.0, 0.1, 0.3, 1.0])),
    ],
)
def test_upper_clipping_is_exact_without_support_growth(counts, model, residual, alpha):
    """Upper clipping is exact without adding structural-zero support."""
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
        result.operator @ np.eye(counts.shape[1]),
        dense,
        rtol=0.0,
        atol=1e-10,
    )


def test_upper_clipping_leaves_negative_tail_unchanged(counts):
    """Upper clipping leaves the package's negative residual tail unchanged."""
    clip = 0.5
    unclipped = scp.transform(counts, scp.Residual()).materialize()
    clipped = scp.transform(
        counts, scp.Residual(clip=clip, clip_mode="upper")
    ).materialize()
    lower_tail = unclipped < -clip

    assert lower_tail.any()
    assert (unclipped > clip).any()

    # Unclipped takes the fast Pearson path and clipping the general one, so these
    # agree to rounding, not bit-for-bit. Symmetric clipping would move them to -clip.
    np.testing.assert_allclose(
        clipped[lower_tail], unclipped[lower_tail], rtol=1e-15, atol=0.0
    )
    assert np.max(clipped) <= clip
    np.testing.assert_allclose(
        clipped,
        _materialize_dense_residual(counts, clip=clip, clip_mode="upper"),
        rtol=0.0,
        atol=1e-12,
    )


def test_matrix_clipping_params_are_recorded(counts):
    """Matrix PCA results record all clipping parameters."""
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


@pytest.mark.parametrize(
    "requested", [None, "float32", "float64"], ids=["default", "float32", "float64"]
)
def test_dtype_contract(counts, requested):
    """Requested dtypes govern operator, score, and component storage."""
    keywords = {} if requested is None else {"dtype": requested}
    expected = np.dtype("float64" if requested is None else requested)

    result = residual_pca_matrix(counts, n_comps=2, return_operator=True, **keywords)

    assert result.scores.dtype == expected
    assert result.components.dtype == expected
    assert result.operator.dtype == expected
    assert (result.operator @ np.ones(counts.shape[1])).dtype == expected
    assert result.params["dtype"] == str(expected)
    assert not hasattr(result, "loadings")
    assert result.singular_values.dtype == np.float64
    assert result.explained_variance.dtype == np.float64

    if requested is None:
        direct = SparseLowRankLinearOperator(
            SparseLowRankMatrix(sparse.csr_matrix(np.eye(3)), np.ones(3), np.ones(3))
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
    """Operator dtype validation rejects unsupported scalar types."""
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        residual_pca_matrix(counts, n_comps=2, dtype=dtype)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        SparseLowRankLinearOperator(
            SparseLowRankMatrix(
                sparse.csr_matrix(np.eye(3)),
                np.ones(3),
                np.ones(3),
            ),
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
    """Residual PCA rejects invalid model and solver parameters."""
    error = (
        NotImplementedError
        if "solver" in kwargs or isinstance(kwargs.get("clip"), str)
        else ValueError
    )
    with pytest.raises(error, match=message):
        residual_pca_matrix(counts, n_comps=2, **kwargs)


def _with_first_stored_value(counts, value):
    """Return float64 counts with the first stored entry replaced."""
    modified = counts.astype(np.float64)
    modified.data[0] = value
    return modified


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (5.25, "non-integer"),
        (100_000.4, "non-integer"),
        (-1.0, "negative"),
    ],
    ids=["small-fractional", "large-fractional", "negative"],
)
def test_count_value_validation(counts, value, message):
    """Residual PCA rejects non-integer and negative stored count values."""
    with pytest.raises(ValueError, match=message):
        residual_pca_matrix(_with_first_stored_value(counts, value), n_comps=2)


def test_check_values_false_admits_non_integer_counts(counts):
    """Disabling the count-likeness check admits non-integer stored values."""
    residual_pca_matrix(
        _with_first_stored_value(counts, 5.25), n_comps=2, check_values=False
    )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_complex_counts_are_rejected_before_cast(counts, sparse_input):
    """Complex count inputs are rejected before any real-valued cast."""
    values = counts.astype(np.complex128)
    values.data[0] += 1j
    X = values if sparse_input else values.toarray()
    with pytest.raises(ValueError, match="real numeric"):
        residual_pca_matrix(X, n_comps=2)


def test_count_integer_tolerance_has_an_explicit_boundary(counts):
    """Float counts pass inside and fail outside the integer tolerance."""
    inside = counts.astype(np.float64)
    inside.data[0] += 0.5 * COUNT_INTEGER_ATOL
    _canonicalize_counts(inside, check_values=True)

    outside = counts.astype(np.float64)
    outside.data[0] += 2.0 * COUNT_INTEGER_ATOL
    with pytest.raises(ValueError, match="non-integer"):
        _canonicalize_counts(outside, check_values=True)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_sparse_counts_are_rejected(counts, value):
    """Sparse count inputs reject nonfinite stored values."""
    invalid = counts.astype(np.float64)
    invalid.data[0] = value
    with pytest.raises(ValueError, match="NaN or inf"):
        residual_pca_matrix(invalid, n_comps=2)


def test_finite_counts_with_overflowing_margins_are_rejected():
    """Finite values fail clearly when their float64 margins overflow."""
    counts = sparse.csr_matrix(np.full((3, 3), 1e308))

    with pytest.raises(ValueError, match="Count margins overflow"):
        residual_pca_matrix(counts, n_comps=2, check_values=False)


def test_nonnumeric_dense_counts_are_rejected():
    """Dense nonnumeric inputs are rejected as count matrices."""
    invalid = np.full((4, 3), "1")
    with pytest.raises(ValueError, match="real numeric"):
        residual_pca_matrix(invalid, n_comps=2)


def test_integer_counts_must_be_exactly_representable_as_float64(counts):
    """Integer counts must round-trip exactly through float64."""
    unsafe = counts.astype(np.uint64)
    unsafe.data[0] = 2**53 + 1
    with pytest.raises(ValueError, match="represented exactly as float64"):
        residual_pca_matrix(unsafe, n_comps=2)


def test_large_exactly_representable_integer_counts_remain_supported():
    """The bounded validation preserves exact integers above two to the 53."""
    exact = sparse.csr_matrix(np.array([[2**54]], dtype=np.uint64))

    canonical = _canonicalize_counts(exact, check_values=True)

    assert canonical is exact


def test_explicit_sparse_zeros_are_removed_without_mutating_input(counts):
    """Canonicalization removes explicit zeros without mutating its input."""
    with_zero = counts.copy()
    insertion = with_zero.indptr[1]
    with_zero.data = np.insert(with_zero.data, insertion, 0)
    with_zero.indices = np.insert(with_zero.indices, insertion, 2)
    with_zero.indptr[1:] += 1
    original_nnz = with_zero.nnz
    expected = residual_pca_matrix(counts, n_comps=2)
    with pytest.warns(UserWarning, match="copied.*explicitly stored zeros"):
        actual = residual_pca_matrix(with_zero, n_comps=2)
    assert with_zero.nnz == original_nnz
    np.testing.assert_allclose(
        actual.singular_values,
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )


def test_canonical_zero_free_csr_is_borrowed_without_copying(counts):
    """Canonical zero-free CSR input is reused without defensive copying."""
    assert counts.has_canonical_format
    assert not (counts.data == 0).any()

    canonical = _canonicalize_counts(counts, check_values=True)

    assert canonical is counts


def test_noncanonical_csr_copy_warns_and_preserves_input(counts):
    """Noncanonical CSR is copied with a warning and the input stays unchanged."""
    unsorted = counts.copy()
    start, stop = unsorted.indptr[:2]
    unsorted.indices[start:stop] = unsorted.indices[start:stop][::-1]
    unsorted.data[start:stop] = unsorted.data[start:stop][::-1]
    unsorted.has_sorted_indices = False
    unsorted.has_canonical_format = False
    original_indices = unsorted.indices.copy()
    original_data = unsorted.data.copy()

    with pytest.warns(UserWarning, match="duplicate or unsorted"):
        canonical = _canonicalize_counts(unsorted, check_values=True)

    assert canonical is not unsorted
    np.testing.assert_array_equal(unsorted.indices, original_indices)
    np.testing.assert_array_equal(unsorted.data, original_data)
    assert canonical.has_canonical_format


def test_dense_zarr_input_is_eagerly_converted(counts, tmp_path):
    """Dense Zarr inputs are eagerly converted to an in-memory CSR matrix."""
    dense = counts.toarray()
    X = zarr.open_array(
        str(tmp_path / "counts.zarr"),
        mode="w",
        shape=dense.shape,
        dtype=dense.dtype,
    )
    X[...] = dense
    expected = residual_pca_matrix(counts, n_comps=2, dtype="float64")

    with pytest.warns(UserWarning, match="Dense input was converted to CSR"):
        result = residual_pca_matrix(X, n_comps=2, dtype="float64")

    np.testing.assert_allclose(
        result.singular_values,
        expected.singular_values,
        rtol=0.0,
        atol=1e-12,
    )


def test_n_comps_is_not_clamped(counts):
    """Invalid component counts raise instead of being silently clamped."""
    with pytest.raises(ValueError, match="n_comps must satisfy"):
        residual_pca_matrix(counts, n_comps=counts.shape[1])
