"""Tests for per-variable means and variances of fitted transforms."""

import math

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import sparse_count_pca as scp
from sparse_count_pca import _operator as operator_module
from sparse_count_pca._representation import SparseLowRankMatrix
from tests._oracles import (
    _dense_count_shifted_clr,
    _dense_dirichlet,
    _dense_log1p_norm,
    _dense_proportion_shifted_clr,
    _materialize_dense_residual,
)

# Matches the float64 operator-statistics tests. Per-column errors from the
# support-replacement identities are of order 1e-15 relative on these small
# well-conditioned inputs, while a wrong denominator or center errs by at
# least 1 / n_obs.
RTOL = 1e-12


@pytest.fixture
def counts_with_empty_gene(counts):
    """Return the shared counts with an appended gene that has no counts."""
    return sparse.hstack(
        [counts, sparse.csr_matrix((counts.shape[0], 1))], format="csr"
    )


SIZE_FACTORS = np.arange(1.0, 7.0)


def _residual_oracle(X, **kwargs):
    """Return dense residuals, zero for genes without counts.

    The specification defines those residuals as zero by continuity; the dense
    Pearson formula evaluates them as 0 / 0.
    """
    with np.errstate(invalid="ignore"):
        dense = _materialize_dense_residual(X, **kwargs)
    dense[:, np.asarray(X.sum(axis=0)).ravel() == 0] = 0.0
    return dense


TRANSFORMS = [
    pytest.param(
        scp.Residual(clip=None),
        lambda X: _residual_oracle(X),
        id="poisson-pearson",
    ),
    pytest.param(
        scp.Residual(clip=0.5, clip_max_nnz_ratio=None),
        lambda X: _residual_oracle(X, clip=0.5),
        id="poisson-pearson-symmetric-clip",
    ),
    pytest.param(
        scp.Residual(clip=0.5, clip_mode="upper"),
        lambda X: _residual_oracle(X, clip=0.5, clip_mode="upper"),
        id="poisson-pearson-upper-clip",
    ),
    pytest.param(
        scp.Residual(model="binomial", residual="deviance", clip=None),
        lambda X: _residual_oracle(X, model="binomial", residual="deviance"),
        id="binomial-deviance",
    ),
    pytest.param(
        scp.Residual(model="scaled_nb", alpha=0.3, clip=None),
        lambda X: _residual_oracle(X, model="scaled_nb", alpha=0.3),
        id="scaled-nb-pearson",
    ),
    pytest.param(
        scp.Log1pNormalized(size_factors=SIZE_FACTORS),
        lambda X: _dense_log1p_norm(X, SIZE_FACTORS),
        id="log1p-normalized",
    ),
    pytest.param(
        scp.ShiftedCLR(count_shift=0.7),
        lambda X: _dense_count_shifted_clr(X, 0.7),
        id="shifted-clr",
    ),
    pytest.param(
        scp.ProportionShiftedCLR(composition_shift=0.05),
        lambda X: _dense_proportion_shifted_clr(X, 0.05),
        id="proportion-shifted-clr",
    ),
    pytest.param(
        scp.DirichletLog(concentration=2.0),
        lambda X: _dense_dirichlet(X, 2.0, None, clr=False),
        id="dirichlet-log",
    ),
    pytest.param(
        scp.DirichletCLR(concentration=2.0),
        lambda X: _dense_dirichlet(X, 2.0, None, clr=True),
        id="dirichlet-clr",
    ),
]


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("block_nnz", [None, 3], ids=["one-block", "row-blocks"])
@pytest.mark.parametrize(("method", "dense_oracle"), TRANSFORMS)
def test_statistics_match_dense_oracle_per_variable(
    counts_with_empty_gene, method, dense_oracle, block_nnz, ddof, monkeypatch
):
    """Every variable's mean and variance match the dense transform.

    The input includes a gene with no counts, and the small block target splits
    the statistics passes into several row blocks.
    """
    if block_nnz is not None:
        monkeypatch.setattr(operator_module, "_STATS_MEAN_BLOCK_NNZ", block_nnz)
        monkeypatch.setattr(operator_module, "_STATS_NORM_BLOCK_NNZ", block_nnz)
    dense = dense_oracle(counts_with_empty_gene)

    stats = scp.transform(counts_with_empty_gene, method).variable_statistics(ddof=ddof)

    np.testing.assert_allclose(
        stats.mean, dense.mean(axis=0), rtol=RTOL, atol=RTOL * np.abs(dense).max()
    )
    np.testing.assert_allclose(
        stats.variance,
        dense.var(axis=0, ddof=ddof),
        rtol=RTOL,
        atol=RTOL * dense.var(axis=0).max(),
    )
    assert stats.mean.dtype == stats.variance.dtype == np.float64
    assert stats.n_obs == counts_with_empty_gene.shape[0]
    assert stats.ddof == ddof
    assert stats.var_names is None


def test_empty_gene_statistics_follow_the_transform(counts_with_empty_gene):
    """A gene without counts has zero residual variance but varies under CLR.

    Residuals of a zero-count gene are zero by continuity, while CLR coordinates
    subtract each cell's own log mean, so the empty gene inherits that
    variation.
    """
    residual = scp.transform(
        counts_with_empty_gene, scp.Residual(clip=None)
    ).variable_statistics()
    clr = scp.transform(
        counts_with_empty_gene, scp.ShiftedCLR(count_shift=0.7)
    ).variable_statistics()

    assert residual.mean[-1] == 0.0
    assert residual.variance[-1] == 0.0
    expected = _dense_count_shifted_clr(counts_with_empty_gene, 0.7)[:, -1].var(ddof=1)
    assert clr.variance[-1] == pytest.approx(expected, rel=RTOL, abs=0.0)
    assert clr.variance[-1] > 0.0


def _exact_column_statistics(column):
    """Return the mean and squared deviation sum of exact float64 values."""
    mean = math.fsum(column) / column.size
    return mean, math.fsum((column - mean) ** 2)


def _concentrated_baseline_column(n_obs):
    """One stored entry cancels a baseline of 1e13, leaving the value 1."""
    left = np.zeros(n_obs)
    left[0] = 1e13
    stored = np.zeros(n_obs)
    stored[0] = -1e13 + 1.0
    return left, stored


def _single_entry_large_baseline_column(n_obs):
    """One stored entry cancels a baseline of 1e16, leaving the value 2e12."""
    left = np.zeros(n_obs)
    left[0] = 1e16
    stored = np.zeros(n_obs)
    stored[0] = -1e16 + 2e12
    return left, stored


@pytest.mark.parametrize(
    "adversarial_column",
    [_concentrated_baseline_column, _single_entry_large_baseline_column],
    ids=["baseline-1e13", "baseline-1e16"],
)
def test_cancelling_column_stays_accurate_beside_a_dominant_column(
    adversarial_column,
):
    """A column whose support cancels its baseline keeps its own accuracy.

    A second, high-variance column dominates the total, so only a per-column
    comparison can detect an error in the cancelling column. Each represented
    entry is exact in float64, so the exact sums are the reference.
    """
    n_obs = 100
    rng = np.random.default_rng(11)
    adversarial_left, adversarial_stored = adversarial_column(n_obs)
    dominant = 1e7 * rng.normal(size=n_obs)
    left = np.column_stack([adversarial_left, dominant])
    stored = np.column_stack([adversarial_stored, np.zeros(n_obs)])
    representation = SparseLowRankMatrix(sparse.csr_matrix(stored), left, np.eye(2))
    transformed = scp.TransformedMatrix(
        representation, transform_label="test", params={}, check_values=False
    )
    represented = stored + left

    stats = transformed.variable_statistics()

    for column in range(2):
        mean, squared = _exact_column_statistics(represented[:, column])
        assert stats.mean[column] == pytest.approx(mean, rel=RTOL, abs=0.0)
        assert stats.variance[column] == pytest.approx(
            squared / (n_obs - 1), rel=RTOL, abs=0.0
        )


def test_nearly_constant_dense_gene_variance_stays_accurate():
    """A gene with large, nearly equal counts keeps its tiny variance.

    Its log1p values agree to about seven significant digits, so the variance
    is about 1e-13 of the squared mean. The reference is the exact variance of
    the stored values.
    """
    gene = np.array([1e6, 1e6, 1e6 + 1, 1e6 + 2, 1e6, 1e6 + 1, 1e6, 1e6 + 3])
    other = np.array([1, 0, 4, 2, 0, 3, 1, 5])
    counts = sparse.csr_matrix(np.column_stack([gene, other]))
    transformed = scp.transform(
        counts, scp.Log1pNormalized(size_factors=np.ones(gene.size))
    )
    values = transformed.materialize()[:, 0]
    mean, squared = _exact_column_statistics(values)

    stats = transformed.variable_statistics()

    assert squared / (gene.size - 1) < 1e-11 * mean**2
    assert stats.mean[0] == pytest.approx(mean, rel=RTOL, abs=0.0)
    assert stats.variance[0] == pytest.approx(
        squared / (gene.size - 1), rel=RTOL, abs=0.0
    )


def test_float32_statistics_are_float64_and_match_stored_values(counts):
    """A float32 transform returns float64 statistics of its stored values.

    The tolerance covers float32 rounding of reconstructed entries (machine
    epsilon about 1.2e-7) and still detects a wrong denominator or center,
    which errs by at least 1 / n_obs.
    """
    transformed = scp.transform(counts, scp.Residual(clip=None), dtype="float32")
    values = transformed.materialize().astype(np.float64)

    stats = transformed.variable_statistics()

    assert stats.mean.dtype == stats.variance.dtype == np.float64
    np.testing.assert_allclose(stats.mean, values.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(stats.variance, values.var(axis=0, ddof=1), rtol=1e-5)


@pytest.mark.parametrize(
    ("selector", "positions"),
    [
        ([3, 0, 3], [3, 0, 3]),
        ([True, False, True, True], [0, 2, 3]),
        (slice(1, 3), [1, 2]),
        (2, [2]),
        (-1, [3]),
        (["d", "a"], [3, 0]),
        ("b", [1]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
    ],
    ids=[
        "repeats",
        "boolean",
        "slice",
        "integer",
        "negative",
        "names",
        "name",
        "every-variable",
    ],
)
def test_selection_preserves_order_without_refitting(adata, selector, positions):
    """Selected statistics equal the full statistics at the selected positions.

    Selection keeps the requested order and repeats, and normalization stays
    fitted on every gene.
    """
    transformed = scp.transform(adata, scp.Residual(clip=None))
    full = transformed.variable_statistics()

    selected = transformed.variable_statistics(var=selector)

    np.testing.assert_array_equal(selected.mean, full.mean[positions])
    np.testing.assert_array_equal(selected.variance, full.variance[positions])
    assert selected.var_names.tolist() == adata.var_names[positions].tolist()


@pytest.mark.parametrize("source", ["matrix", "anndata"])
@pytest.mark.parametrize(
    "selector",
    [True, np.array([], dtype=bool), [True, False], [[0, 1]]],
    ids=["scalar-boolean", "empty-mask", "short-mask", "two-dimensional"],
)
def test_selectors_outside_the_contract_raise(counts, adata, source, selector):
    """Selectors that materialize rejects raise instead of reshaping results."""
    transformed = scp.transform(
        counts if source == "matrix" else adata, scp.Residual(clip=None)
    )

    with pytest.raises(IndexError, match="Invalid variable selector"):
        transformed.variable_statistics(var=selector)


def test_empty_selection_returns_empty_statistics(adata):
    """An empty variable selection returns empty arrays and names."""
    stats = scp.transform(adata, scp.Residual(clip=None)).variable_statistics(var=[])

    assert stats.mean.shape == stats.variance.shape == (0,)
    assert stats.var_names.tolist() == []


def test_variances_of_pca_variables_sum_to_pca_total_variance(adata):
    """With ddof=1, selected variances sum to the masked PCA's total variance.

    This is a consistency check between two uses of the same statistics, not
    an independent accuracy test.
    """
    transformed = scp.transform(adata, scp.Residual())
    mask = np.array([True, False, True, True])

    stats = transformed.variable_statistics(var=mask)
    result = transformed.pca(n_comps=2, mask_var=mask)

    assert math.fsum(stats.variance) == pytest.approx(
        result.total_variance, rel=1e-14, abs=0.0
    )


def test_constant_transform_returns_zero_variance_where_pca_raises():
    """Statistics accept zero variance, which PCA rejects as undefined."""
    counts = sparse.csr_matrix(np.tile([3, 1, 2], (5, 1)))
    transformed = scp.transform(counts, scp.Log1pNormalized(size_factors=np.ones(5)))

    stats = transformed.variable_statistics()

    np.testing.assert_array_equal(stats.variance, np.zeros(3))
    with pytest.raises(ValueError, match="numerically zero centered variance"):
        transformed.pca(n_comps=1)


@pytest.mark.parametrize(
    ("ddof", "error", "match"),
    [
        (1.0, TypeError, "ddof must be an integer"),
        (True, TypeError, "ddof must be an integer"),
        ("1", TypeError, "ddof must be an integer"),
        (-1, ValueError, r"0 <= ddof < n_obs=6"),
        (6, ValueError, r"0 <= ddof < n_obs=6"),
    ],
)
def test_invalid_ddof_is_rejected(counts, ddof, error, match):
    """ddof must be an integer that leaves a positive variance denominator."""
    transformed = scp.transform(counts, scp.Residual(clip=None))

    with pytest.raises(error, match=match):
        transformed.variable_statistics(ddof=ddof)


def test_numpy_integer_ddof_is_accepted(counts):
    """A NumPy integer ddof is accepted and recorded as a Python int."""
    stats = scp.transform(counts, scp.Residual(clip=None)).variable_statistics(
        ddof=np.int64(0)
    )

    assert stats.ddof == 0
    assert type(stats.ddof) is int


def test_named_selection_requires_anndata(counts):
    """Matrix-derived transforms reject variable names."""
    transformed = scp.transform(counts, scp.Residual(clip=None))

    with pytest.raises(TypeError, match="requires an AnnData input"):
        transformed.variable_statistics(var=["a"])


def test_anndata_statistics_carry_variable_names(adata):
    """Statistics from an AnnData-derived transform name every variable."""
    stats = scp.transform(
        AnnData(adata.X.copy(), var=adata.var.copy()), scp.Residual(clip=None)
    ).variable_statistics()

    assert stats.var_names.tolist() == ["a", "b", "c", "d"]
