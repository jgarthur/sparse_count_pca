"""Compatibility tests against Scanpy total normalization and log1p."""

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from sparse_count_pca import Log1pNormalized, transform

scanpy = pytest.importorskip("scanpy")


@pytest.mark.parametrize(
    "target_sum",
    [None, 1e4],
    ids=["median-target", "explicit-target"],
)
def test_log1p_norm_matches_scanpy_transform(target_sum):
    """Median and explicit targets agree with Scanpy on a six-by-four count table."""
    counts = sparse.csr_matrix(
        [
            [5, 1, 0, 2],
            [1, 4, 2, 0],
            [0, 2, 5, 1],
            [3, 0, 1, 4],
            [2, 3, 0, 2],
            [1, 1, 3, 2],
        ]
    )
    explicit = AnnData(counts.astype(np.float64))
    scanpy.pp.normalize_total(explicit, target_sum=target_sum)
    scanpy.pp.log1p(explicit)
    expected_values = explicit.X.toarray()

    actual_values = transform(
        counts,
        Log1pNormalized(target_sum=target_sum),
        dtype="float64",
    ).materialize()
    np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=1e-12)
