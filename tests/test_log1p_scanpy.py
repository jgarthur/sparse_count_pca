"""Compatibility tests against Scanpy total normalization and log1p."""

import numpy as np
import pytest
from anndata import AnnData

from sparse_count_pca import Log1pNormalized, transform

scanpy = pytest.importorskip("scanpy")


@pytest.mark.parametrize(
    "target_sum",
    [None, 1e4],
    ids=["median-target", "explicit-target"],
)
def test_log1p_norm_matches_scanpy_transform(counts, target_sum):
    """Target-normalized log1p values agree with Scanpy."""
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
