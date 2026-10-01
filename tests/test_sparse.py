"""Tests for sparse support ownership and index-width selection."""

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca._sparse import _csr_with_data


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_csr_with_data_owns_compact_support(index_dtype):
    """Small outputs own int32 support even when input uses int64 indices."""
    counts = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    counts.indices = counts.indices.astype(index_dtype)
    counts.indptr = counts.indptr.astype(index_dtype)
    data = np.array([3.0, 4.0], dtype=np.float32)
    result = _csr_with_data(counts, data)

    assert result.dtype == data.dtype
    assert np.shares_memory(result.data, data)
    for name in ("indices", "indptr"):
        assert getattr(result, name).dtype == np.dtype(np.int32)
        assert not np.shares_memory(getattr(result, name), getattr(counts, name))
        np.testing.assert_array_equal(getattr(result, name), getattr(counts, name))
    np.testing.assert_array_equal(result.toarray(), [[3.0, 0.0], [0.0, 4.0]])


def test_csr_with_data_preserves_required_int64_support():
    """A sparse matrix whose columns exceed int32 range retains int64 support."""
    column = np.iinfo(np.int32).max
    counts = sparse.csr_matrix(
        ([1.0], np.array([column], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        shape=(1, column + 1),
    )
    result = _csr_with_data(counts, np.array([2.0]))

    for name in ("indices", "indptr"):
        assert getattr(result, name).dtype == np.dtype(np.int64)
        assert not np.shares_memory(getattr(result, name), getattr(counts, name))
        np.testing.assert_array_equal(getattr(result, name), getattr(counts, name))
