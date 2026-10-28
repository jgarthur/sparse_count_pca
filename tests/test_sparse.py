"""Tests for sparse support ownership and index-width selection."""

import tracemalloc

import numpy as np
import pytest
from scipy import sparse

from sparse_count_pca import _sparse as sparse_module
from sparse_count_pca._sparse import _csr_with_data, _support_checksum


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


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_csr_with_data_borrows_support_at_input_width(index_dtype):
    """Borrowed small-matrix support keeps the input index width without a copy."""
    counts = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    counts.indices = counts.indices.astype(index_dtype)
    counts.indptr = counts.indptr.astype(index_dtype)
    result = _csr_with_data(counts, np.array([3.0, 4.0]), share_structure=True)

    for name in ("indices", "indptr"):
        assert getattr(result, name).dtype == np.dtype(index_dtype)
        assert np.shares_memory(getattr(result, name), getattr(counts, name))
    np.testing.assert_array_equal(result.toarray(), [[3.0, 0.0], [0.0, 4.0]])


@pytest.mark.parametrize("share_structure", [False, True])
def test_csr_with_data_preserves_required_int64_support(share_structure):
    """A sparse matrix whose columns exceed int32 range retains int64 support."""
    column = np.iinfo(np.int32).max
    counts = sparse.csr_matrix(
        ([1.0], np.array([column], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        shape=(1, column + 1),
    )
    result = _csr_with_data(counts, np.array([2.0]), share_structure=share_structure)

    for name in ("indices", "indptr"):
        assert getattr(result, name).dtype == np.dtype(np.int64)
        assert (
            np.shares_memory(getattr(result, name), getattr(counts, name))
            == share_structure
        )
        np.testing.assert_array_equal(getattr(result, name), getattr(counts, name))


def _strided_support(n_rows, n_cols):
    """Return CSR whose borrowed indices are a strided view."""
    dense = np.ones((n_rows, n_cols))
    contiguous = sparse.csr_matrix(dense)
    padded = np.zeros((contiguous.nnz, 2), dtype=contiguous.indices.dtype)
    padded[:, 0] = contiguous.indices
    strided = sparse.csr_matrix(
        (contiguous.data, padded[:, 0], contiguous.indptr),
        shape=contiguous.shape,
        copy=False,
    )
    assert not strided.indices.flags.c_contiguous
    return contiguous, strided


@pytest.mark.parametrize("chunk_size", [1, 7, 1_000_000])
def test_support_checksum_is_independent_of_chunking_and_strides(
    monkeypatch, chunk_size
):
    """Chunked hashing of strided support matches one pass over a copy."""
    contiguous, strided = _strided_support(13, 11)
    expected = _support_checksum(contiguous)
    monkeypatch.setattr(sparse_module, "_SUPPORT_CHECKSUM_CHUNK_SIZE", chunk_size)

    assert _support_checksum(contiguous) == expected
    assert _support_checksum(strided) == expected


def test_support_checksum_copies_strided_support_one_chunk_at_a_time(monkeypatch):
    """Hashing strided support allocates at most one chunk, not the whole array."""
    _, strided = _strided_support(400, 500)
    monkeypatch.setattr(sparse_module, "_SUPPORT_CHECKSUM_CHUNK_SIZE", 1_000)
    tracemalloc.start()
    _support_checksum(strided)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < strided.indices.nbytes / 10
