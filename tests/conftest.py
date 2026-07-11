import pytest
from anndata import AnnData
from scipy import sparse


@pytest.fixture
def counts():
    return sparse.csr_matrix(
        [
            [5, 1, 0, 2],
            [1, 4, 2, 0],
            [0, 2, 5, 1],
            [3, 0, 1, 4],
            [2, 3, 0, 2],
            [1, 1, 3, 2],
        ]
    )


@pytest.fixture
def adata(counts):
    result = AnnData(counts.copy())
    result.var_names = ["a", "b", "c", "d"]
    result.layers["counts"] = counts.copy()
    return result
