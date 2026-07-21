"""Run residual PCA on a small generated AnnData count matrix."""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import sparse_count_pca as scp


def main() -> None:
    """Construct counts, run residual PCA, and print the output shapes."""
    counts = csr_matrix(
        np.array(
            [
                [4, 0, 1, 0],
                [0, 3, 0, 2],
                [5, 1, 0, 0],
                [0, 2, 4, 1],
                [2, 0, 1, 3],
            ],
            dtype=np.int32,
        )
    )
    adata = AnnData(counts.copy())
    adata.layers["counts"] = counts

    scp.residual_pca(
        adata,
        layer="counts",
        n_comps=2,
        model="poisson",
        residual="pearson",
    )

    print("scores:", adata.obsm["X_pca"].shape)
    print("components:", adata.varm["PCs"].shape)


if __name__ == "__main__":
    main()
