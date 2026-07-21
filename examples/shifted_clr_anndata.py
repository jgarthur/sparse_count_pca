"""Run fixed-count shifted-CLR PCA on a generated AnnData object."""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import sparse_count_pca as scp


def main() -> None:
    """Apply a unit count shift and print the resulting PCA shapes."""
    counts = csr_matrix(
        np.array(
            [
                [6, 0, 1, 0],
                [0, 4, 0, 2],
                [3, 1, 0, 2],
                [0, 2, 5, 1],
                [2, 0, 1, 4],
            ],
            dtype=np.int32,
        )
    )
    adata = AnnData(counts.copy())
    adata.layers["counts"] = counts

    scp.shifted_clr_pca(
        adata,
        layer="counts",
        n_comps=2,
        count_shift=1.0,
    )

    print("scores:", adata.obsm["X_pca"].shape)
    print("components:", adata.varm["PCs"].shape)


if __name__ == "__main__":
    main()
