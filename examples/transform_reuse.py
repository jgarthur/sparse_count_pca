"""Fit one residual transform and reuse it for inspection and PCA."""

import numpy as np
from scipy.sparse import csr_matrix

import sparse_count_pca as scp


def main() -> None:
    """Materialize one row and run PCA on a selected variable subset."""
    counts = csr_matrix(
        np.array(
            [
                [4, 0, 1, 0],
                [0, 3, 0, 2],
                [5, 1, 0, 1],
                [0, 2, 4, 1],
                [2, 0, 1, 3],
            ],
            dtype=np.int32,
        )
    )
    transformed = scp.transform(
        counts,
        scp.Residual(model="poisson", residual="pearson"),
    )

    first_row = transformed.materialize(obs=0)
    result = transformed.pca(
        n_comps=2,
        mask_var=np.array([True, True, True, False]),
    )

    print("materialized row:", first_row.shape)
    print("masked components:", result.components.shape)


if __name__ == "__main__":
    main()
