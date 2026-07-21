# %%
"""Run classical correspondence analysis on a generated count table."""

import numpy as np
from scipy.sparse import csr_matrix

import sparse_count_pca as scp


# %%
def main() -> None:
    """Compute two CA axes and print row and column coordinate shapes."""
    counts = csr_matrix(
        np.array(
            [
                [8, 1, 0, 2],
                [1, 7, 2, 0],
                [0, 2, 8, 1],
                [3, 0, 1, 7],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        )
    )
    result = scp.correspondence_analysis_matrix(counts, n_comps=2)

    print("row coordinates:", result.row_principal_coordinates.shape)
    print("column coordinates:", result.column_principal_coordinates.shape)


# %%
if __name__ == "__main__":
    main()
