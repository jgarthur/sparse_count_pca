# %%
"""Fit one transform and reuse it for inspection and several PCA masks.

Companion notebook for the transform-reuse guide. The percent-cell format
opens directly as a notebook in Jupyter or VS Code.
"""

import numpy as np
from scipy.sparse import csr_matrix

import sparse_count_pca as scp

# %%
counts = csr_matrix(
    np.array(
        [
            [4, 0, 1, 0, 2],
            [0, 3, 0, 2, 1],
            [5, 1, 0, 1, 3],
            [0, 2, 4, 1, 0],
            [2, 0, 1, 3, 1],
            [1, 4, 2, 0, 2],
        ],
        dtype=np.int32,
    )
)

# %% [markdown]
# **Fit the normalization once.** `transform` accepts the same count sources as
# the one-step APIs and returns an uncentered linear operator.

# %%
transformed = scp.transform(
    counts,
    scp.Residual(model="poisson", residual="pearson"),
)

print("shape:", transformed.shape)
print("dtype:", transformed.dtype)

# %% [markdown]
# **Materialize bounded slices.** Materialization always returns a
# two-dimensional array, including for a single observation.

# %%
first_row = transformed.materialize(obs=0)
block = transformed.materialize(obs=[0, 2], var=[1, 3, 4])

print("one observation:", first_row.shape)
print("observation block:", block.shape)
print(block.round(3))

# %% [markdown]
# **Reuse the fitted normalization with different PCA masks.** Each call
# selects variables and then computes its own column mean.

# %%
first = transformed.pca(
    n_comps=2,
    mask_var=np.array([True, True, True, False, False]),
)
second = transformed.pca(
    n_comps=2,
    mask_var=np.array([False, True, True, True, True]),
)

print("first mask components:", first.components.shape)
print("second mask components:", second.components.shape)
print("normalization genes:", first.params["normalization_n_vars"])

# %% [markdown]
# **The mask selects PCA genes; it does not refit the normalization.** `first`
# ran PCA on genes 0-2 using cell totals and gene proportions fitted on all
# five. Refitting the normalization on only those three genes is a different
# calculation, and gives different singular values.

# %%
subset = scp.residual_pca_matrix(
    counts[:, :3],
    n_comps=2,
    model="poisson",
    residual="pearson",
)

print("reused normalization:", first.singular_values.round(3))
print("refitted on subset:  ", subset.singular_values.round(3))
