# %%
"""Shifted-CLR PCA on a generated AnnData count matrix.

Companion notebook for the shifted-CLR guide. The percent-cell format opens
directly as a notebook in Jupyter or VS Code.
"""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import sparse_count_pca as scp

# %%
counts = csr_matrix(
    np.array(
        [
            [6, 0, 1, 0, 2],
            [0, 4, 0, 2, 1],
            [3, 1, 0, 2, 4],
            [0, 2, 5, 1, 0],
            [2, 0, 1, 4, 1],
            [1, 3, 2, 0, 3],
        ],
        dtype=np.int32,
    )
)
adata = AnnData(counts.copy())
adata.layers["counts"] = counts

# %% [markdown]
# **Count-scale shifted CLR**, which subtracts each observation's mean shifted
# log. `count_shift` is a raw-count shift, not library-size normalization.

# %%
scp.shifted_clr_pca(
    adata,
    layer="counts",
    n_comps=2,
    count_shift=1.0,
)

print("shifted-CLR scores:", adata.obsm["X_pca"].shape)

# %% [markdown]
# **The PFlog parameterization** is count-scale shifted CLR with
# `count_shift = 1 / (4 * alpha)`.

# %%
alpha = 0.25
pflog = scp.shifted_clr_pca_matrix(
    counts,
    n_comps=2,
    count_shift=1 / (4 * alpha),
)

print("PFlog singular values:", pflog.singular_values.round(3))

# %% [markdown]
# **Composition-scale shifted CLR** divides by each observation total before
# adding its shift, so its effective raw-count shift varies with depth.

# %%
proportion = scp.proportion_shifted_clr_pca_matrix(
    counts,
    n_comps=2,
    composition_shift=0.1,
)

print("composition-scale singular values:", proportion.singular_values.round(3))

# %% [markdown]
# The two CLR transforms give different results on the same counts, because
# observation totals differ.

# %%
count_scale = scp.shifted_clr_pca_matrix(counts, n_comps=2, count_shift=1.0)

print("observation totals:", np.asarray(counts.sum(axis=1)).ravel())
print("count-scale:", count_scale.singular_values.round(3))
print("composition-scale:", proportion.singular_values.round(3))
