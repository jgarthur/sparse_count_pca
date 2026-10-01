# %%
"""Normalized-log1p PCA with target totals and supplied size factors.

Companion notebook for the normalized-log1p guide. The percent-cell format
opens directly as a notebook in Jupyter or VS Code.
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
            [2, 1, 0, 2, 1],
        ],
        dtype=np.int32,
    )
)
adata = AnnData(counts.copy())
adata.layers["counts"] = counts

# %% [markdown]
# **The default target is the median observation total.** Counts remain in
# the layer; PCA outputs use the standard Scanpy keys.

# %%
scp.log1p_norm_pca(adata, layer="counts", n_comps=2)

totals = np.asarray(counts.sum(axis=1)).ravel()
print("sorted observation totals:", np.sort(totals))
print("resolved target:", adata.uns["pca"]["params"]["resolved_target_sum"])
print("scores:", adata.obsm["X_pca"].shape)
print("components:", adata.varm["PCs"].shape)

# %% [markdown]
# **An explicit target** changes the scale before taking `log1p`. The matrix
# API returns a result object.

# %%
explicit = scp.log1p_norm_pca_matrix(counts, n_comps=2, target_sum=1e4)
print("explicit target:", explicit.params["resolved_target_sum"])
print("singular values:", explicit.singular_values.round(3))

# %% [markdown]
# **Supplied factors are used as-is.** These illustrative factors are stored
# in `adata.obs["size_factor"]`. PCA scores are stored separately in
# `adata.obsm["supplied"]`, using the `key_added` value below.

# %%
adata.obs["size_factor"] = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 0.8])
scp.log1p_norm_pca(
    adata,
    layer="counts",
    n_comps=2,
    size_factors="size_factor",
    key_added="supplied",
)
print("factor source:", adata.uns["supplied"]["params"]["size_factor_source"])
print('PCA scores in adata.obsm["supplied"]:', adata.obsm["supplied"].shape)

# %% [markdown]
# **Inspect transformed values** through the two-step API. This small slice
# also checks the normalization formula directly.

# %%
normalized = scp.transform(
    adata,
    scp.Log1pNormalized(size_factors="size_factor"),
    layer="counts",
)
values = normalized.materialize(obs=slice(0, 2))
expected = np.log1p(
    counts[:2].toarray() / adata.obs["size_factor"].to_numpy()[:2, None]
)
np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-14)
print("first two normalized observations:")
print(values.round(3))

# %% [markdown]
# **Run PCA on selected genes.** The log1p values above were computed using
# the supplied size factors. Selecting four genes for PCA does not recalculate
# those normalized values.

# %%
mask = np.array([True, True, True, True, False])
masked = normalized.pca(n_comps=2, mask_var=mask)
print("normalization genes:", masked.params["normalization_n_vars"])
print("PCA genes:", masked.params["pca_n_vars"])
print("masked components:", masked.components.shape)
