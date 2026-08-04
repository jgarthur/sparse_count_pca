# %%
"""Count sources, variable masks, result keys, and copy semantics in AnnData.

Companion notebook for the AnnData-workflows guide. The percent-cell format
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
            [4, 0, 1, 0, 2],
            [0, 3, 0, 2, 1],
            [5, 1, 0, 0, 3],
            [0, 2, 4, 1, 0],
            [2, 0, 1, 3, 1],
            [1, 4, 2, 0, 2],
        ],
        dtype=np.int32,
    )
)
adata = AnnData(counts.copy())
adata.layers["counts"] = counts

# %% [markdown]
# **Choose `.X` or a named layer.** Omitting `layer` reads `adata.X`.

# %%
scp.residual_pca(adata, n_comps=2, key_added="from_x")
scp.residual_pca(adata, layer="counts", n_comps=2, key_added="from_layer")

for key in ("from_x", "from_layer"):
    params = adata.uns[key]["params"]
    print(key, "-> layer:", params["layer"])

# %% [markdown]
# **Masks accept a `var` column name or a boolean array**, and must have
# genuinely boolean dtype.

# %%
adata.var["highly_variable"] = np.array([True, True, True, True, False])

scp.residual_pca(adata, layer="counts", n_comps=2, key_added="by_name")
scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    mask_var=np.array([False, True, True, True, True]),
    key_added="by_array",
)
scp.residual_pca(adata, layer="counts", n_comps=2, mask_var=None, key_added="all_genes")

for key in ("by_name", "by_array", "all_genes"):
    params = adata.uns[key]["params"]
    print(key, "-> PCA genes:", params["pca_n_vars"])

# %% [markdown]
# `adata.var["highly_variable"]` is used by default when it exists, so
# `by_name` above resolved the mask without being asked. Normalization is
# always fitted on every gene in the chosen count matrix.

# %%
print("resolved mask:", adata.uns["by_name"]["params"]["mask_var_details"])
print("normalization genes:", adata.uns["by_name"]["params"]["normalization_n_vars"])

# %% [markdown]
# **Results are written to three mappings.** `key_added` uses that exact key in
# all three.

# %%
print("obsm:", adata.obsm["by_name"].shape)
print("varm:", adata.varm["by_name"].shape)
print("uns:", sorted(adata.uns["by_name"]))

# %% [markdown]
# Masked variables receive `NaN` component values rather than zeros, so an
# excluded gene cannot be mistaken for one with no loading.

# %%
print(adata.varm["by_name"].round(3))

# %% [markdown]
# **`copy=True` leaves the input unchanged** and returns a modified copy.

# %%
fresh = AnnData(counts.copy())
fresh.layers["counts"] = counts
returned = scp.residual_pca(fresh, layer="counts", n_comps=2, copy=True)

print("original obsm keys:", list(fresh.obsm))
print("returned obsm keys:", list(returned.obsm))

# %% [markdown]
# **Recorded metadata** supports reproducing a result later.

# %%
for name, value in sorted(adata.uns["by_name"]["params"].items()):
    print(f"{name}: {value}")
