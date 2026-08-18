# %%
"""Residual PCA on a generated AnnData count matrix.

Companion notebook for the residual-PCA guide. The percent-cell format opens
directly as a notebook in Jupyter or VS Code.
"""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import sparse_count_pca as scp

# %% [markdown]
# **Build a small count matrix with observations in rows and genes in columns.**

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
# **Poisson Pearson residual PCA, written to Scanpy-compatible keys.**

# %%
scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    model="poisson",
    residual="pearson",
)

print("scores:", adata.obsm["X_pca"].shape)
print("components:", adata.varm["PCs"].shape)
print("variance ratio:", adata.uns["pca"]["variance_ratio"].round(3))

# %% [markdown]
# **Deviance residuals under the scaled-NB model, with supplied overdispersion.**
#
# The package consumes per-gene overdispersion values; it does not estimate
# them.

# %%
adata.var["overdispersion"] = np.full(adata.n_vars, 0.1)

scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    model="scaled_nb",
    residual="deviance",
    alpha="overdispersion",
    key_added="scaled_nb_pca",
)

print("scaled-NB scores:", adata.obsm["scaled_nb_pca"].shape)

# %% [markdown]
# **Select PCA genes with a mask.** Normalization is still fitted on all five
# genes; masked genes receive `NaN` component values.

# %%
adata.var["highly_variable"] = np.array([True, True, True, False, False])

scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    mask_var="highly_variable",
    key_added="masked_pca",
)

print("normalization genes:", adata.uns["masked_pca"]["params"]["normalization_n_vars"])
print("PCA genes:", adata.uns["masked_pca"]["params"]["pca_n_vars"])
print("masked component rows:", np.isnan(adata.varm["masked_pca"]).all(axis=1).sum())

# %% [markdown]
# **Residuals are clipped before centering by default**, symmetrically at the
# named `"seurat"` threshold `sqrt(n_obs / 30)`. The result records both the
# request and the number it resolved to.

# %%
print("requested clip:", adata.uns["pca"]["params"]["clip"])
print("resolved threshold:", adata.uns["pca"]["params"]["clip_threshold"])

# %% [markdown]
# **A name chooses the threshold, never the mode.** Upper-only clipping at the
# same named threshold never expands the stored sparse support, and `clip=None`
# turns clipping off.

# %%
scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    clip="seurat",
    clip_mode="upper",
    key_added="clipped_pca",
)

print("clipped scores:", adata.obsm["clipped_pca"].shape)

# %% [markdown]
# **The matrix API returns a result object instead of writing to AnnData.**

# %%
result = scp.residual_pca_matrix(
    counts,
    n_comps=2,
    model="poisson",
    residual="deviance",
)

print("scores:", result.scores.shape)
print("components:", result.components.shape)
print("singular values:", result.singular_values.round(3))
