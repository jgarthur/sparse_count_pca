# %%
"""Inspect and plot normalized genes, then reuse a transform for PCA.

Companion notebook for the normalized-values and transform-reuse guide. The
percent-cell format opens directly as a notebook in Jupyter or VS Code.
"""

from copy import deepcopy

import numpy as np
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
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
adata = AnnData(counts.copy())
adata.layers["counts"] = counts
adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
adata.obs["cell_type"] = ["A", "B", "A", "B", "A", "B"]
adata.uns["cell_type_colors"] = ["#4477AA", "#EE6677"]

# %% [markdown]
# **Fit the normalization once.** `transform` accepts the same count sources as
# the one-step APIs and returns an uncentered linear operator.

# %%
transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
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
# **Collect selected genes for plotting.** `normalized_for_plotting` returns
# the named genes' normalized values in a new AnnData object, with copies of
# `adata`'s `.obs`, `.obsm`, and `.uns`. `adata` is unchanged.


# %%
def normalized_for_plotting(adata, transformed, genes):
    """Return normalized values of genes for adata's cells as a new AnnData.

    genes must be named in both objects and adata's cells in transformed.
    """
    return AnnData(
        X=transformed.materialize(obs=adata.obs_names.tolist(), var=genes),
        obs=adata.obs.copy(),
        var=adata.var.loc[genes].copy(),
        obsm=adata.obsm.copy(),
        uns=deepcopy(adata.uns),
    )


# %% [markdown]
# **Plot with Scanpy.** PCA is stored in `adata` before `plot_data` is built,
# so the helper copies its scores along with the `cell_type` labels and colors.

# %%
result = transformed.pca(n_comps=2, mask_var=None)
adata.obsm["X_pca"] = result.scores
genes = ["gene_0", "gene_2"]
plot_data = normalized_for_plotting(adata, transformed, genes)
sc.pl.violin(plot_data, keys=genes, groupby="cell_type", show=False)
plt.show()
plt.close("all")

# %% [markdown]
# **Color an embedding by gene.** This example colors PCA coordinates so that
# it needs no neighbor graph or UMAP; with real data, `sc.pl.umap` uses the
# copied `X_umap` the same way. Pearson residuals take both signs, so the
# colormap is diverging and centered at zero.

# %%
sc.pl.embedding(
    plot_data,
    basis="pca",
    color=genes,
    size=100,
    cmap="RdBu_r",
    vcenter=0,
    show=False,
)
plt.show()
plt.close("all")

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
