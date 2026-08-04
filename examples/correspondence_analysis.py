# %%
"""Classical correspondence analysis of a generated count table.

Companion notebook for the correspondence-analysis guide. The percent-cell
format opens directly as a notebook in Jupyter or VS Code.
"""

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

import sparse_count_pca as scp

# %%
counts = csr_matrix(
    np.array(
        [
            [8, 1, 0, 2, 1],
            [1, 7, 2, 0, 1],
            [0, 2, 8, 1, 2],
            [3, 0, 1, 7, 1],
            [4, 3, 2, 1, 6],
        ],
        dtype=np.int32,
    )
)

# %% [markdown]
# **Matrix workflow.** Correspondence analysis reports coordinates scaled by
# row and column masses rather than PCA scores.

# %%
result = scp.correspondence_analysis_matrix(counts, n_comps=2)

print("row coordinates:", result.row_principal_coordinates.shape)
print("column coordinates:", result.column_principal_coordinates.shape)
print("principal inertias:", result.principal_inertias.round(4))
print("total inertia:", round(result.total_inertia, 4))
print("inertia ratio:", result.inertia_ratio.round(3))

# %% [markdown]
# `total_inertia` equals the Pearson chi-squared statistic divided by the grand
# total.

# %%
dense = counts.toarray().astype(float)
expected = np.outer(dense.sum(axis=1), dense.sum(axis=0)) / dense.sum()
chi_squared = ((dense - expected) ** 2 / expected).sum()

print("chi-squared / N:", round(chi_squared / dense.sum(), 4))
print("total inertia:  ", round(result.total_inertia, 4))

# %% [markdown]
# **AnnData workflow**, which writes to correspondence-analysis keys rather
# than the PCA keys.

# %%
adata = AnnData(counts.copy())
adata.layers["counts"] = counts

scp.correspondence_analysis(adata, layer="counts", n_comps=2)

print("obsm keys:", list(adata.obsm))
print("varm keys:", list(adata.varm))
print("row coordinates:", adata.obsm["X_ca"].shape)

# %% [markdown]
# **A variable mask defines a new contingency table.** Margins, masses, and
# inertia are recomputed from the selected columns, so masking a variable and
# removing it from the input give the same answer. Residual and CLR PCA behave
# differently: there, the mask selects PCA variables only.

# %%
adata.var["highly_variable"] = np.array([True, True, True, True, False])

scp.correspondence_analysis(
    adata,
    layer="counts",
    n_comps=2,
    mask_var="highly_variable",
    key_added="masked_ca",
)
subset = scp.correspondence_analysis_matrix(counts[:, :4], n_comps=2)

print("full table inertia:  ", round(result.total_inertia, 4))
print("masked table inertia:", round(adata.uns["masked_ca"]["total_inertia"], 4))
print("subset table inertia:", round(subset.total_inertia, 4))
