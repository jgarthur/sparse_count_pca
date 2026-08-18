# Getting started

## Installation

Activate the Python environment where you use Scanpy or AnnData, then install
the package from PyPI:

```bash
pip install sparse-count-pca
```

Python 3.10 or newer is required. The runtime dependencies are AnnData, NumPy,
SciPy, and scikit-learn.

## A complete residual-PCA example

The package expects observations in rows and variables in columns. In an
`AnnData` object, counts can be read from `.X` or a named layer. This example
preserves counts in a layer and writes PCA results in place:

```python
import numpy as np
import sparse_count_pca as scp
from anndata import AnnData
from scipy.sparse import csr_matrix

counts = csr_matrix(
    np.array(
        [
            [4, 0, 1, 0],
            [0, 3, 0, 2],
            [5, 1, 0, 0],
            [0, 2, 4, 1],
        ],
        dtype=np.int32,
    )
)
adata = AnnData(counts.copy())
adata.layers["counts"] = counts

scp.residual_pca(
    adata,
    layer="counts",
    n_comps=2,
    model="poisson",
    residual="pearson",
)

print(adata.obsm["X_pca"].shape)  # (4, 2)
print(adata.varm["PCs"].shape)    # (4, 2)
```

The default call mutates `adata` and returns `None`. Pass `copy=True` to return
a modified copy instead.

Residuals are clipped by default, symmetrically at
`clip="seurat"` — `sqrt(n_obs / 30)`, resolved from the count matrix being
fitted. Pass `clip="scanpy"` for `sqrt(n_obs)`, a float for an explicit
threshold, or `clip=None` for no clipping. See
[clipping and precision](concepts/clipping-and-precision.md).

## Choosing `n_comps`

`n_comps` defaults to 50, the usual starting point for single-cell PCA. There
is no automatic selection; `adata.uns["pca"]["variance_ratio"]` reports the
variance explained by each retained component.

The ARPACK solver requires `n_comps` to be strictly less than both the number
of observations and the number of selected variables whose transformed column
is not identically zero, which is why the example above uses 2 for a
four-by-four matrix. Such a column carries no variance and cannot support a
component. Whether a gene with no counts produces one depends on the transform;
each transform's guide states its own behavior.

## What was written

By default, PCA outputs follow Scanpy's key layout:

| Location | Contents |
| --- | --- |
| `adata.obsm["X_pca"]` | observation scores |
| `adata.varm["PCs"]` | variable loadings with shape `(n_variables, n_components)` |
| `adata.uns["pca"]` | variance statistics, with reproducibility parameters nested under `["params"]` |

Passing `key_added="my_pca"` uses that exact key in all three mappings.

Variance statistics are direct entries, such as
`adata.uns["pca"]["variance_ratio"]`. Everything describing *how* the run was
configured is one level deeper, such as
`adata.uns["pca"]["params"]["n_empty_vars"]`. The matrix API exposes the same
two groups as `result.explained_variance_ratio` and `result.params[...]`.

## Continue in Scanpy

Because the scores land at Scanpy's own keys, the rest of a standard workflow
runs unchanged:

```python
import scanpy as sc

sc.pp.neighbors(adata)
sc.tl.umap(adata)
```

Scanpy looks for `obsm["X_pca"]` by name. Pass `use_rep` with your own key if
you used a custom `key_added`, or `use_rep="X_pca"` if the matrix has 50 or
fewer variables and Scanpy would otherwise read `.X` directly.

## Next steps

- Read the [transform catalog](transforms.md) before switching
  normalization families.
- Use the [residual PCA guide](guides/residual-pca.md) for model, residual,
  clipping, and masking options.
- Use [AnnData workflows](guides/anndata-workflows.md) for `.X`, layers, raw
  counts, copying, and result keys.
- Use the [matrix API](reference/api/matrix.md) when you do not have an
  `AnnData` object.
