# Normalized values and transform reuse

The full matrix of normalized values is not stored, by design. Use `transform`
to fit the normalization once. The fitted object can then materialize the
values you need for plotting or other uses, compute each gene's variance to
select highly variable genes, and run PCA several times with different gene
masks.

This guide's [complete example](#complete-example) is
[`examples/transform_reuse.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/transform_reuse.py).

## Plot normalized gene values

Starting from an `adata` that already has PCA, a UMAP, and Leiden clusters,
fit the transform with the same parameters as the PCA call, collect a few
genes, and plot them with Scanpy:

```python
from copy import deepcopy

import scanpy as sc
from anndata import AnnData

import sparse_count_pca as scp


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


transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
)

genes = ["CD3E", "MS4A1", "LYZ"]
plot_data = normalized_for_plotting(adata, transformed, genes)

sc.pl.umap(plot_data, color=genes, cmap="RdBu_r", vcenter=0)
sc.pl.violin(plot_data, keys=genes, groupby="leiden")
```

The helper preserves annotations, embeddings, and colors without changing
`adata`.

Fit `transformed` from AnnData with unique names. Requested genes must be in
both objects, and plotted cells in `transformed`.

Residuals and CLR coordinates take both signs, so a diverging colormap centered
at zero, as above, keeps the sign readable.

## Fit a transform

`transform` accepts the same count sources as the one-step functions and
returns a `TransformedMatrix`:

```python
import sparse_count_pca as scp

transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
)
```

Other primary transform specifications are `Log1pNormalized`, `ShiftedCLR`,
and `ProportionShiftedCLR`:

```python
log_normalized = scp.transform(
    adata,
    scp.Log1pNormalized(size_factors="size_factors"),
    layer="counts",
)
```

String-valued per-observation parameters, such as the `adata.obs` size-factor
key above, are resolved when the transform is fitted. See the
[normalized-log1p guide](log1p-normalization.md) for target totals, supplied
factors, and observation alignment.

Correspondence analysis remains a separate one-step API because its variable
selection changes the fitted table margins.

## Inspect selected values

`TransformedMatrix` is an uncentered `scipy.sparse.linalg.LinearOperator`.
Materialization always returns a two-dimensional array, including for a single
row or variable:

```python
cell = transformed.materialize(obs=10)
gene = transformed.materialize(var=25)
subset = transformed.materialize(obs=[10, 3], var=[25, 2, 8])
```

Use positional slices or boolean masks for larger selections:

```python
first_cells = transformed.materialize(obs=slice(0, 5))
selected_cells = transformed.materialize(obs=adata.obs["cell_type"] == "B cell")
selected_genes = transformed.materialize(
    obs=slice(0, 5), var=adata.var["highly_variable"].to_numpy()
)
```

The metadata columns in these examples must exist in your AnnData object.
For a transform fitted from AnnData, names work too:

```python
named = transformed.materialize(
    obs=adata.obs_names[:3].tolist(), var=adata.var_names[:2].tolist()
)
```

Lists preserve order and may repeat entries. Selecting both axes returns a
rectangular block, not paired entries. See the
[`materialize` API examples](../reference/api/transforms.md#sparse_count_pca.TransformedMatrix.materialize)
for single values, negative indices, repeated positions, and output buffers.

The current backend is observation-oriented CSR. Observation blocks are the
efficient access direction. Selecting variables across every observation may
scan the CSR rows and emits `SparseEfficiencyWarning`.

## Bound materialization memory

Provide `out` and `block_size` to stream row blocks into an existing NumPy
array or memory map:

```python
import numpy as np

out = np.memmap(
    "transformed.dat",
    mode="w+",
    shape=transformed.shape,
    dtype="float32",
)
transformed.materialize(out=out, block_size=2048)
```

This bounds temporary output memory, but it does not make the current fitting
backend out of core: backed sparse inputs are loaded into an in-memory CSR
matrix during count canonicalization.

## Rank genes by variance

Highly variable gene (HVG) selection keeps the genes whose normalized values
vary most across cells. `variable_statistics` efficiently computes each gene's
mean and variance under the fitted transform. This selects the 2,000 genes
with the largest Pearson-residual variance and runs PCA on them:

```python
import numpy as np

stats = transformed.variable_statistics()

highly_variable = np.zeros(adata.n_vars, dtype=bool)
highly_variable[np.argsort(-stats.variance)[:2000]] = True
adata.var["highly_variable"] = highly_variable
adata.var["residual_variance"] = stats.variance

result = transformed.pca(n_comps=50)
```

`pca` uses `adata.var["highly_variable"]` by default, including a column added
after the transform was fitted.

The statistics have these properties:

- **They describe the fitted values, including any residual clipping.**
- **The default `ddof=1` divides by `n_obs - 1`,** as PCA's `total_variance`
  does, so the variances of the genes passed to `pca` sum to its
  `total_variance`. Scanpy's `residual_variances` divide by `n_obs`; pass
  `ddof=0` to match them. The choice of `ddof` does not change the ranking.
- **`var=` selects genes without refitting the normalization.** It accepts the
  same selectors as `materialize`, and results follow the selection's order.
- **Genes with no counts** have zero variance under residuals and normalized
  `log1p`, and generally positive variance under CLR and Dirichlet transforms.

Scanpy's `highly_variable_genes(flavor="pearson_residuals")` with
`theta=np.inf` uses the same Poisson residuals. For negative-binomial
residuals, use `Residual(model="scaled_nb")`. Unlike Scanpy's default
`theta=100`, which applies one overdispersion to every gene and cell, its
per-gene overdispersion scales inversely with cell depth, as the
[compatibility
reference](../reference/compatibility.md#the-shared-theta-model-is-not-offered)
explains.

## Reuse normalization with different PCA masks

```python
first = transformed.pca(n_comps=20, mask_var="highly_variable")
second = transformed.pca(n_comps=20, mask_var=my_alternative_mask)
```

Normalization state was fitted once on the full count matrix. Each `pca` call
selects variables and then computes its own column mean. This ordering is
described in [normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

A string `mask_var`, and the default `"highly_variable"`, are read from
`adata.var` when `pca` is called, so annotations added after fitting are used.
If `adata` has since been deleted or subset to other genes, pass a boolean
array aligned with `transformed.var_names` instead:

```python
mask = transformed.var_names.isin(adata.var_names[adata.var["highly_variable"]])
result = transformed.pca(n_comps=20, mask_var=mask)
```

Unlike the one-step APIs, `pca` **returns** a `PCAResult` and writes nothing
back, even when the transform was fitted from an `AnnData` object. That is what
makes several results comparable side by side. To store one, assign it
yourself:

```python
adata.obsm["X_pca"] = first.scores
adata.varm["PCs"] = first.components.T
```

With a `mask_var`, `components` covers only the selected variables, so a
full-width `varm` array needs the unselected rows filled with `np.nan`.

## Lifetime and ownership

By default the fitted object owns the sparse support needed to isolate it
from later mutation of the caller's matrix. It does not keep the AnnData
object it was fitted from in memory. Once that object is deleted, or if it was
never assigned to a variable, as in `scp.transform(sc.read_h5ad(path), ...)`,
`pca` needs a boolean-array `mask_var` or `mask_var=None`.

The fitted object lives only in the current Python process:
fitting a transform adds nothing to the AnnData object, so `write_h5ad` does
not store it, and it has to be refitted after reloading the file or restarting
Python. Only PCA results written to `.obsm`, `.varm`, and `.uns` persist.

Passing `return_operator=True` to `pca` retains the centered operator ARPACK
used. It is isolated from the `TransformedMatrix` it came from: mutating either
one cannot affect the other.

By default, the fitted object keeps its own copy of the count matrix's column
indices and row pointers, so it uses more memory than the equivalent one-step
function, which reuses them unless it returns its operator. For large matrices,
`transform(..., share_structure=True)` lets the fitted object reuse them too,
unless fitting has to change or convert the structure: inputs that need
canonicalization or conversion to CSR, and residual clipping that adds entries,
still produce a copy, with a `UserWarning` saying so.
In return, do not modify the count matrix's sparsity structure while the
fitted object is in use, for example with `eliminate_zeros()` after zeroing
some values. If it changes, `pca` and `materialize` raise an error asking you
to fit the transform again. Changing nonzero count values or replacing the
matrix, for example by assigning a new `adata.X`, is safe. Returned operators
are always isolated.

## Complete example

--8<-- "examples/transform_reuse.md"
