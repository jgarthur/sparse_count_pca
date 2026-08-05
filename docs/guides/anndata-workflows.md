# AnnData workflows

The AnnData functions are meant to drop into a Scanpy workflow in place of a
normalization-plus-PCA step. They read counts from `.X` or a layer, resolve the
same `highly_variable` mask, and write scores and components to the same keys,
so everything downstream continues unchanged.

This guide's [complete example](#complete-example) is
[`examples/anndata_workflows.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/anndata_workflows.py).

## Select the count matrix

Use `.X` or a named layer:

```python
scp.residual_pca(adata)                  # adata.X
scp.residual_pca(adata, layer="counts")  # adata.layers["counts"]
```

Input arrays must contain finite, nonnegative counts. Floating-point inputs are
accepted when their values are integer-like within the package tolerance.

## Select PCA variables

When `mask_var` is omitted, `adata.var["highly_variable"]` is used if present;
otherwise all variables are used. Explicit `mask_var=None` always uses all
variables.

```python
scp.residual_pca(adata, mask_var=None)
scp.residual_pca(adata, mask_var="my_mask")
scp.residual_pca(adata, mask_var=boolean_array)
```

Masks must have genuinely boolean dtype and the correct length. Numeric,
string, and object columns are rejected rather than coerced.

A pandas nullable `boolean` column is accepted when it holds no missing values.
One containing `pd.NA` is rejected, since `pd.NA` has no defined meaning for
variable selection. Decide what missing should mean, then fill it in:

```python
adata.var["my_mask"] = adata.var["my_mask"].fillna(False).astype(bool)
```

Scanpy's older `use_highly_variable` argument is accepted but deprecated, and
emits a warning. `use_highly_variable=True` means `mask_var="highly_variable"`;
`use_highly_variable=False` means `mask_var=None`.

For residual, log, and CLR PCA, the mask changes the PCA variables but not the
normalization universe. For correspondence analysis, it defines a new table
and new margins. See
[normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

## Produce a highly-variable-gene mask

This package consumes `adata.var["highly_variable"]` but does not compute it.
Scanpy provides the selection step, and its flavor should match the analysis:

```python
import scanpy as sc

# Pairs with residual PCA; selects on Pearson-residual variance from raw counts.
sc.experimental.pp.highly_variable_genes(
    adata,
    flavor="pearson_residuals",
    layer="counts",
    n_top_genes=2000,
)

scp.residual_pca(adata, layer="counts", n_comps=50)
```

Selection and PCA fit separate null models, and their parameters need not
agree: Scanpy's `"pearson_residuals"` flavor defaults to a negative binomial
with `theta=100` and clips residuals, while `residual_pca` defaults to Poisson
without clipping. The pairing above is a match of residual family, not of
parameters; selection only decides which genes enter the PCA.

Scanpy's other flavors (`"seurat"`, `"cell_ranger"`, `"seurat_v3"`) each expect
a particular input scale; any boolean `var` column works here.

## Output keys

PCA functions use these defaults:

```text
adata.obsm["X_pca"]
adata.varm["PCs"]
adata.uns["pca"]
```

Correspondence analysis defaults to `"X_ca"`, `"CA"`, and `"ca"`.

When `key_added="name"` is provided, all three result mappings use the exact
key `"name"`. This matches current Scanpy key behavior, even though it means
the same key appears in separate AnnData mappings.

Component arrays in `.varm` are variables by components. Masked variables
receive `NaN`, an intentional difference from Scanpy's zero-filled excluded
loadings.

## Mutation and copying

The default `copy=False` writes into the supplied AnnData object and returns
`None`:

```python
scp.residual_pca(adata, layer="counts")
```

With `copy=True`, the function returns a modified copy and leaves the original
unchanged:

```python
result_adata = scp.residual_pca(adata, layer="counts", copy=True)
```

`copy=False` works on a `backed="r"` object, because `.obsm`, `.varm`, and
`.uns` are ordinary in-memory mappings. Nothing is written back to the file;
save the results yourself if you want them persisted.

Staying backed does not avoid loading the counts: the current backend reads a
backed sparse count matrix into memory for canonicalization and fitting. With
`copy=True`, the returned object is fully in memory.

## Downstream Scanpy use

The default PCA score and component keys are compatible with downstream Scanpy
workflows, so an analysis continues without an intermediate conversion step:

```python
scp.residual_pca(adata, layer="counts", n_comps=50)

sc.pp.neighbors(adata)
sc.tl.leiden(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
```

Scanpy finds those scores by name, under `obsm["X_pca"]`. Two cases defeat
that: a custom `key_added`, where Scanpy does not find them and runs its own
default PCA instead, and a matrix of 50 or fewer variables, where it reads `.X`
directly. Naming the representation avoids both — with the key you actually
wrote to:

```python
sc.pp.neighbors(adata, use_rep="X_pca")
```

The metadata also records the count source, mask resolution, transform
parameters, calculation dtype, solver, and package version for reproducibility.

## Complete example

--8<-- "examples/anndata_workflows.md"
