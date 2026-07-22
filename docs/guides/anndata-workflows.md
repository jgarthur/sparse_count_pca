# AnnData workflows

The AnnData APIs select counts, resolve PCA variables, run an implicit
analysis, and write results using Scanpy-compatible locations.

## Select the count matrix

Use exactly one source:

```python
scp.residual_pca(adata)                  # adata.X
scp.residual_pca(adata, layer="counts")  # adata.layers["counts"]
scp.residual_pca(adata, use_raw=True)    # adata.raw.X
```

`layer` and `use_raw=True` are mutually exclusive. Input arrays must contain
finite, nonnegative counts. Floating-point inputs are accepted when their
values are integer-like within the package tolerance.

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
string, and nullable masks are rejected instead of being coerced.

For residual, log, and CLR PCA, the mask changes the PCA variables but not the
normalization universe. For correspondence analysis, it defines a new table
and new margins. See
[normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

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

For backed AnnData, `copy=True` returns a fully in-memory object. With
`copy=False`, the AnnData object remains backed, but the current backend loads
a backed sparse count matrix into memory for canonicalization and fitting.

## Downstream Scanpy use

The default PCA score and component keys are compatible with downstream Scanpy
workflows such as neighbors and UMAP. The metadata also records the count
source, mask resolution, transform parameters, calculation dtype, solver, and
package version for reproducibility.
