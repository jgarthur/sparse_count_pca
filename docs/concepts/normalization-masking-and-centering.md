# Normalization, masking, and centering

Variable selection can mean two different things: defining the data used to
fit normalization, or choosing the columns decomposed by PCA. The package keeps
those roles separate.

## The PCA transform order

For residual, shifted-log, shifted-CLR, and proportion-shifted-CLR transforms,
the order is:

```text
select X, layer, or raw counts
        ↓
fit normalization on the full variable universe
        ↓
apply mask_var to choose PCA variables
        ↓
center the selected transformed columns
        ↓
run truncated SVD
```

This ordering means `mask_var` is a PCA mask, not a preprocessing filter.

For residual transforms, cell totals and gene proportions come from the full
chosen count matrix. For CLR transforms, row log means use the full variable
universe.

Consequently, these two operations need not agree:

```python
# Fit normalization after physically removing genes.
scp.residual_pca(adata[:, mask].copy(), mask_var=None)

# Fit normalization on all genes, then use only `mask` for PCA.
scp.residual_pca(adata, mask_var=mask)
```

Use the second form when the omitted variables should still contribute to the
normalization state.

If genes should be removed as a preprocessing step, subset the `AnnData` object
before calling the package:

```python
adata_for_pca = adata[:, keep_genes].copy()
scp.residual_pca(adata_for_pca, layer="counts", mask_var=None)
```

Subsetting this way removes the genes from `.X` and every layer used for the
analysis. The example creates a filtered copy and leaves the original `adata`
unchanged. If the full-gene data are already stored appropriately in `.raw`, the
filtered copy can instead replace the working object. Keeping `.copy()` is still
recommended because slicing an `AnnData` object produces a view.

## Centering follows variable selection

PCA column means are computed from the transformed columns that survive the
mask. Reusing one `TransformedMatrix` with several masks therefore reuses the
normalization state but calculates a distinct centered operator for each PCA
selection.

Clipping, when enabled, occurs before this centering step.

## Correspondence analysis is different

In correspondence analysis, a variable mask defines the count table being
analyzed. The implementation applies the mask first, then recomputes cell
totals, gene totals, the expected count for each cell-gene entry under an
independence model, and the relative weights of cells and genes.

This can look surprising because Poisson Pearson-residual PCA and classical CA
both analyze Pearson residuals based on the same independence expectation: each
cell's total multiplied by each gene's overall proportion. The distinction is
what remains fixed. Residual PCA estimates those quantities from the full input
and uses `mask_var` only to select PCA genes. CA treats the selected count table
as the complete dataset, so removing a gene can change the totals, relative
weights, and coordinates of every remaining cell and gene.

```text
select X, layer, or raw counts
        ↓
apply mask_var to define the table
        ↓
compute selected cell totals and gene proportions
        ↓
run uncentered truncated SVD
```

CA does not apply ordinary PCA column centering.

## AnnData metadata

PCA result metadata distinguishes:

- `normalization_n_vars`: variables in the count matrix used to fit
  normalization;
- `pca_n_vars`: variables selected for PCA.

Masked component rows in `adata.varm` contain `NaN`. This makes exclusion
visible and prevents a masked variable from being mistaken for a valid zero
component value.

Readers who need the exact default-mask resolution and metadata schema can
consult the
[variable masking](../development/specification.md#variable-masking) and
[output keys](../development/specification.md#output-keys) sections of the
package specification.
