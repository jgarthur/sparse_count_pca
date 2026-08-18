# Normalization, masking, and centering

Variable selection can mean two different things: defining the data used to
fit normalization, or choosing the columns decomposed by PCA. The package keeps
those roles separate.

## The PCA transform order

For residual, count-scale shifted-CLR, and composition-scale shifted-CLR
transforms, the order is:

```text
select the count matrix (`.X` or a layer)
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
normalization state. Use the first to remove genes as a preprocessing step:
subsetting drops them from `.X` and every layer used for the analysis, so they
no longer reach the normalization at all.

## Centering follows variable selection

PCA column means are computed from the transformed columns that survive the
mask. Reusing one `TransformedMatrix` with several masks therefore reuses the
normalization state but calculates a distinct centered operator for each PCA
selection.

Residual clipping occurs at fit time, before this centering step.

## Correspondence analysis is different

In correspondence analysis, a variable mask defines the count table being
analyzed. The implementation applies the mask first, then recomputes cell
totals, gene totals, the expected count for each cell-gene entry under an
independence model, and the relative weights of cells and genes.

Poisson Pearson-residual PCA and classical CA analyze residuals against the
same independence expectation — each cell's total multiplied by each gene's
overall proportion — so the difference lies in what stays fixed. Residual PCA
estimates those quantities once from the full input; CA treats the selected
table as the complete dataset, so removing a gene can change the totals,
weights, and coordinates of every remaining cell and gene.

```text
select the count matrix (`.X` or a layer)
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

Both live under the `params` entry, so read them as
`adata.uns["pca"]["params"]["normalization_n_vars"]`, or
`result.params["normalization_n_vars"]` from the matrix API. Variance
statistics sit one level up, directly under `adata.uns["pca"]`.

Masked component rows in `adata.varm` contain `NaN`. This makes exclusion
visible and prevents a masked variable from being mistaken for a valid zero
component value.

For the exact default-mask resolution and metadata schema, see
[variable masking](../development/specification.md#variable-masking) and
[output keys](../development/specification.md#output-keys) in the package
specification.
