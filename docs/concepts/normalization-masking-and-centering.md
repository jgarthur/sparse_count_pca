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

## Centering follows variable selection

PCA column means are computed from the transformed columns that survive the
mask. Reusing one `TransformedMatrix` with several masks therefore reuses the
normalization state but calculates a distinct centered operator for each PCA
selection.

Clipping, when enabled, occurs before this centering step.

## Correspondence analysis is different

In correspondence analysis, a variable mask defines the contingency table
being analyzed. The implementation applies the mask first, then recomputes row
totals, column totals, masses, expected values, and inertia:

This can look surprising because Poisson Pearson-residual PCA and classical CA
both construct expected counts from row and column margins. The distinction is
what the analysis holds fixed. Residual PCA treats the full-matrix margins as
fitted normalization state and uses `mask_var` for downstream PCA feature
selection. CA treats the selected contingency table as the object of analysis,
so changing its columns necessarily changes its margins, masses, and inertia.

```text
select X, layer, or raw counts
        ↓
apply mask_var to define the table
        ↓
fit table margins and residual representation
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

The exact default-mask resolution and metadata schema are specified in the
[variable masking](../development/specification.md#variable-masking) and
[output keys](../development/specification.md#output-keys) contracts.
