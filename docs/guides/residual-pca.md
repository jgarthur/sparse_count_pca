# Residual PCA

Residual PCA compares observed counts with an expected count model, represents
the resulting Pearson or deviance residual matrix implicitly, centers its
selected columns, and computes a truncated SVD.

## AnnData workflow

```python
import sparse_count_pca as scp

scp.residual_pca(
    adata,
    layer="counts",
    n_comps=50,
    model="poisson",
    residual="pearson",
)
```

Observations are rows and genes are columns. Counts can come from `adata.X`, a
layer, or `adata.raw.X`; see [AnnData workflows](anndata-workflows.md).

## Choose the count model

| `model` | Interpretation | Additional input |
| --- | --- | --- |
| `"poisson"` | Count variance equals the expected count. | None |
| `"binomial"` | Counts are draws from each cell total with gene probability `p_j`. | None |
| `"scaled_nb"` | Negative-binomial variance with per-gene overdispersion and exposure-scaled means. | Nonnegative `alpha` |

The fitted cell totals and gene proportions use every gene in the chosen count
matrix before the PCA variable mask is applied. A selected gene with zero total
count is invalid because its residual scale is undefined.

For `model="scaled_nb"`, pass a scalar, a vector of length `n_vars`, or an
`adata.var` column name:

```python
scp.residual_pca(
    adata,
    layer="counts",
    model="scaled_nb",
    residual="deviance",
    alpha="overdispersion",
)
```

Values of `alpha` below `1e-8` use the Poisson limit. The package consumes
overdispersion estimates but does not fit them.

## Pearson or deviance residuals

Pearson residuals divide the observed-minus-expected difference by the model
standard deviation. Deviance residuals use the signed square root of each
entry's contribution to model deviance. Both are supported for all residual
models.

Choose the residual definition based on the scientific analysis rather than
computational convenience: both have exact sparse-plus-low-rank
representations in this package.

## Variable selection

If `adata.var["highly_variable"]` exists, it is used by default. Override the
selection with a column name or boolean array, or explicitly pass `None` to use
all genes:

```python
scp.residual_pca(adata, mask_var=None)
scp.residual_pca(adata, mask_var="my_gene_mask")
scp.residual_pca(adata, mask_var=my_boolean_array)
```

Masking affects the PCA variables, not the fitted normalization universe.
Masked variables receive `NaN` component values in `adata.varm`, making them
distinguishable from valid zero values. See
[normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

## Clipping

Residuals are not clipped unless `clip` is provided:

```python
scp.residual_pca(
    adata,
    layer="counts",
    clip=10.0,
    clip_mode="symmetric",
)
```

Clipping occurs on uncentered residuals before PCA column centering. Symmetric
clipping can change zero-count entries and expand sparse support; upper-only
clipping cannot. See [clipping and precision](../concepts/clipping-and-precision.md).

## Matrix workflow

Use the matrix entry point outside AnnData:

```python
result = scp.residual_pca_matrix(
    counts,
    n_comps=20,
    model="poisson",
    residual="deviance",
)

result.scores       # observations by components
result.components   # components by variables
result.loadings     # variables by components
```

The matrix path does not apply a variable mask. Slice the count matrix first,
or use the two-step [`transform`](transform-reuse.md) workflow when the
normalization universe and PCA variables must differ.

## Relationship to SCTransform

The scaled-NB residual transform is related to, but is not generally identical
to, default SCTransform v2 output. Agreement requires controlled parameter,
variance-floor, clipping, and centering conditions. See the
[compatibility reference](../reference/compatibility.md) before making parity
claims.
