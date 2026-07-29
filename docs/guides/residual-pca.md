# Residual PCA

Residual PCA compares observed counts with an expected count model, represents
the resulting Pearson or deviance residual matrix implicitly, centers its
selected columns, and computes a truncated SVD.

This guide's [complete example](#complete-example) is
[`examples/residual_pca.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/residual_pca.py).

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
| `"scaled_nb"` | Negative-binomial variance whose per-gene overdispersion scales inversely with cell depth. | Nonnegative `alpha` |

`scaled_nb` is a package-specific label, not a standard method name. The
[transform catalogue](../transforms.md#pearson-and-deviance-residuals) gives
each model's null distribution, variance, and provenance, and derives the
fitted mean that all three share.

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

To estimate `alpha` under this model,
[Yu, Huber, and Vitek (2013)](https://doi.org/10.1093/bioinformatics/btt143)
take per-gene method-of-moments estimates and then shrink them toward a common
value. That procedure targets the same inverse-size-factor dispersion
parameterization used here, though it derives its size factors differently.

Do not reuse an SCTransform `theta` as `1 / alpha`. Those are dispersions of a
different negative-binomial model, and the two parameterizations coincide only
when every cell has the same total count.

## Pearson or deviance residuals

Pearson residuals divide the observed-minus-expected difference by the model
standard deviation. Deviance residuals use the signed square root of each
entry's contribution to model deviance. Both are supported for all residual
models.

## Variable selection

If `adata.var["highly_variable"]` exists, it is used by default. Override the
selection with a column name or boolean array, or explicitly pass `None` to use
all genes:

```python
scp.residual_pca(adata, mask_var=None)
scp.residual_pca(adata, mask_var="my_gene_mask")
scp.residual_pca(adata, mask_var=my_boolean_array)
```

The mask chooses which genes enter PCA. It does not change which genes the
normalization was fitted on: cell totals and gene proportions still come from
every gene in the count matrix.
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

`loadings` is exactly `components.T`, kept for convenience when writing
variable-oriented output. It is not a variance-weighted statistical loading.

The matrix path does not apply a variable mask. Slice the count matrix first,
or use the two-step [`transform`](transform-reuse.md) workflow when the
normalization universe and PCA variables must differ.

## Relationship to SCTransform

The scaled-NB residual transform is related to the Pearson-residual
normalization introduced with
[SCTransform by Hafemeister and Satija (2019)](https://doi.org/10.1186/s13059-019-1874-1),
but is not an implementation of it. At positive overdispersion, its Pearson
residuals are identical to SCTransform's only when every cell has the same
total count, with additional model-fitting and post-processing choices matched.
(At `alpha = 0` both models reduce to Poisson, where matching fitted means
suffice and equal depth is not required.) Equal cell depth is not expected in
ordinary real data and is believed to be required for SCTransform residuals to
retain this package's efficient sparse-plus-low-rank representation. See the
[compatibility reference](../reference/compatibility.md) for the full
conditions before making parity claims.

## Complete example

--8<-- "examples/residual_pca.md"
