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

Observations are rows and genes are columns. Counts can come from `adata.X` or
a layer; see [AnnData workflows](anndata-workflows.md).

## Choose the count model

| `model` | Interpretation | Additional input |
| --- | --- | --- |
| `"poisson"` | Count variance equals the expected count. | None |
| `"binomial"` | Counts are draws from each cell total with gene probability `p_j`. | None |
| `"scaled_nb"` | Negative-binomial variance whose per-gene overdispersion scales inversely with cell depth. | Nonnegative `alpha` |

`scaled_nb` is a package-specific label, not a standard method name. The
[transform catalog](../transforms.md#pearson-and-deviance-residuals) gives
each model's null distribution, variance, and provenance, and derives the
fitted mean that all three share.

The fitted cell totals and gene proportions use every gene in the chosen count
matrix before the PCA variable mask is applied. A gene with zero total count is
kept, not dropped: its fitted mean is zero, so its residual is zero and it
carries no variance. Its coefficient therefore comes back as zero, every other
gene's result is untouched, and the count is reported under
`adata.uns["pca"]["params"]["n_empty_vars"]`, or `result.params["n_empty_vars"]`
from the matrix API. You can pass a matrix straight from a cell subset without
re-filtering genes.

Empty genes do not count toward the `n_comps` limit, since they carry no
variance. A cell subset that leaves many genes empty can therefore admit fewer
components than its column count suggests; the error names the usable count if
you ask for too many.

Empty *cells* are a different matter and are rejected outright. Remove
zero-total rows from the same matrix or layer before calling.

For `model="scaled_nb"`, pass a scalar, a vector of length `n_vars`, or an
`adata.var` column name:

```python
scp.residual_pca(
    adata,
    layer="counts",
    model="scaled_nb",
    residual="pearson",
    alpha="overdispersion",
)
```

`alpha` is the overdispersion itself, not its inverse: gene `j` contributes
variance `mu_ij * (1 + alpha_j * mean_n * p_j)`, so larger values mean more
variance and `alpha = 0` is the Poisson limit. Estimates reported on a `size`,
`theta`, or `r` scale must be inverted before they are passed here.

Values of `alpha` below `1e-8` use the Poisson limit. The package consumes
overdispersion estimates but does not fit them;
[Yu, Huber, and Vitek (2013)](https://doi.org/10.1093/bioinformatics/btt143)
estimate dispersions under the same inverse-size-factor parameterization.

Estimates fitted under a different negative-binomial model are not in general
valid here. In particular, do not reuse an SCTransform `theta` as `1 / alpha`:
the two parameterizations coincide only when every cell has the same total
count.

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

Residuals are clipped by default, symmetrically at the named `"seurat"`
threshold \(\sqrt{n_\mathrm{obs}/30}\). Pass `clip` to choose another
threshold, and `clip=None` to disable clipping:

```python
scp.residual_pca(adata, layer="counts")                 # clip="seurat"
scp.residual_pca(adata, layer="counts", clip="scanpy")  # sqrt(n_obs)
scp.residual_pca(adata, layer="counts", clip=10.0)      # explicit number
scp.residual_pca(adata, layer="counts", clip=None)      # unclipped
```

A name selects the threshold only; `clip_mode` is independent of it. Upper-only
clipping at the Seurat threshold combines the two:

```python
scp.residual_pca(
    adata,
    layer="counts",
    clip="seurat",
    clip_mode="upper",
)
```

Clipping occurs on uncentered residuals before PCA column centering. Symmetric
clipping can change zero-count entries and expand sparse support; upper-only
clipping cannot. Because the default is symmetric, a call with default
arguments can raise on the `clip_max_nnz_ratio` guard. See
[clipping and precision](../concepts/clipping-and-precision.md).

## Matrix workflow

Use the matrix entry point outside AnnData:

```python
result = scp.residual_pca_matrix(
    counts,
    n_comps=20,
    model="poisson",
    residual="pearson",
)

result.scores       # observations by components
result.components   # components by variables
result.components.T # variables by components
```

The matrix path does not apply a variable mask. Slice the count matrix first,
or use the two-step [`transform`](transform-reuse.md) workflow when the
normalization universe and PCA variables must differ.

## Relationship to SCTransform

The scaled-NB residual transform is related to the Pearson-residual
normalization introduced with
[SCTransform by Hafemeister and Satija (2019)](https://doi.org/10.1186/s13059-019-1874-1),
but is not an implementation of it. The two agree only under conditions that
ordinary real data does not meet; see the
[compatibility reference](../reference/compatibility.md#sctransform-v2) for the
full list before making parity claims.

The models also differ computationally: in SCTransform's standard
parameterization there is no obvious sparse-plus-low-rank factorization of the
residual matrix.

## Complete example

--8<-- "examples/residual_pca.md"
