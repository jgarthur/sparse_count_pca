# sparse-residual-pca

Implicit spectral analysis of sparse single-cell, spatial-transcriptomics, and
contingency-table count data.

Zero counts generally map to nonzero residuals, so an explicit residual matrix
is dense. This package represents transformed matrices exactly as a sparse
matrix plus a low-rank term and runs truncated SVD without materializing the
dense result.

Supports both Pearson and deviance residuals for the following models:

- Poisson
- Binomial
- [Negative binomial with size-factor-scaled dispersion](SPEC.md#supported-residuals).
  This is similar but not identical to the residual transform used by
  SCTransform.

It also supports shifted CLR PCA and classical correspondence analysis.

Python 3.10 or newer is required because the package uses Python 3.10 type
annotation syntax. The package depends on NumPy, SciPy, AnnData, and
scikit-learn.

## AnnData API

The primary API writes Scanpy-compatible PCA outputs:

```python
import sparse_residual_pca as srp

srp.residual_pca(
    adata,
    layer="counts",
    n_comps=50,
    model="poisson",
    residual="pearson",
)
```

Counts can come from `adata.X` (default), or users can specify `layer=...`, or `use_raw=True`.

The AnnData selection, masking, copy, and default output conventions follow
`scanpy.pp.pca` as of Scanpy 1.12.1 where applicable. Residual-model and
clipping arguments are specific to this package. For backed inputs, `copy=True`
loads the AnnData object into memory before computation.

### Variable selection

By default, `adata.var["highly_variable"]` is used when present; otherwise all
genes are used. Passing `mask_var=None` explicitly disables highly variable
gene filtering:

```python
srp.residual_pca(adata, mask_var=None)                 # all genes
srp.residual_pca(adata, mask_var="my_gene_mask")       # adata.var column
srp.residual_pca(adata, mask_var=my_boolean_array)
```

Masked-out genes receive `NaN` loadings. This is an intentional difference
from Scanpy, making excluded genes distinguishable from valid zero loadings.
Cell totals and gene proportions are always estimated from the full selected
count matrix before the gene mask is applied.

When `key_added="foo"`, scores, loadings, and metadata are all written under
the exact key `"foo"`, matching Scanpy.

### Scaled negative binomial

The scaled-NB model requires nonnegative per-gene overdispersion values:

```python
srp.residual_pca(
    adata,
    model="scaled_nb",
    residual="deviance",
    alpha="overdispersion",  # column in adata.var
)
```

`alpha` may also be a scalar or an array of length `adata.n_vars`. Values below
`1e-8` fall back to the Poisson model.

## Sparse matrix API

Use `residual_pca_matrix` for a CSR-compatible count matrix outside of an `AnnData` object:

```python
result = srp.residual_pca_matrix(
    X,
    n_comps=20,
    model="binomial",
    residual="deviance",
)

result.scores                    # (n_obs, n_comps)
result.components                # (n_comps, n_vars)
result.loadings                  # (n_vars, n_comps)
result.singular_values
result.explained_variance
result.explained_variance_ratio
```

Pass `return_operator=True` to expose the centered `scipy.sparse.linalg.LinearOperator` used by ARPACK.
The representation defaults to `float64`; pass `dtype="float32"` explicitly
for a lower-memory approximate mode. Other representation dtypes are rejected.

## Shifted CLR PCA

The shifted CLR implementation follows the definition called PFlogPF by
Booeshaghi et al. For cell depth `n_i`, feature proportion
`u_ij = x_ij / n_i`, and positive pseudocount `c`, it analyzes

```text
log(u_ij + c) - mean_j(log(u_ij + c)).
```

The default is `c=1`. The equivalent sparse values are
`log1p(x_ij / (c * n_i))`; within-cell centering is represented by a rank-one
term.

```python
result = srp.shifted_clr_pca_matrix(X, n_comps=20, pseudocount=1.0)
srp.shifted_clr_pca(adata, n_comps=20, layer="counts")
```

For AnnData, shifted CLR normalization uses all input genes and `mask_var` is
applied afterward to choose genes entering PCA.

## Correspondence analysis

Classical correspondence analysis decomposes the standardized residual matrix
without ordinary PCA column centering and reports canonical row and column
coordinates:

```python
result = srp.correspondence_analysis_matrix(X, n_comps=2)
result.row_principal_coordinates
result.column_principal_coordinates
result.principal_inertias
result.inertia_ratio

srp.correspondence_analysis(adata, n_comps=2, layer="counts")
```

When a variable mask is used, correspondence-analysis margins are recomputed
from the selected contingency table. Total inertia equals Pearson
`chi_squared / grand_total`.

## Clipping

`clip=None` performs no clipping. A positive finite threshold enables one of
two exact clipping modes:

- `clip_mode="symmetric"` (default) clips to `[-clip, clip]`.
- `clip_mode="upper"` clips only the upper tail to `clip`.

Symmetric clipping can affect zero-count residuals and therefore expand the
sparse correction matrix. `clip_max_nnz_ratio` limits the resulting support
growth and defaults to `2.0`; clipping raises when the ratio meets or exceeds
that limit. Set it to `1.0` to reject any support growth or `None` to allow
unlimited exact expansion.

Upper clipping leaves all negative residuals unchanged. Since zero-count
residuals are negative for every supported model, it never expands sparse
support. See the [clipping details](SPEC.md#clipping).

## Input constraints

- Inputs must be finite, nonnegative counts.
- Inputs must have a real numeric dtype and be safely representable in
  `float64` for validation and residual computation.
- Floating-point inputs are checked for integer-like values with zero relative
  tolerance and an absolute tolerance of `1e-8` by default.
- Variable masks must have genuinely boolean dtype; numeric, string, and
  nullable masks are rejected rather than coerced.
- Cells with zero total counts and selected genes with zero total counts are
  rejected with `ValueError`.
- A transformed matrix with numerically zero centered variance is rejected
  before ARPACK because its PCA directions are undefined.
- Only `solver="arpack"` is supported.
- `n_comps` must be smaller than both the number of cells and the number of
  selected genes.

## Possible future additions

- Highly variable gene selection based on residual variance
- Arbitrary size factors, e.g., from scran
- Support for Dask backed arrays

## Development setup

```bash
uv sync --extra test
uv run pytest
```
