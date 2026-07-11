# sparse-count-pca

PCA and related spectral analysis of sparse single-cell,
spatial-transcriptomics, and contingency-table count data.

Count transforms often map structural zeros to nonzero values, making the
transformed matrix dense. This package represents supported transforms exactly
as a sparse matrix plus a low-rank term and runs truncated SVD without
materializing the dense result.

Supports both Pearson and deviance residuals for the following models:

- Poisson
- Binomial
- [Negative binomial with size-factor-scaled dispersion](docs/specification.md#supported-residuals).
  This is similar but not identical to the residual transform used by
  SCTransform.

It also supports fixed-count shifted-log and shifted-CLR PCA,
fixed-composition shifted-CLR PCA, Dirichlet-log and Dirichlet-CLR PCA, and
classical correspondence analysis. See the [documentation index](docs/index.md)
or start with the source-linked [architecture review](docs/architecture.md).

Python 3.10 or newer is required because the package uses Python 3.10 type
annotation syntax. The package depends on NumPy, SciPy, AnnData, and
scikit-learn.

## AnnData API

The primary API writes Scanpy-compatible PCA outputs:

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
scp.residual_pca(adata, mask_var=None)                 # all genes
scp.residual_pca(adata, mask_var="my_gene_mask")       # adata.var column
scp.residual_pca(adata, mask_var=my_boolean_array)
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
scp.residual_pca(
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
result = scp.residual_pca_matrix(
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

## Shifted-log and shifted-CLR PCA

Shift domains are explicit and shift arguments are required.

`shifted_log` applies the exactly sparse transform
`log1p(x_ij / count_shift)`. It has the same column-centered PCA as
`log(x_ij + count_shift)`:

```python
log_result = scp.shifted_log_pca_matrix(
    X,
    n_comps=20,
    count_shift=1.0,
)
```

`shifted_clr` uses a fixed raw-count shift and subtracts the within-cell gene
mean:

```text
clr(x_i + count_shift) =
    log(x_ij + count_shift) - mean_j(log(x_ij + count_shift)).
```

Current PFlog is obtained with `count_shift=1 / (4 * alpha)`:

```python
clr_result = scp.shifted_clr_pca_matrix(
    X,
    n_comps=20,
    count_shift=1 / (4 * alpha),
)
scp.shifted_clr_pca(
    adata,
    n_comps=20,
    count_shift=1 / (4 * alpha),
    layer="counts",
)
```

`proportion_shifted_clr` is the distinct historical formula with a fixed shift
after library-size division:

```text
clr(x_i / cell_total_i + composition_shift).
```

```python
historical = scp.proportion_shifted_clr_pca_matrix(
    X,
    n_comps=20,
    composition_shift=1.0,
)
```

Its effective count shift is `cell_total_i * composition_shift`; unlike the
fixed-count method, it rejects empty cells. All CLR row means are computed over
the full input gene universe before `mask_var` selects genes for PCA.

## Dirichlet PCA transforms

For total prior concentration `A` and strictly positive prior proportions
`p_j`, the Dirichlet posterior-mean composition is

```text
q_ij = (x_ij + A p_j) / (n_i + A).
```

`dirichlet_log` analyzes `log(q_ij)`. `dirichlet_clr` additionally subtracts
the within-cell mean of `log(q_ij)`. Both dense transforms have exact
sparse-plus-rank-at-most-two representations. The default is `A=1` with a
uniform prior, meaning one total prior count distributed as `1 / n_vars` per
gene.

```python
log_result = scp.dirichlet_log_pca_matrix(
    X,
    n_comps=20,
    concentration=1.0,
)
clr_result = scp.dirichlet_clr_pca_matrix(
    X,
    n_comps=20,
    concentration=2.0,
    prior_proportions=prior,
)

scp.dirichlet_log_pca(adata, n_comps=20, layer="counts")
scp.dirichlet_clr_pca(
    adata,
    n_comps=20,
    prior_proportions="prior_proportion",  # column in adata.var
)
```

The prior and posterior normalization use all input genes before `mask_var`
selects genes for PCA. Dirichlet transforms and fixed-count shifted transforms
permit cells with zero observed counts.

## Correspondence analysis

Classical correspondence analysis decomposes the standardized residual matrix
without ordinary PCA column centering and reports canonical row and column
coordinates:

```python
result = scp.correspondence_analysis_matrix(X, n_comps=2)
result.row_principal_coordinates
result.column_principal_coordinates
result.principal_inertias
result.inertia_ratio

scp.correspondence_analysis(adata, n_comps=2, layer="counts")
```

The AnnData API writes row principal coordinates to `adata.obsm["X_ca"]`,
column principal coordinates to `adata.varm["CA"]`, and metadata to
`adata.uns["ca"]`. Passing `key_added="foo"` uses `"foo"` for all three keys.

The result stores principal coordinates, which are the coordinates used by the
AnnData API. Standard coordinates are not stored; when needed, derive them as
`principal_coordinates / singular_values`.

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
support. See the [clipping details](docs/specification.md#clipping).

## Input constraints

- Inputs must be finite, nonnegative counts.
- Inputs must have a real numeric dtype and be safely representable in
  `float64` for validation and residual computation.
- Floating-point inputs are checked for integer-like values with zero relative
  tolerance and an absolute tolerance of `1e-8` by default.
- Variable masks must have genuinely boolean dtype; numeric, string, and
  nullable masks are rejected rather than coerced.
- Cells with zero total counts are supported by Dirichlet and fixed-count log
  transforms. Residual, correspondence, and proportion-shifted transforms
  reject them where their normalization is undefined.
- A transformed matrix with numerically zero centered variance is rejected
  before ARPACK because its PCA directions are undefined.
- Only `solver="arpack"` is supported.
- `n_comps` must be smaller than both the number of cells and the number of
  selected genes.

## Possible future additions

- Highly variable gene selection based on residual variance
- Arbitrary size factors, e.g., from scran
- Support for Dask backed arrays
- Block-wise reconstruction from retained PCs through the inverse transform to
  approximate count space. Truncated PCA may yield negative reconstructed
  counts; the package should preserve them and leave clipping to consumers.

## Development setup

```bash
uv sync --extra test
uv run python -m pytest
uv run ruff check .
```
