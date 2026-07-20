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
  SCTransform. When all cells have the same total and the fitted means agree,
  its Pearson residuals reduce to the usual SCTransform negative-binomial form
  with `alpha = 1 / theta`. Clipping is applied to residuals before PCA column
  centering.

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

Count selection, mask defaults, copy semantics, and output keys follow
`scanpy.pp.pca` as of Scanpy 1.12.1. Residual-model and clipping arguments are
specific to this package, and masked loading values intentionally differ as
described below. For backed inputs, `copy=True` loads the full AnnData object
into memory before computation. With `copy=False`, the AnnData object remains
backed, but a backed sparse count matrix is currently loaded into memory for
canonicalization.

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
For residual PCA, cell totals and gene proportions are always estimated from
the full selected count matrix before the gene mask is applied.

Two metadata fields make those distinct gene counts explicit:

- `normalization_n_vars` is the number of genes in the chosen `X`, layer, or
  raw matrix before `mask_var` is applied.
- `pca_n_vars` is the number of genes actually passed to PCA after masking.

For AnnData results, `pca_n_vars` is also available under
`params["mask_var_details"]["n_vars_used"]` for compatibility. These counts can
differ intentionally: normalization is fitted on `normalization_n_vars`, then
PCA decomposes only `pca_n_vars`. In CLR transforms, even an all-zero gene in
the normalization universe changes the row-centering denominator, so removing
a gene before the call is not equivalent to masking it out of PCA.

The arrays stored in `.varm` are `components.T`, following Scanpy's convention.
They are sometimes called loadings, but they are not variance-weighted
statistical loadings.

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

## Two-step transform API

Normalization and PCA can be separated when transformed values need to be
inspected or several PCA selections will reuse one normalization:

```python
transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
)

cell = transformed.materialize(obs=10)
gene = transformed.materialize(var=25)
subset = transformed.materialize(obs=[10, 3], var=[25, 2, 8])
result = transformed.pca(n_comps=50, mask_var="highly_variable")
```

`TransformedMatrix` is an uncentered `scipy.sparse.linalg.LinearOperator`.
It remains implicit until `materialize` is called, and PCA applies column
centering only after `mask_var` selects variables. Transform specifications are
`Residual`, `ShiftedLog`, `ShiftedCLR`, `ProportionShiftedCLR`, `DirichletLog`,
and `DirichletCLR`. Correspondence analysis remains a separate one-step API
because its variable mask changes the contingency-table margins.

`obs` and `var` accept integers, slices, boolean masks, ordered index arrays,
and, for AnnData input, observation or variable names. Integer selections keep
a two-dimensional result. Observation blocks are efficient with the current
CSR backend. Selecting variables across all observations emits a
`SparseEfficiencyWarning`; this access direction can require scanning CSR rows
whether the source was originally backed or in memory.

Use `out=` and `block_size=` to bound temporary memory while materializing:

```python
import numpy as np

out = np.memmap(path, mode="w+", shape=transformed.shape, dtype="float32")
transformed.materialize(out=out, block_size=2048)
```

The transformed object exists only in the current Python process and is not
stored by `write_h5ad`. Backed sparse inputs are currently materialized during
count canonicalization; the operator and block-materialization interface is
intended to admit a future out-of-core backend without changing this public API.

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

Pass `return_operator=True` to expose the centered
`scipy.sparse.linalg.LinearOperator` used by ARPACK. One-step APIs transfer
their private representation to that operator without another CSR copy. An
operator returned by `TransformedMatrix.pca` is instead isolated from the
still-live fitted transform and therefore requires a support-sized copy.
The representation defaults to `float64`, which is recommended for numerical
precision. Passing `dtype="float32"` is a lower-memory approximate mode: it
casts the representation and the operator passed to ARPACK, so it lowers the
precision of the PCA calculation rather than only the precision of returned
arrays. Other representation dtypes are rejected.

To calculate in `float64` but store selected outputs more compactly, downcast
them only after PCA. For example:

```python
scores32 = result.scores.astype("float32")
loadings32 = result.loadings.astype("float32")

# Or, after an AnnData call:
adata.obsm["X_pca"] = adata.obsm["X_pca"].astype("float32")
adata.varm["PCs"] = adata.varm["PCs"].astype("float32")
```

Post-computation downcasting leaves the completed decomposition and its
float64 variance statistics unchanged, but downstream calculations using the
downcast arrays have float32 precision.

Canonical zero-free CSR inputs are borrowed without copying. If a caller-owned
CSR contains duplicate or unsorted indices or explicitly stored zeros, it is
copied before canonicalization and a `UserWarning` explains why. Other sparse
formats necessarily allocate a CSR during conversion. Fitting a transform
still allocates transformed sparse values and takes one owned snapshot of CSR
support, so later mutation of the input cannot alter the fitted matrix.

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

Its effective count shift is `cell_total_i * composition_shift`. All CLR row
means are computed over the full input gene universe before `mask_var` selects
genes for PCA.

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

### Experimental scaled-NB residual ordination

Passing `model="scaled_nb"` replaces the Poisson variance with the package's
size-factor-scaled negative-binomial variance and analyzes the resulting
Pearson residual matrix divided by `sqrt(grand_total)`:

```python
experimental = scp.correspondence_analysis_matrix(
    X,
    n_comps=2,
    model="scaled_nb",
    alpha=overdispersion,
)
scp.correspondence_analysis(
    adata,
    n_comps=2,
    model="scaled_nb",
    alpha="overdispersion",
    layer="counts",
)
```

This mode is a correspondence-like residual ordination, not classical
correspondence analysis. It emits a warning, records `experimental=True` in
the result parameters, and requires `alpha` with the same meaning as
scaled-NB residual PCA. Row and column coordinates continue to use the
observed count masses. However, `total_inertia` is scaled-NB Pearson residual
inertia rather than `chi_squared / grand_total`, and the full chi-square and
barycentric interpretations of classical correspondence analysis do not
apply.

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
- Cells with zero total counts are rejected consistently before transform
  fitting. Empty observations would otherwise participate in PCA centering and
  covariance even for transforms that can assign them finite values.
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
