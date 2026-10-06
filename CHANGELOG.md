# Changelog

All notable user-facing changes are recorded here.

## Unreleased

### New features

- Add `share_structure=False` to `transform`. With `share_structure=True`, the
  fitted transform may reuse the count matrix's CSR indices and row pointers
  instead of copying them; the caller must then not modify that sparsity
  structure in place (for example with `sort_indices()` or `eliminate_zeros()`)
  while the transform is in use. Transformed values, fitted factors, and
  operators returned by `pca` remain independent copies.

### Performance

- PCA setup is faster, most for normalized `log1p`, whose variance statistics
  are now computed directly from the sparse data.
- Count validation and log, CLR, and Dirichlet fitting process arithmetic in
  blocks, reducing temporary memory, especially with `dtype="float32"`
  storage. Total memory still depends on input and transformed storage, fitted
  state, and any required copies.
- One-step analyses that do not return an operator reuse the count matrix's
  sparsity structure for log, CLR, Dirichlet, unclipped Pearson residual, and
  correspondence analyses instead of copying it. With `return_operator=True`
  the returned operator still owns its arrays.

### Fixes

- Count totals are accumulated in float64 for every input dtype. With SciPy
  1.17.1, gene totals of float32 count matrices were accumulated in float32;
  for a highly expressed gene across millions of cells the error reached about
  1e-4 relative. Residual PCA and correspondence analysis of float32 counts,
  which use gene totals, can therefore differ from 1.0.0 by more than the last
  few digits on large datasets. Log, CLR, and Dirichlet transforms use only
  cell totals, which were already exact for realistic counts.
- PCA results, means, and variance statistics may differ from 1.0.0 in their
  last few digits because sums are accumulated in a different order. They
  remain within the documented float32 and float64 tolerances.

## 1.0.0 - 2026-10-01

Changes since 1.0.0rc1:

### Breaking changes

- **Residual clipping is enabled by default**, using `clip="seurat"`
  (`sqrt(n_obs / 30)`). Pass `clip=None` to recover the previous unclipped
  behavior. The symmetric default can trigger the existing support-growth
  guard; `clip_mode` and `clip_max_nnz_ratio` are unchanged.
- **Stricter validation:** composition-scale shifted CLR now raises `ValueError`
  when multiplying the shift by a cell's total count overflows float64, instead
  of silently collapsing transformed nonzero counts to zero.

### New features

- Add size-factor-normalized `log1p` PCA with a median-depth default, explicit
  target totals, or supplied size factors, including factors stored in AnnData
  observation columns.
- Add named clipping thresholds: `"seurat"` (`sqrt(n_obs / 30)`) and `"scanpy"`
  (`sqrt(n_obs)`), resolved against the fitted matrix's observation count.
  Names select only the threshold; positive floats and `None` remain supported.
- Record the resolved numeric clipping threshold as `clip_threshold` in
  transform and PCA metadata, alongside the request in `clip`.
- Identify PCA and correspondence-analysis results with
  `package_name="sparse-count-pca"` alongside `package_version` in metadata.

### Performance and fixes

- **Lower memory use:** substantially reduce peak memory use for large sparse
  inputs and when constructing clipped residual transforms.
- Improve numerical accuracy when computing variance and inertia.

### Documentation

- Add a log1p normalization guide and worked example, and expand `materialize()`
  examples for its supported selectors.
- Clarify transform formulas, shift scales, and overdispersion parameters across
  the public API, including how PFlog's dataset-wide `alpha` differs from
  per-gene `scaled_nb` overdispersion and how PFlog relates to `cleartools`.

## 1.0.0rc1 - 2026-08-05

- Provide exact sparse-plus-low-rank representations for residual, shifted-CLR,
  composition-shifted-CLR, and Dirichlet-pseudocount transforms.
- Compute PCA from implicit transforms without materializing the dense matrix.
- Provide classical and scaled-negative-binomial correspondence analysis.
- Support one-step matrix and AnnData APIs plus reusable fitted transforms.
- Validate count inputs, masks, clipping behavior, precision, and degenerate
  margins with explicit public contracts.
- Document compatibility boundaries and verify results against dense formulas
  and pinned external references.
