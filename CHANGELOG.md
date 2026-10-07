# Changelog

All notable user-facing changes are recorded here.

## Unreleased

### Performance

- Faster PCA setup, especially for normalized `log1p`, whose variance
  statistics are computed directly from sparse data.
- Count validation and log, CLR, and Dirichlet fitting process arithmetic in
  blocks, reducing temporary memory, especially with `dtype="float32"`
  storage.
- One-step analyses use less memory by reusing the count matrix's sparsity
  structure when `return_operator=False`. Symmetric clipping of negative
  residuals at zero counts requires new sparsity structure, as the default
  `clip="seurat"` can on real data. Returned operators own their arrays.

### New features

- Add `transform(..., share_structure=True)` to let two-step workflows reuse
  the count matrix's sparsity structure and save memory, as one-step analyses
  do automatically. Sharing is opt-in; do not modify the count matrix's
  sparsity structure while the fitted transform is in use.

### Fixes

- More accurate gene totals for float32 counts correct the normalization used
  by residual PCA and correspondence analysis. Synthetic regression tests
  recover `100000000 + 1` as `100000001`, where float32 accumulation lost the
  small contribution, and verify that fitting agrees with an exact float64
  copy of the input. Count totals accumulate in float64 for every input dtype.
  These corrections can change float32-input results beyond the last few
  digits compared with 1.0.0.

### Other changes

- Changed summation order introduces last-digit differences in PCA results,
  means, and variance statistics compared with 1.0.0.

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
