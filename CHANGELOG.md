# Changelog

All notable user-facing changes are recorded here.

## 1.0.0 - 2026-10-01

Changes since 1.0.0rc1:

### Breaking changes

- **Residual clipping is enabled by default**, using `clip="seurat"`
  (`sqrt(n_obs / 30)`). Pass `clip=None` to recover the previous unclipped
  behavior. The symmetric default can trigger the existing support-growth
  guard; `clip_mode` and `clip_max_nnz_ratio` are unchanged.
- **Stricter validation:** composition-scale shifted CLR now raises `ValueError`
  when a cell's total count times the shift overflows or rounds to zero in
  float64, or when dividing a count by that product overflows, instead of
  silently producing zero or infinite transformed values.

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
