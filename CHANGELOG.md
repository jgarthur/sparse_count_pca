# Changelog

All notable user-facing changes are recorded here.

## 1.0.0 - 2026-10-28

Changes since 1.0.0rc1:

### Breaking changes

- Python 3.11 or newer is required. Current AnnData and Scanpy releases require
  Python 3.12 or newer, while installing this project's test extra on Python
  3.10 selects a Pillow version with known security vulnerabilities through
  Scanpy's Matplotlib dependency. Python 3.11 supports the patched Pillow
  release while retaining compatibility with earlier AnnData and Scanpy
  releases.

- **Residual clipping is enabled by default**, using `clip="seurat"`
  (`sqrt(n_obs / 30)`). Pass `clip=None` to recover the previous unclipped
  behavior. Because the default clips symmetrically, it raises `RuntimeError`
  when clipping would grow sparse support to `clip_max_nnz_ratio` (default
  `2.0`) times the input's stored entries; for example, pass
  `clip_mode="upper"`, which never grows support. The `clip_mode` and
  `clip_max_nnz_ratio` defaults are unchanged.
- `TransformedMatrix.pca` raises `ValueError` when it must read a `mask_var`
  key or the default `"highly_variable"` column after the AnnData object it was
  fitted from no longer exists. This includes a transform fitted from an
  AnnData that was never assigned to a variable, as in
  `scp.transform(sc.read_h5ad(path), ...)`. Pass a boolean-array `mask_var` or
  `mask_var=None`. The transform does not keep the AnnData in memory.
- **Stricter validation:** composition-scale shifted CLR now raises `ValueError`
  when multiplying the shift by a cell's total count overflows float64, instead
  of silently collapsing transformed nonzero counts to zero.

### New features

- Add `TransformedMatrix.variable_statistics` to compute each gene's mean and
  variance under a fitted transform without materializing it, for example to
  select highly variable genes by Pearson-residual variance. Statistics include
  any residual clipping and default to the `n_obs - 1` denominator used by PCA
  total variance. With `Residual(model="poisson", clip="scanpy")` and `ddof=0`,
  the variances agree with Scanpy's Pearson-residual `residual_variances` for
  `theta=np.inf` in a small synthetic regression test.
- Add size-factor-normalized `log1p` PCA with `log1p_norm_pca`,
  `log1p_norm_pca_matrix`, and the `Log1pNormalized` transform, using a
  median-depth default, explicit target totals, or supplied size factors,
  including factors stored in AnnData observation columns.
- Add named clipping thresholds: `"seurat"` (`sqrt(n_obs / 30)`) and `"scanpy"`
  (`sqrt(n_obs)`), resolved against the fitted matrix's observation count.
  Names select only the threshold; positive floats and `None` remain supported.
- Add `transform(..., share_structure=True)` to let two-step workflows reuse
  the count matrix's sparsity structure and save memory, as one-step analyses
  do automatically. Inputs that need canonicalization or conversion to CSR,
  int32 indices that must widen to int64, and residual clipping that adds
  entries still produce a copy, with a `UserWarning`. Sharing is
  opt-in; do not modify the count matrix's sparsity structure in place, for
  example with `sort_indices()`, while the fitted transform is in use. `pca`
  and `materialize` raise `RuntimeError` if it changed.
- Record the resolved numeric clipping threshold as `clip_threshold` in
  transform and PCA metadata, alongside the request in `clip`.
- Identify PCA and correspondence-analysis results with
  `package_name="sparse-count-pca"` alongside `package_version` in metadata.

### Performance

- Faster PCA setup, especially for normalized `log1p`, whose variance
  statistics are computed directly from sparse data.
- Substantially lower peak memory use for large sparse inputs and when
  constructing clipped residual transforms.
- Count validation and log, CLR, and Dirichlet fitting process arithmetic in
  blocks, reducing temporary memory, especially with `dtype="float32"`
  storage.
- One-step analyses use less memory by reusing the count matrix's sparsity
  structure when `return_operator=False`. Symmetric clipping that changes
  residuals at zero counts, as the default `clip="seurat"` can on real data,
  builds new structure instead.

### Fixes

- Locked test and documentation environments use patched certifi, Pillow,
  nbconvert, Tornado, and urllib3 dependencies.
- More accurate gene totals for float32 counts correct the normalization used
  by residual PCA and correspondence analysis. Synthetic regression tests
  recover `100000000 + 1` as `100000001`, where float32 accumulation lost the
  small contribution, and verify that fitting agrees with an exact float64
  copy of the input. Count totals accumulate in float64 for every input dtype.
  These corrections can change float32-input results beyond the last few
  digits compared with 1.0.0rc1.
- Squared norms behind total variance, explained-variance ratios, and
  correspondence-analysis inertia are recomputed directly when cancellation
  would leave them accurate to only about eight digits.
- `TransformedMatrix.pca` reads `mask_var` keys and the default
  `"highly_variable"` mask from the AnnData object's current `adata.var`, so
  highly variable genes selected after fitting a transform are used. It
  previously read a copy taken at fitting, silently selecting every gene when
  that copy had no `"highly_variable"` column. If the AnnData's genes change
  after fitting, keys raise `ValueError`; boolean-array masks and
  `mask_var=None` still work.
- `TransformedMatrix.materialize` raises `IndexError` for a boolean mask whose
  length does not match its axis, including an empty mask, which previously
  selected nothing, and for scalar-boolean or two-dimensional selectors.

### Other changes

- Changed summation order introduces last-digit differences in PCA results,
  means, and variance statistics compared with 1.0.0rc1.

### Documentation

- Add a log1p normalization guide and worked example, and expand `materialize()`
  examples for its supported selectors.
- Clarify transform formulas, shift scales, and overdispersion parameters across
  the public API, including how PFlog's dataset-wide `alpha` differs from
  per-gene `scaled_nb` overdispersion and how PFlog relates to `cleartools`.
- Add citation metadata in `CITATION.cff` and citation instructions to the
  README and documentation home page; releases are archived on Zenodo with a
  DOI.

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
