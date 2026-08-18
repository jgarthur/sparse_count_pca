# Changelog

All notable user-facing changes are recorded here.

## Unreleased

- **Breaking:** residual transforms now clip by default. `clip` accepts the
  named thresholds `"seurat"` (`sqrt(n_obs / 30)`, the new default) and
  `"scanpy"` (`sqrt(n_obs)`) alongside a positive float and `None`, resolving
  the name against the observation count of the fitted matrix. A name selects
  only the threshold; `clip_mode` and `clip_max_nnz_ratio` are unchanged, so
  the symmetric default can trigger the support-growth guard under default
  arguments. Pass `clip=None` to recover the previous unclipped behavior.
- Record the resolved numeric clipping threshold as `clip_threshold` in
  transform and PCA metadata, alongside the request in `clip`.
- Add size-factor-normalized `log1p` PCA with median-depth or explicit target
  totals, supplied size factors, AnnData observation-key alignment, and an
  exact rank-zero sparse representation verified against Scanpy. Composition-
  scale shifted CLR now also rejects row-divisor overflow instead of silently
  collapsing transformed nonzero counts to zero.
- Build every residual family's sparse correction in bounded support blocks
  and write it directly in the requested calculation dtype, substantially
  reducing construction peak RSS when clipping is enabled.
- Clarify parameter documentation across the public API: state each transform's
  formula where it is short, say that shifts are additive and on which scale,
  and record that `alpha` is the overdispersion itself rather than its inverse.
- Describe PFlog's `alpha` as a dataset-wide overdispersion under a
  negative-binomial size-factor model, distinct from the per-gene `scaled_nb`
  `alpha`, and note that `cleartools/scclr` is built on the `runorm` crate,
  which expresses the shift as a proportional-fitting target rather than a
  pseudocount.

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
