# Changelog

All notable user-facing changes are recorded here.

## Unreleased

- Compute the operator's stable column means and Frobenius norms from
  column-major stripes rather than one full CSC copy of the sparse part,
  removing a transient the size of the sparse representation from the PCA peak.
  Results are bitwise unchanged.
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
