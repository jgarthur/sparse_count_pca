# Wishlist

This file records plausible additions without treating them as committed API.

## Block-wise inverse reconstruction to count space

Provide native reconstruction from a `PCAResult` without materializing the
full observation-by-feature matrix.

The first stage is common to every PCA transform. For an observation block:

```python
transformed = result.scores[rows] @ result.components + result.mean
```

The second stage is transform-specific and may require stored or supplied
normalization state.

### Desired behavior

- Offer a block iterator and/or an `out=` destination rather than returning a
  mandatory dense full matrix.
- Support reconstruction in transformed space independently of count-space
  inversion.
- Preserve negative reconstructed counts. Low-rank approximation can leave
  the valid range; consumers choose whether and how to clip.
- Make approximation explicit: omitted PCs cannot be recovered.
- Validate that the result contains the state required by its inverse.
- Keep masked-gene behavior explicit. Reconstructing only PCA-selected genes
  is not the same as reconstructing the original full matrix.

### Identifiability by transform

| Transform | Count-space inverse requirements |
| --- | --- |
| shifted log | `count_shift`; direct `count_shift * expm1(z)` |
| shifted CLR | `count_shift` plus original or supplied row total to restore the removed log gauge |
| proportion-shifted CLR | `composition_shift` recovers proportions; original row totals are needed for counts |
| Dirichlet log | concentration, prior proportions, and row totals |
| Dirichlet CLR | prior counts plus row totals to restore the removed log gauge |
| residual PCA | fitted null-model state and a stable analytic or numerical inverse for the selected residual family |
| correspondence analysis | row/column masses and grand total, with semantics distinct from PCA reconstruction |

For fixed-count shifted CLR with reconstructed CLR values `z`, row total `s`,
and `G` genes, the missing log gauge is identifiable from

```text
exp(t) = (s + G * count_shift) / sum(exp(z)).
```

Then the approximate counts are `exp(z + t) - count_shift`. Analogous closure
constraints apply to Dirichlet CLR and proportion-shifted CLR.

### Likely API shape

```python
result.reconstruct_transformed(block_size=..., out=...)

inverse_transform_counts(
    result,
    *,
    row_totals=...,
    block_size=...,
    out=...,
)
```

A standalone inverse dispatcher may be cleaner than putting transform-specific
logic on `PCAResult`. Decide after the metadata/state design is reviewed.

## Other candidates

- Split the large package specification into focused API, numerical, and
  verification references once maintaining the single normative document
  becomes cumbersome. Preserve one clearly identified normative entry point
  and stable cross-links when doing so.
- Highly variable gene selection based on residual variance.
- Supplied arbitrary size factors, including scran-derived factors.
- Dask-backed inputs and block-aware transformations.
- Exact negative-binomial Anscombe PCA after dispersion semantics are settled.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
