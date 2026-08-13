# Wishlist

This file records unfinished plausible additions without treating them as
committed API. When work is completed, remove it from this file and record the
outcome in the appropriate specification, design, architecture, or provenance
documentation; do not leave completed work described as a wish.

## After 1.0

### Named residual-clipping thresholds and a clipped default

Accept named `clip` values for the two established cell-count-dependent
thresholds:

```python
clip="seurat"  # sqrt(n_obs / 30)
clip="scanpy"  # sqrt(n_obs)
```

Here, `n_obs` is the number of observations in the count matrix when the
transform is fitted. It therefore reflects cell filtering performed by the
caller, but is independent of gene masking or later PCA variable selection.
Continue to accept a positive finite float as an explicit threshold and
`None` to disable clipping.

The aliases must resolve only the numeric threshold. They must have no effect
on `clip_mode`: `clip_mode="symmetric"` and `clip_mode="upper"` retain their
existing meanings for named and numeric thresholds alike. In particular,
`clip="seurat", clip_mode="upper"` means upper-only clipping at
`sqrt(n_obs / 30)`; the `"seurat"` name must not silently select symmetric
clipping. Make this orthogonality explicit in the API reference, clipping
concept page, docstrings, and examples. The existing `clip_max_nnz_ratio`
guard likewise continues to depend on `clip_mode`, not on how the threshold
was specified.

Probably change the default from `clip=None` to `clip="seurat"`, while keeping
the default `clip_mode="symmetric"`. The lower Seurat threshold is the more
useful general default because extreme residuals do need control; the Scanpy
threshold remains available for compatibility. Treat this as a deliberate
behavior change and document that callers can recover the current unclipped
behavior with `clip=None`.

The implementation and tests should:

- resolve the alias after the input observation axis is finalized and before
  constructing the residual representation;
- record both the requested alias and resolved numeric threshold in result
  metadata so a fitted analysis is reproducible without reinterpreting the
  alias;
- apply the same resolution in the matrix, AnnData, one-step, and two-step
  entry points;
- test both aliases with both clipping modes, explicit numeric thresholds,
  `None`, cell-subset inputs, support-growth guards, and invalid strings; and
- update compatibility documentation that currently says clipping is opt-in.

As a scale check, the combined four-donor PBMC matrix after filtering to cells
with at least 100 detected genes and genes detected in at least three cells has
22,111 observations. Its Seurat threshold is approximately `27.148`; symmetric
Poisson Pearson clipping changes 157,739 positive and 84 negative residuals in
that dataset without adding structural-zero entries to the sparse correction.
That absence of support growth is dataset-specific and must not become an API
assumption.

### Block-wise inverse reconstruction to count space

Provide native reconstruction from a `PCAResult` without materializing the
full observation-by-feature matrix.

The first stage is common to every PCA transform. For an observation block:

```python
transformed = result.scores[rows] @ result.components + result.mean
```

The second stage is transform-specific and may require stored or supplied
normalization state.

#### Desired behavior

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

#### Identifiability by transform

| Transform | Count-space inverse requirements |
| --- | --- |
| shifted CLR | `count_shift` plus original or supplied row total to restore the removed log gauge |
| composition-scale shifted CLR | `composition_shift` recovers proportions; original row totals are needed for counts |
| Dirichlet log | concentration, prior proportions, and row totals |
| Dirichlet CLR | prior counts plus row totals to restore the removed log gauge |
| residual PCA | fitted null-model state and a stable analytic or numerical inverse for the selected residual family |
| correspondence analysis | row/column masses and grand total, with semantics distinct from PCA reconstruction |

For count-scale shifted CLR with reconstructed CLR values `z`, row total `s`,
and `G` genes, the missing log gauge is identifiable from

```text
exp(t) = (s + G * count_shift) / sum(exp(z)).
```

Then the approximate counts are `exp(z + t) - count_shift`. Analogous closure
constraints apply to Dirichlet CLR and composition-scale shifted CLR.

#### Likely API shape

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

### Exact negative-binomial Anscombe PCA

The exact NB variance-stabilizing transform can be written stably as

```math
g_\alpha(x)
=
\frac{2}{\sqrt\alpha}\operatorname{asinh}(\sqrt{\alpha x})
=
\frac{1}{\sqrt\alpha}\operatorname{acosh}(1+2\alpha x),
\qquad \alpha>0,
```

with Poisson limit \(g_0(x)=2\sqrt{x}\).

Representation is not the obstacle. Because \(g_\alpha(0)=0\), applying it to
`X.data` alone gives an exact CSR matrix with unchanged support — a rank-zero
sparse-plus-low-rank representation.

The open questions are semantic, and should be settled before it gets a public
API:

- whether `alpha` is scalar or per gene;
- whether the package estimates `alpha` or only consumes supplied values;
- how AnnData-aligned dispersion arrays and masks behave;
- whether this raw-count VST is useful without a separate depth model;
- how its purpose differs from scaled-NB residual PCA.

If added, call it `nb_anscombe_pca`; the large-count logarithmic approximation
is not the exact transform.

### Correspondence analysis on Freeman-Tukey residuals

Look into a Freeman-Tukey residual variant of correspondence analysis. It sits
in the same variance-stabilizing family as the
[NB Anscombe transform above](#exact-negative-binomial-anscombe-pca), and the
same question decides whether it earns a public API: whether it gives a result
that the Poisson and scaled-NB modes do not already give.

### Real-data oracle suite

Add independently generated outputs from the external implementations whose
compatibility the package claims but which are not yet exercised on the shared
PBMC3k fixture: Scanpy, Townes, CORRAL, and the pinned BHGP formulas. Continue
to use the deterministic cells and genes where the upstream method permits it.
The completed fixture and dense-formula coverage are recorded in the
[PBMC3k fixture documentation](https://github.com/jgarthur/sparse_count_pca/blob/main/tests/real_data_reference/README.md).

For each transform, compare the most diagnostic independently generated
artifacts available: selected transformed entries, clipping support and `nnz`,
column means, total variance or inertia, singular values, and component
subspaces. Do not rely only on final PCs when an earlier representation error
could be hidden by sign or subspace invariance.

Keep ordinary tests offline and deterministic:

1. Record the source URL, source-data checksum, subset rule, feature/cell names,
   oracle package version, and regeneration environment.
2. Commit a modest compressed CSR fixture and compact golden outputs directly
   to Git when their total size is reasonable. This is more reliable in CI than
   requiring Git LFS or a live download.
3. Put larger datasets or expensive regeneration jobs behind an optional test
   marker and a scheduled/manual workflow with checksum-verified caching.
4. Treat regeneration scripts as provenance; tests consume pinned artifacts and
   never silently refresh them.

### Relationship between CA and Poisson Pearson-residual PCA

The correspondence-analysis standardized residual is exactly the Poisson
Pearson residual divided by the grand total's square root, which
`docs/transforms.md` now states. The stronger and more useful statement is not
yet written down: how the two sets of cell scores relate, believed to be the
same up to a factor proportional to \(\sqrt{n_i}\), given that CA omits column
centering and applies mass weighting.

Derive and verify this before documenting it. It would give readers a concrete
reason to prefer one over the other rather than treating them as unrelated
methods.

### Uncentered Pearson residual PCA

Consider offering `zero_center=False` for Pearson residual PCA. Residuals
already have approximately zero column mean under the null model, so centering
is close to a no-op, and skipping it is believed to correspond more directly
to GLM-PCA — believed, not verified, and worth checking before it is claimed
anywhere.

The plumbing partly exists: `zero_center` is already a reported field in
`PCAResult.params`, hardcoded to `True` for PCA (`_pca.py`) and `False` for
correspondence analysis (`_correspondence.py`). Nothing exposes it as a
caller-facing option.

Treat this as a step toward the
[CA and Poisson Pearson-residual relationship above](#relationship-between-ca-and-poisson-pearson-residual-pca)
rather than a standalone feature. Omitted column centering is one of the two
differences between the two analyses, so an uncentered residual PCA isolates
the other one — mass weighting — and makes the comparison concrete.

### Deferred performance work

None of these change public behavior:

- Explore a fully lazy residual operator that reads the caller's canonical CSR
  counts and computes normalized values during matrix products, avoiding a
  second matrix with duplicated support. This requires an explicit
  borrowed-input mutation contract; exact symmetric clipping may need an eager
  fallback or a hybrid correction matrix because it can add support at
  zero-count entries.
- For heavily masked log and Dirichlet PCA, make a two-pass builder that computes
  normalization statistics over all genes but allocates sparse transformed
  values and support only for PCA-selected genes.
- Track CSR ownership explicitly so a transient one-step analysis that does not
  return an operator can borrow canonical input support during synchronous PCA.
  A persistent `TransformedMatrix` or returned operator must still own one
  support snapshot so later caller mutation cannot change its behavior.
- Merge clipping corrections directly into CSR rather than concatenating COO
  arrays and sorting them during conversion.
- Combine or cache the operator's CSC-based mean and squared-norm traversals;
  investigate a CSR-native uncentered-statistics path for correspondence
  analysis.
- Normalize explicit selectors that equal the full ordered axis to the same
  fast path as `obs=None` or `var=None` during materialization.
- Revisit the full `adata.var` snapshot retained by `TransformedMatrix`.
  Preserving stable later mask resolution is useful, but retaining only names
  and required metadata columns could be substantially smaller.
- Consolidate the private implementation of the two shifted-CLR wrapper pairs
  while keeping their scientifically distinct public names.

### Deferred API decisions

These require code or API decisions rather than editorial changes. Each was
originally held for 1.0 on the grounds that changing it afterwards would be a
breaking change; with no users at release that cost is small, so settle them
when there is a reason to rather than on the release schedule.

- **Decide whether `scaled_nb` is sufficiently clear as a public model name**,
  or should be renamed to expose its exposure-scaled dispersion
  parameterization. See [the scaled-NB null model](reference/scaled-nb-model.md).
- **Compare package-level outputs and defaults directly with
  [`cleartools/scclr`](https://github.com/cleartools/scclr).** The current dense
  PFlog oracle establishes formula-level parity only, not end-to-end package
  equivalence, and the compatibility page says so.
- **Revisit the masking API.** The current direction preserves `mask_var` and
  its method-specific ordering, with the distinction stated in the README,
  concept page, CA guide, and public docstrings. Confirm that this visibility is
  sufficient; otherwise consider a more explicit correspondence-analysis
  parameter name without changing the underlying mathematics.

### Documentation structure

Preserve `architecture.md` as a guided review document: its reading order,
diagrams, invariants, and review checklist are useful and should not be replaced
by a conventional module inventory. The documentation root is an appropriate
home for that guide. If the filename is ever changed, `review-guide.md` would be
more literal, but the link churn is not urgent.

When splitting the normative specification, put implementation reference
material in a separate `internals.md` or similarly named document rather than
repurposing the review guide. Put numerical contracts and oracle policy in
focused `numerics.md` and `verification.md` documents, while preserving one
clearly identified normative API/specification entry point.

## Unscoped ideas

- Check whether `corral` uses sparse tricks worth borrowing.
- Merge `_dirichlet_pca` and `_log_pca`, which are separate files for no
  evident reason. Distinct from consolidating the one-step wrappers above.
- Widen the docs site's main content panel so the transform catalog table
  wraps less. Minor polish.
- Quick recipes to reproduce Seurat, Cell Ranger, scanpy, BHGP, correspondence analysis recommendations
- Skill.md for agentic usage. recommend shifted clr and possibly correpondence analysis
- Highly variable gene selection based on residual variance.
- Supplied arbitrary size factors, including scran-derived factors.
- Dask-backed inputs and block-aware transformations.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
