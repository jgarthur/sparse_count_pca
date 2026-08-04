# Wishlist

This file records unfinished plausible additions without treating them as
committed API. When work is completed, remove it from this file and record the
outcome in the appropriate specification, design, architecture, or provenance
documentation; do not leave completed work described as a wish.

## Before 1.0

### Define the validation status labels in one place

The transform catalogue's Status column, the Dirichlet sections, and the
scaled-NB correspondence-analysis mode all use "experimental" in the
scientific sense — not thoroughly validated — but the package never says so.
A reader is free to take it as a statement about API stability instead, which
is not what is meant.

The meaning is currently derivable only from per-section caveats: the
Dirichlet caveats note there is no external reference implementation to pin,
and the correspondence-analysis section flags `scaled_nb` as an extension past
the `corral` fixture. `docs/transforms.md` already states the general bar in
prose — every transform is tested against an independently written dense
implementation of its formula, with pinned external fixtures named per
section. Turn that paragraph into an explicit legend for the Status column:

- **Supported** — the formula is verified against an independently written
  dense implementation, plus whatever external fixtures that section names.
- **Experimental** — package-defined, or extended past any external reference;
  the implementation is tested, but the method has no outside corroboration.

State that neither label promises anything about API stability.

The legend also has to resolve a conflation the current two-value column
hides. The Dirichlet transforms are package-defined prior-count
generalizations, so no external reference will ever exist and they can never
graduate to supported under an oracle-based definition. That is a different
situation from `scaled_nb`, which extends a method that does have a reference
and could be validated later. Either add a third state, or reword the
Dirichlet rows as package-defined and reserve "experimental" for the second
case.

Raised as P1-3 in the 2026-07-22 documentation review; the rest of that review
is resolved.

### State the batch and depth caveat for `scaled_nb`

The scaled-NB null model fits one exposure-scaled dispersion relationship
across the whole matrix. That is not appropriate for data with multiple
batches, or with large sequencing-depth differences between cells, where the
dispersion-exposure relationship differs by group and the model should really
be fitted batch-wise.

The documentation nowhere says this. Batches are not discussed at all outside
one citation title, so a reader has no signal that the mode has a scope
condition. Add the caveat where `scaled_nb` is introduced — the transform
catalogue's correspondence-analysis caveats and the scaled-NB null model
reference — and say what to do instead, rather than only that it is
experimental.

Whether the package should offer batch-wise fitting itself is a separate
question; stating the limitation does not depend on answering it.

### Remove the raw-count shifted log transform

Remove `shifted_log_pca`, `shifted_log_pca_matrix`, and `ShiftedLog` before the
first stable release. Applying `log1p(X / count_shift)` to raw counts without a
depth model has no clear use outside input that is already depth-normalized,
and the name invites exactly the misuse the catalogue's "what this package does
not do" section warns against.

The three log-family transforms are flat siblings in `_compute_log_pca`, so
removal is an excision of one branch rather than an untangling. It touches
roughly 84 sites across `src`, tests, docs, and the README, and it removes the
only supported plain-log transform, leaving the two CLR transforms and the
experimental Dirichlet log. Do it before the `use_raw` removal, which otherwise
edits code this deletes.

### Remove `use_raw` from the AnnData APIs

Remove the `use_raw` option before the first stable release. Current Scanpy PCA
does not offer `.raw.X` as an input convention, and this package must realign
`.raw` to the current `adata.var_names` in any case, so genes retained only in
`.raw` do not re-enter the transform universe. The option therefore adds a
second count-source path and non-obvious alignment behavior without providing
the full-gene semantics a caller might expect from `.raw`.

Keep `.X` and `layer` as the explicit AnnData count sources. Users who need a
different variable universe can construct or subset the working `AnnData`
object before calling the package. Removing the option requires updating the
AnnData wrappers, API reference, specification, guides, examples, and tests that
currently cover `use_raw=True`.

### Deduplicate `components` and `loadings` on `PCAResult`

`PCAResult.loadings` is assigned `components.T` (`_pca.py`), so the two
attributes are the same array transposed and every result object carries both.
It is not a variance-weighted statistical loading, which is what a reader is
likely to assume from the name, and the docstring has to say so.

Options, in rough order of preference:

- drop `loadings` and let callers transpose, since `components` matches the
  scikit-learn and Scanpy convention that the rest of the API follows;
- keep `loadings` as a property computed on access rather than a stored field,
  removing the duplicate array without breaking callers;
- keep both but rename `loadings` to something that does not invite the
  statistical-loading reading.

Decide before the first stable release, since removing a public attribute
afterwards is a breaking change. The AnnData path is unaffected either way:
`.varm` is written from the variable-oriented array regardless.

### Unify all-zero gene and cell handling across the APIs

`transform(X, Residual(...))` validates every gene when the transform is
fitted, so an all-zero gene is rejected even when a later `.pca(mask_var=...)`
would exclude it. The one-step AnnData and matrix residual APIs accept the same
gene, because they resolve the PCA mask before building residuals
(`tests/test_zero_genes.py`). Log and Dirichlet transforms accept all-zero
genes on both paths, so the divergence is specific to the residual families.

Make the behavior identical across the one-step AnnData, one-step matrix, and
two-step entry points. Two directions unify it:

- permissive: defer the two-step check from fit time to `.pca`, so an all-zero
  gene is rejected only when it is actually selected. This matches current
  one-step behavior but moves when the error surfaces, which matters for a
  `TransformedMatrix` fitted once and reused with several masks;
- strict: reject all-zero genes on every path regardless of masking, which is
  simpler to state but removes something that works today.

The asymmetry is structural rather than accidental: the one-step APIs know the
mask at fit time and the two-step API does not. Pick a direction deliberately
and state it in the specification.

All-zero cells are not covered by the current validation, which only rejects
masks that select nothing. Confirm whether they need the same treatment before
settling the rule.

Then explain the settled rule against what a Scanpy or Seurat user expects.
Those pipelines normally filter empty cells and genes upstream, so a reader
arriving from either one may never have met the question and will be surprised
to hit an error rather than a dropped row or column. Whatever the rule turns
out to be, state it where such a reader will find it, and say plainly that
zero cells and zero genes are not treated alike if they are not.

### Backed-AnnData test gaps

`tests/test_residual_anndata.py::test_backed_sparse_x` asserts that
`uns["pca"]["singular_values"]` is written and that the object is still backed
afterwards. Two behaviors the AnnData guide now states are not covered:

- `obsm[key]` and `varm[key]` are written too, not only `uns`. Backed mode
  keeps just `.X` on disk, so all three result mappings are ordinary in-memory
  attributes and `copy=False` populates them normally.
- The backing file is left untouched. Reopening the `h5ad` after an in-place
  call shows empty `obsm`, `varm`, and `uns`, which is why the guide tells
  readers to save results themselves.

Both were confirmed by hand against anndata 0.12.16 while writing the guide.
Add them to the existing test so the documented behavior is pinned rather than
assumed, since it depends on AnnData's backed-mode semantics rather than on
anything this package controls.

### Materialization test gaps

`TransformedMatrix.materialize` accepts positional, boolean, slice, and
AnnData-name selectors (`_resolve_selection`), and the guides say so. Tests in
`tests/test_transform.py` cover names, scalar indices, and ordered integer
arrays including repeats and reordering (`obs=[4, 1, 4]`), but **not slice or
boolean-mask selectors**. Add cases for both so the documented surface is
covered.

Separately, `out` and `block_size` are tested for correctness — values match,
the destination array is returned, and invalid shape, dtype, and block size are
rejected — but nothing tests that they actually *bound* peak memory. The guide
presents that as the reason to use them. A test that materializes into a
memory map and asserts no full-size temporary is allocated would make the
claim real rather than assumed.

### Prune stale rendered example fragments

`docs/hooks/render_examples.py` writes `docs/examples/<stem>.md` for every
`examples/*.py`, but never removes outputs whose source no longer exists.
Renaming an example therefore leaves its old fragment behind indefinitely: it
is regenerated by nothing, matched by no cache entry, and included by no guide.
Two such orphans accumulated during the documentation overhaul and were deleted
by hand.

They are harmless — gitignored, and `exclude_docs` keeps them out of the built
site — but they make `docs/examples/` a misleading picture of what the build
produces, and a stale fragment whose name a guide later reuses would be
silently included until the first re-render.

Have `on_config` delete files under the output directory that no current
example accounts for, using the same set it already assembles for the cache.

### Use American spellings throughout

Settle on American English. The only inconsistency currently present is
"catalogue", in eight places across the transform catalogue, the index, the
getting-started page, two guides, the scaled-NB reference, and this file. No
other British spellings appear.

`docs/transforms.md` titles its table "Transform catalogue" and other pages
refer to it by that name, so check the heading anchor and any prose
cross-references when renaming rather than replacing the word blindly.

### Release and packaging

**The README's documentation links will break on PyPI.** GitHub resolves
relative Markdown links against the repository; PyPI does not rewrite them at
all, so every `docs/...` link in `README.md` resolves against `pypi.org` and
404s. There are currently 12 of them. `pyproject.toml` also declares no
`[project.urls]`, so the PyPI sidebar will show no Homepage, Documentation, or
Repository link.

Fix both when the documentation site is published, in the same change that adds
the Read the Docs configuration, so the absolute URLs exist when they are
written:

- add the Read the Docs project and its `.readthedocs.yaml` configuration, which
  is what makes the absolute URLs exist;
- rewrite the README's `docs/...` links to absolute published URLs, leaving
  repository-internal links such as `CONTRIBUTING.md` relative, since those are
  correct on GitHub and are not the PyPI reader's concern;
- add `[project.urls]` with Homepage, Documentation, and Repository.

The guides' evidence links have the same publication dependency. Every link
into `examples/*.py` and `tests/` targets `main` on a repository that is
currently private, and the examples were renamed during the documentation
overhaul, so `main` does not yet contain those paths at all. They resolve once
this branch merges and the repository is public. Re-verify them at that point
rather than rewriting them now. The same applies to the advertised
`pip install sparse-count-pca`, which cannot work until the package is
published.

The division of labor these links assume: `README.md` serves readers who have
not yet decided to install, on PyPI and GitHub, and must support that decision
on its own. `docs/index.md` serves readers already on the site and is a router
into the guides, concepts, and reference.

## After 1.0

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
| shifted log | `count_shift`; direct `count_shift * expm1(z)` |
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

If added, call it `nb_anscombe_pca` and keep it distinct from
`shifted_log(count_shift=1/(4*alpha))`, which is only a large-count log
approximation to it rather than the exact transform.

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
- Consolidate the private implementation of one-step log/CLR wrappers while
  keeping their scientifically distinct public names and matrix/AnnData pairs.
  Removing the raw-count shifted log leaves two wrappers to consolidate rather
  than three.

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
- Widen the docs site's main content panel so the transform catalogue table
  wraps less. Minor polish.
- Quick recipes to reproduce Seurat, Cell Ranger, scanpy, BHGP, correspondence analysis recommendations
- Skill.md for agentic usage. recommend shifted clr and possibly correpondence analysis
- Highly variable gene selection based on residual variance.
- Supplied arbitrary size factors, including scran-derived factors.
- Dask-backed inputs and block-aware transformations.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
