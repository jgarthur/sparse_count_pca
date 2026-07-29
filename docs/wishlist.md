# Wishlist

This file records unfinished plausible additions without treating them as
committed API. When work is completed, remove it from this file and record the
outcome in the appropriate specification, design, architecture, or provenance
documentation; do not leave completed work described as a wish.

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

## Real-data oracle suite

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

## Deferred performance work

These are worthwhile but not launch blockers:

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

## Prune stale rendered example fragments

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

## Backed-AnnData test gaps

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

## Materialization test gaps

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

## Deduplicate `components` and `loadings` on `PCAResult`

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

## Relationship between CA and Poisson Pearson-residual PCA

The correspondence-analysis standardized residual is exactly the Poisson
Pearson residual divided by the grand total's square root, which
`docs/transforms.md` now states. The stronger and more useful statement is not
yet written down: how the two sets of cell scores relate, believed to be the
same up to a factor proportional to \(\sqrt{n_i}\), given that CA omits column
centering and applies mass weighting.

Derive and verify this before documenting it. It would give readers a concrete
reason to prefer one over the other rather than treating them as unrelated
methods.

## Release and packaging

**The README's documentation links will break on PyPI.** GitHub resolves
relative Markdown links against the repository; PyPI does not rewrite them at
all, so every `docs/...` link in `README.md` resolves against `pypi.org` and
404s. There are currently 13 of them. `pyproject.toml` also declares no
`[project.urls]`, so the PyPI sidebar will show no Homepage, Documentation, or
Repository link.

Fix both when the documentation site is published, in the same change that adds
the Read the Docs configuration, so the absolute URLs exist when they are
written:

- rewrite the README's `docs/...` links to absolute published URLs, leaving
  repository-internal links such as `CONTRIBUTING.md` relative, since those are
  correct on GitHub and are not the PyPI reader's concern;
- add `[project.urls]` with Homepage, Documentation, and Repository.

The division of labor these links assume: `README.md` serves readers who have
not yet decided to install, on PyPI and GitHub, and must support that decision
on its own. `docs/index.md` serves readers already on the site and is a router
into the guides, concepts, and reference.

## Documentation structure

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

## Other candidates

- Decide whether `transform(X, Residual(...))` should permit all-zero genes so
  a later `.pca(mask_var=...)` can exclude them. The one-step AnnData residual
  API can do this because it resolves the PCA mask before building residuals;
  the two-step API currently validates every gene when the transform is fitted.
- Quick recipes to reproduce Seurat, Cell Ranger, scanpy, BHGP, correspondence analysis recommendations
- Skill.md for agentic usage. recommend shifted clr and possibly correpondence analysis
- Highly variable gene selection based on residual variance.
- Supplied arbitrary size factors, including scran-derived factors.
- Dask-backed inputs and block-aware transformations.
- Exact negative-binomial Anscombe PCA after dispersion semantics are settled.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
