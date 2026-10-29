# Wishlist

This file records unfinished plausible additions without treating them as
committed API. When work is completed, remove it from this file and record the
outcome in the appropriate specification, design, architecture, or provenance
documentation; do not leave completed work described as a wish.

## Before the 1.0.0 release

### Final benchmarks and the claims that cite them

Run the final Scanpy comparison on the Mac and on a remote machine, possibly
with a larger dataset there. Use the results to quantify the README's
efficiency claim and to give the 1.0.0 changelog's performance entries
identifiable evidence: the dataset, scale, revision, and whether each figure is
a benchmark or a real-data measurement.

### Citation DOI

Zenodo's GitHub integration is enabled for this repository, so publishing the
`v1.0.0` GitHub Release archives it and mints a version DOI plus a concept DOI
that resolves to the latest version. `CITATION.cff` supplies the record's
metadata; an invalid file blocks archiving, so check it with
`uvx cffconvert --validate -i CITATION.cff` before releasing.

- Before publishing, set `date-released` in `CITATION.cff` and the `## 1.0.0`
  heading in `CHANGELOG.md` to the release date.
- After Zenodo archives the release, add the concept DOI as `doi:` in
  `CITATION.cff`, add a DOI badge and the DOI to the README's Citation section
  and the documentation home page's "Citing sparse-count-pca" section, and
  remove this entry.

## After 1.0

### Opt-in parallel sparse products for PCA (1.1 target)

Provide real multicore sparse matrix products through an explicit Python
option, without requiring users to set environment variables. Keep this work
separate from the 1.0 release until implementation and validation are complete.
Target opt-in execution in 1.1, preserving serial sparse products when the
option is omitted. The proposed API name `n_jobs` is not committed API.

#### User contract and release decision

- Make the option available consistently on `TransformedMatrix.pca`, one-step
  matrix PCA functions, and AnnData PCA wrappers. Keep configuration local to
  a calculation rather than implicitly inheriting Scanpy's global settings.
- Preserve the serial sparse-product path for `n_jobs=1`, including ownership,
  solver, tolerance, dtype, masking, and output contracts. This means no sparse
  worker pool; it does not promise that every native library uses one thread.
- Preserve omitted-argument behavior in 1.1 and require explicit opt-in for
  parallel execution. Adding an optional argument on those terms does not
  require a major version change.
- An automatic worker count, if offered, must also be explicitly requested;
  automatic parallel execution is outside the 1.1 default policy. Evaluate
  choosing at most four available CPUs, accounting for affinity, container CPU
  quotas, and external process-level parallelism; host CPU count alone is
  insufficient. Retain explicit one-worker execution.
- Define validation for supported thread counts and any automatic sentinel;
  do not assume joblib's negative-count semantics just from the name `n_jobs`.
- Users must be able to enable the feature with the Python API alone, both with
  untouched environments and with native thread limits imposed by a pipeline.
  Coordinate native thread budgets internally, preserve stricter limits, and
  restore any temporarily changed limits on success and failure. Never mutate
  process environment variables as the implementation of a per-call option.
- Define which settings control sparse workers versus native libraries. Native
  limits can affect unrelated work in the same process, so nested/concurrent
  calls need an explicit policy. Establish support for each native backend;
  do not assume that one threadpool-control mechanism covers Accelerate,
  OpenBLAS, and OpenMP identically.

Scanpy's documented `settings.n_jobs` default is one, and its ordinary
in-memory sparse ARPACK PCA path delegates to scikit-learn without consulting
that setting. It is not evidence for automatically choosing all available
cores here. See [Scanpy's setting](https://scanpy.readthedocs.io/en/stable/generated/scanpy.settings.n_jobs.html)
and its [PCA implementation](https://github.com/scverse/scanpy/blob/1.12.4/src/scanpy/preprocessing/_pca/__init__.py).

#### Implementation direction

Parallelize the sparse products inside the linear operator while retaining a
single ARPACK invocation and the same low-rank and centering corrections.
Workers are threads in one process sharing immutable matrix buffers, not
processes with separate copies of the input.

Create one bounded pool per PCA solve and reuse it for every forward and
transpose product, including matrix-block products. Give each worker a CSR
row block. Forward products produce disjoint output rows; transpose products
produce private partial outputs that are reduced in a defined order. Cache
lightweight block and transpose views for reuse. Avoid copying the full sparse
matrix or constructing a separate full transpose, and budget partial-output
storage explicitly. Check buffer sharing rather than relying on `copy=False`.
Consider nonzero-balanced row boundaries for uneven row densities.

Put this into the operator and shared PCA execution layers, with option
plumbing through public wrappers. Keep transforms and solver mathematics
unchanged. Preserve the package's owned/borrowed support contracts: borrowing
within a solve must not weaken the isolation promised for returned operators.
Guarantee pool cleanup after exceptions and interrupted solves. Returned
operators must remain usable after the solve's pool closes, without retaining
live workers or unexpected references to temporary buffers. Define behavior
for other supported solvers and correspondence analysis explicitly; sharing
operator machinery must not silently change their defaults.

#### Correctness and lifecycle tests

- Compare forward/transpose vector and block products with independent dense
  oracles, including the adjoint identity, for float32 and float64, centered
  and uncentered operators, and rank-zero and nonzero low-rank corrections.
- Cover every supported representation reaching the shared operator: residual
  families/clipping, log1p, CLR, and Dirichlet transforms; include masking and
  owned/borrowed input support. Check that inputs remain unchanged.
- Exercise empty rows/blocks, more workers than rows, uneven nonzero counts,
  tiny matrices, read-only buffers, and supported int32/int64 index layouts.
  Verify buffer sharing and bound partial-output allocations.
- Compare serial and parallel singular values, principal subspaces, aligned
  scores/components, means, total variance, and variance ratios using the
  package's dtype-appropriate criteria. Treat summation-order differences as
  numerical effects, not fixes; do not require bitwise identity across worker
  counts or relax accuracy thresholds to accommodate failures.
- Check omitted arguments and explicit `n_jobs=1` for serial compatibility,
  including the absence of a pool and unchanged native settings. Exercise
  public wrappers and repeated PCA on a fitted transform.
- Exercise worker failures, solver nonconvergence, interruption, repeated
  solves, nested/concurrent calls, and returned-operator reuse. Verify thread
  cleanup and restoration of native settings, including exception paths.
- Test the supported Python/NumPy/SciPy range; the package's minimum SciPy
  predates the C ARPACK rewrite. Avoid depending on prototype-only private
  method replacement or assumptions about solver reentrancy.

#### Performance and ordinary-user validation

Run fresh-process comparisons on Apple Silicon/macOS with Accelerate and
Linux with OpenBLAS, using the public API of an isolated candidate checkout.
Record source/input hashes, dependency versions, native backends, resolved
worker counts, CPU allocation, and whether environment settings were inherited
or explicitly controlled. Keep concurrent benchmark jobs off each host.

1. First test the actual ordinary-user workflow with relevant BLAS, OpenMP,
   and Accelerate thread variables unset: omitted option, explicit one worker,
   four workers, eight workers, and any proposed automatic mode. Environment
   manipulation belongs in the benchmark launcher, not the package API.
2. Repeat with inherited native limits of one and larger limits, plus restricted
   CPU allocations and outer process parallelism. Check that the Python option
   remains sufficient, caller limits are not increased, and native settings are
   restored. Compare runtime and CPU consumption to diagnose oversubscription;
   allocated thread counts alone do not establish simultaneous CPU contention.
3. Use interleaved repeated runs on real PBMC and larger sparse brain subsets,
   covering both precisions and residual/log1p PCA. Include small inputs where
   setup can dominate and skewed sparsity where equal-row splits may underuse
   workers. Measure pool setup, transform/statistics, solve, and complete
   calculation separately, with input loading and output writing identified.
4. Measure process CPU seconds divided by wall time and, where useful,
   per-thread activity. Demonstrate real multicore work and complete-workflow
   speedups rather than inferring them from configured limits. Compare Scanpy
   sparse log1p PCA separately from dense residual PCA, preserving dtype and
   solver settings and excluding lazy compilation consistently.
5. Measure sampled peak RSS/high-water memory and paging, not just post-phase
   RSS. Verify that sparse-buffer storage is not multiplied by worker count.
   Establish per-worker memory containment before Linux capacity/OOM trials so
   a failed calculation cannot take down the coordinator. Attribute system-wide
   swap counters cautiously. Use subsets on the Mac; full-brain Mac execution
   is outside the benchmark's authorization.

Keep hardware-dependent speed assertions in manual benchmarks rather than
ordinary CI. Use these results to decide the automatic-mode policy and whether
four workers are a useful recommendation across input sizes. Separate solver
and tolerance experiments from threading comparisons so their effects are not
confounded. Follow the [testing guide](development/testing.md) for baseline
numerical and ownership contracts.

#### Prototype and evidence locations

The companion `compbio-notebooks` repository contains the executable prototype
and retained comparisons. These local cross-repository links assume sibling
checkouts; they are not published documentation URLs:

- [Cross-host findings, raw-output links, and reproduction instructions](../../compbio-notebooks/projects/scp-testing/results/threading-20261029-sparse-validation/findings.md).
- [Sparse-product prototype](../../compbio-notebooks/projects/scp-testing/scripts/probe_mac_headroom.py)
  and [sweep driver](../../compbio-notebooks/projects/scp-testing/scripts/validate_sparse_threads.py).
- [Results index](../../compbio-notebooks/projects/scp-testing/results/index.md),
  including Mac residual measurements and frozen-source provenance.

The retained runs use a frozen source snapshot, controlled native thread
budgets, and one small PBMC fixture. They motivate this work but do not satisfy
its unset-environment, larger-input, supported-version, or production-lifecycle
acceptance tests. Revalidate against the integration revision after concurrent
package work settles; keep the separate tolerance study independent.

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
  second matrix with duplicated support. This requires extending the
  `share_structure` contract from support to count values; exact symmetric
  clipping may need an eager fallback or a hybrid correction matrix because it
  can add support at zero-count entries.
- For heavily masked log and Dirichlet PCA, make a two-pass builder that computes
  normalization statistics over all genes but allocates sparse transformed
  values and support only for PCA-selected genes.
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

### Retire the 1.0.0 clipping notice

Remove the "1.0.0 release: residual clipping has changed" admonition from the
top of `docs/index.md` in the first release after 1.0.0. The changelog and
clipping guide keep the migration note.

## Unscoped ideas

- Check whether `corral` uses sparse tricks worth borrowing.
- Merge `_dirichlet_pca` and `_log_pca`, which are separate files for no
  evident reason. Distinct from consolidating the one-step wrappers above.
- Widen the docs site's main content panel so the transform catalog table
  wraps less. Minor polish.
- Quick recipes to reproduce Seurat, Cell Ranger, scanpy, BHGP, correspondence analysis recommendations
- Skill.md for agentic usage. recommend shifted clr and possibly correpondence analysis
- Highly variable gene selection based on residual variance.
- Dask-backed inputs and block-aware transformations.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
