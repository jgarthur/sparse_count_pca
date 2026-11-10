# Wishlist

This file records unfinished plausible additions without treating them as
committed API. When work is completed, remove it from this file and record the
outcome in the appropriate specification, design, architecture, or provenance
documentation; do not leave completed work described as a wish.

## Before the 1.0.0 release

### Citation DOI

Zenodo's GitHub integration is enabled for this repository, so publishing the
`v1.0.0` GitHub Release archives it and mints a version DOI plus a concept DOI
that resolves to the latest version. `CITATION.cff` supplies the record's
metadata; an invalid file blocks archiving, so check it with
`uvx cffconvert --validate -i CITATION.cff` before releasing.

- `date-released` in `CITATION.cff` and the `## 1.0.0` heading in
  `CHANGELOG.md` give 2026-11-04. If the release is published later, update
  both to the publication date.
- After Zenodo archives the release, add the concept DOI as `doi:` in
  `CITATION.cff`, add a DOI badge and the DOI to the README's Citation section
  and the documentation home page's "Citing sparse-count-pca" section, and
  remove this entry.

## After 1.0

### Parallel sparse products for PCA (1.1 target)

Provide real multicore sparse matrix products, controlled from Python without
requiring users to set environment variables. Keep this work separate from the
1.0 release until implementation and validation are complete. The 1.1
candidate default is an automatic worker count of at most four. That default
is conditional on reproducibility: PCA results must not depend on how many
workers ran. If that cannot be met, ship 1.1 with serial products by default
and parallel execution as an explicit opt-in. The proposed API name `n_jobs`
is not committed API.

#### User contract and release decision

- Make the option available consistently on `TransformedMatrix.pca`, one-step
  matrix PCA functions, and AnnData PCA wrappers. Keep configuration local to
  a calculation rather than implicitly inheriting Scanpy's global settings.
- Resolve the omitted option (a sentinel such as `n_jobs=None`) to
  `min(4, available CPUs, caller thread limit)`. Explicit `n_jobs=1` runs
  without a sparse worker pool, preserving ownership, solver, tolerance, dtype,
  masking, and output contracts; it does not promise that every native library
  uses one thread. Define whether explicit counts above the available CPUs are
  honored or capped. Do not assume joblib's negative-count semantics just from
  the name `n_jobs`.
- Count available CPUs as those the process may use: CPU affinity
  (`os.sched_getaffinity` where it exists) and container CPU quotas. Host CPU
  count (`os.cpu_count()`) is insufficient: a SLURM job allocated one CPU or a
  Docker container started with `--cpus=2` still reports every host core.
  Affinity covers cpuset-based allocations; only the cgroup quota covers
  `--cpus`. Python 3.11 lacks `os.process_cpu_count`, and the standard library
  does not read cgroup quotas, so plan to port or depend on loky's CPU
  detection.
- Treat OpenMP and BLAS limits the caller already set, through environment
  variables or `threadpoolctl`, as a ceiling on the automatic count. A
  pipeline that pins native threads to one then stays serial without new
  configuration. Bound the remaining oversubscription risk, process-level
  parallelism that sets no limits, through the cap and the benchmarks below.
- Reproducibility gate for the automatic default: results must be bitwise
  identical for every worker count, including one, given the same input,
  host, and environment. Otherwise results would change with the CPU
  allocation, for example between a laptop and a two-core CI runner. The
  [implementation direction](#implementation-direction) describes the
  partition that provides this.
- Omitted-argument results then differ from 1.0 by summation order only,
  because the transpose reduction sums block partials. Record that and the
  higher default CPU use under other changes in the changelog, not fixes;
  neither requires a major version change. Open question: whether any path
  must reproduce 1.0 bitwise. A fixed partition used for every worker count,
  including one, rules that out.
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
that setting. See [Scanpy's setting](https://scanpy.readthedocs.io/en/stable/generated/scanpy.settings.n_jobs.html)
and its [PCA implementation](https://github.com/scverse/scanpy/blob/1.12.4/src/scanpy/preprocessing/_pca/__init__.py).
That setting governs process-level parallelism. Native threading in the same
stack uses multiple cores by default: this package's dense low-rank products
already run on multithreaded BLAS, and scikit-learn's OpenMP code uses the
available cores while honoring `OMP_NUM_THREADS`. In-process sparse workers
sharing one copy of the matrix resemble native threading more than
process-level parallelism.

#### Implementation direction

Parallelize the sparse products inside the linear operator while retaining a
single ARPACK invocation and the same low-rank and centering corrections.
Workers are threads in one process sharing immutable matrix buffers, not
processes with separate copies of the input.

Create one bounded pool per PCA solve and reuse it for every forward and
transpose product, including matrix-block products. Split the CSR matrix into
row blocks whose boundaries depend only on the matrix, for example a fixed
number of nonzero-balanced blocks no smaller than the worker cap, never on the
worker count; workers take blocks from that partition, and one-worker
execution runs the same blocks in sequence. Forward products produce disjoint
output rows, so they match for any partition. Transpose products produce one
private partial output per block, reduced in block order, which makes results
independent of the worker count. Cache lightweight block and transpose views
for reuse. Avoid copying the full sparse matrix or constructing a separate
full transpose, and budget partial-output storage, which scales with the
block count, explicitly. Check buffer sharing rather than relying on
`copy=False`.

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
- Require bitwise-identical PCA results for one worker, every supported worker
  count, and the automatic count, given identical input, environment, and
  settings. This is the reproducibility gate for the automatic default.
- Compare the partitioned products against the unpartitioned 1.0 serial
  products: singular values, principal subspaces, aligned
  scores/components, means, total variance, and variance ratios using the
  [numerical agreement targets](development/testing.md#numerical-agreement-targets):
  `1e-4` for float32 and `1e-12` for float64, independently for maximum
  singular-value relative error and maximum per-component score/loading
  relative L2 errors. Preserve component order and apply each loading-derived
  sign to the corresponding score column; do not rotate or mix components.
  Report whole-matrix Frobenius errors and principal angles as diagnostics
  without additional thresholds. Preserve separate statistics checks. Treat
  summation-order differences from 1.0 as numerical effects, not fixes; do
  not relax the targets to accommodate failures.
- Check that the omitted option resolves to the documented automatic count
  under restricted affinity, container quotas, and caller-imposed native
  limits, and that explicit `n_jobs=1` creates no pool and leaves native
  settings unchanged. Exercise public wrappers and repeated PCA on a fitted
  transform.
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
   and Accelerate thread variables unset: omitted option (automatic count),
   explicit one worker, four workers, and eight workers. Environment
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
ordinary CI. Use these results to confirm the cap of four across input sizes
and that the automatic default does not slow restricted-CPU or outer-parallel
runs; if it does, fall back to an opt-in release. Separate solver
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

### Faster per-gene statistics (1.1 target)

Make the operator's column statistics cheaper, and let `pca(mask_var=...)`
reuse the statistics that `variable_statistics` computed. In the
highly-variable-gene workflow, `TransformedMatrix.variable_statistics` builds
an operator over every gene, reads its column means and centered squared
norms, and discards it. `pca(mask_var=...)` then builds a second operator over
the selected genes, whose constructor recomputes their statistics. In a
profile on a 100k-cell brain subset (200M stored counts, 20k genes, rank-1
low-rank part), `variable_statistics` spent 11.6 s in the operator
constructor: 7.2 s in the norm sweep (`_squared_norms_about`) and 4.0 s in
column means (`_column_means`). Most of that time is overhead that scales with
the number of blocks and genes, not with the stored entries.

- **Reuse statistics across calls.** Each gene's mean and centered squared
  norm depend only on its own column, so PCA over selected genes could slice
  statistics cached on the `TransformedMatrix`. Slices of statistics computed
  over all genes differ in low-order bits from a computation over only the
  selected genes. Two things cause this: row-block boundaries follow the
  selected matrix's stored entries, and dense columns are reconstructed in
  shared batches. Tentatively, PCA output must not depend on whether
  `variable_statistics` ran first, matching the reproducibility gate for
  parallel products. The decision stays open until the implementation
  measures what that costs. Meeting it needs one of two approaches. PCA could
  always slice statistics over all genes, computing them on first use, so a
  masked PCA run without `variable_statistics` pays for every gene. Or
  per-column results could stop depending on which other columns are present.
  If both cost too much, the alternative is to let PCA slice whatever is
  cached, with differences within the statistics' documented accuracy. Cached
  values remain valid as long as the support check that
  `share_structure=True` transforms run before each calculation passes.
- **Derive the uncentered norms rather than sweeping for them.** A centered
  operator computes norms about zero and about the mean in the same sweep,
  which doubles the per-entry work. `centered_variance_is_numerically_zero`
  still needs the uncentered total as its scale. For each column it follows
  from the centered norm as
  $\sum x^2 = \sum (x - c)^2 + c\,(2\sum x - n c)$, with $c$ the stored center
  and $\sum x$ the float64 column sum from the mean pass. Uncentered
  operators, which correspondence analysis uses, already sweep about zero
  only.
- **Size norm blocks relative to the gene count.** `_STATS_NORM_BLOCK_NNZ`
  (100,000) gives about 2,000 blocks on the profiled input. Each block
  allocates partial sums with one entry per gene and center, and runs
  `bincount` with `minlength=n_vars` four times, so its per-gene cost is about
  the same as its per-entry work. Dense columns also need a CSC copy of each
  block, which took 1.4 s in total. Blocks should hold several times `n_vars`
  stored entries while keeping the scratch bound that motivated the small
  target: the sweep gathers left- and right-factor rows for every stored
  entry in a block, so scratch grows with block entries times rank. Update the
  [budget table](development/architecture.md#bounded-computation-budgets) to
  match.
- **Cheaper dense-column means.** Columns stored in more than half the cells
  take their means directly, using `math.fsum` per column per block. On the
  profiled input, about 940 such genes made 190k `fsum` calls, which took
  2.3 s of the 4.0 s. The specification uses `fsum` because represented values
  can have mixed signs. Block results are combined with ordinary float64
  addition, so the protection covers cancellation within a block. A plain
  pairwise `np.sum` gives that up: `test_dense_means_preserve_signed_sum`
  places the column `[2**60, 1, -2**60]` in one block, and pairwise summation
  rounds it to 0 instead of 1. A faster reduction must keep this accuracy.
  One option is pairwise sums with a bound computed from the summed absolute
  values, which sends only the columns that fail it to `fsum`. That works only
  if the specification's accuracy requirement permits it.

Apart from the dense-mean reduction, these changes reorder summation or
derive values through exact identities. They move means, variances, and PCA
totals only by rounding, which belongs under other changes in the changelog.
Validate them separately
from the [parallel sparse products](#parallel-sparse-products-for-pca-11-target),
which also change summation order and compare against 1.0 statistics.

### Float32 accuracy of residual PCA

Find out why float32 Pearson-residual PCA departs further from its float64
result than Scanpy's float32 does, and reduce the gap without giving up the
memory float32 saves. The comparison metric is the one in the
[numerical agreement targets](development/testing.md#numerical-agreement-targets):
the maximum sign-aligned per-component relative L2 error of loadings and
scores, here measured between float32 and float64 runs of the same
implementation on the same counts.

The scp-bench sweep of sparse-count-pca 1.0.0 on the 10x E18 mouse brain
measured these errors. Runs used 1 thread on a Hetzner CCX43, 30 components,
residual PCA on 5,000 highly variable genes selected inside each workflow,
and log1p PCA on all genes. Singular values agreed with float64 to about
1e-6 everywhere, so the error is in the individual axes.

| Cells | SCP residual, max loading error | SCP residual, median over components | Scanpy residual, max loading error |
| ---: | ---: | ---: | ---: |
| 50k | 7.1e-5 | 1.5e-5 | 5.2e-5 |
| 100k | 3.9e-4 | 1.5e-5 | 1.4e-4 |
| 200k | 1.6e-4 | 3.0e-5 | out of memory |
| 400k | 2.8e-4 | 5.7e-5 | out of memory |
| 600k | 5.4e-4 | 7.8e-5 | out of memory |
| 800k | 3.9e-4 | 1.0e-4 | out of memory |

For log1p, SCP and Scanpy float32 errors are about equal at every size and
grow similarly, reaching 9.2e-5 (SCP) and 1.2e-4 (Scanpy) at 1M cells. The
worst residual components are not those with the smallest relative
singular-value gaps: at 200k cells and above, the worst component's gap to
its nearest neighbor is about 3e-2, while the smallest gap among the 30 is
2e-3 to 5e-3. The scp-bench runs retain every float32 and float64 output, so
these numbers can be recomputed per component from saved files.

Statistics are not the cause: column means and centered norms are computed
in float64 with `math.fsum` (`_operator.py`). Every operator product runs in
the operator dtype, though, and three float32 effects are candidates:

- **Cancellation between terms.** The residual representation is sparse plus
  rank one (`left = -sqrt(n)`, `right = sqrt(p / variance_scale)` in
  `_residuals.py`), then centered. `_matvec`, `_rmatvec`, `_matmat`, and
  `_rmatmat` add the sparse product, the rank-one product, and the centering
  correction in float32. Each term can be much larger than their sum, so
  float32 rounding in each term is amplified relative to the result. The
  log1p representation has rank zero, which fits log1p matching Scanpy while
  residuals do not.
- **Long float32 accumulations.** `self.S.T @ y` sums over every cell for
  each gene, and SciPy's sparse kernels accumulate in the data dtype. The
  centering term uses `np.sum(y, dtype=self.dtype)`, and `self.mean` is the
  float64 mean rounded to float32. Accumulation error grows with the number
  of cells, which fits the rising median error.
- **Single-precision Lanczos.** `svds` on a float32 operator runs ARPACK in
  single precision with `tol=0`, so the Krylov basis and its
  reorthogonalization carry float32 error independent of the products.

Separate the three with ablations that each change one thing and keep the
transformed matrix stored in float32:

1. Combine the sparse, rank-one, and centering terms in float64 and cast the
   product once. The rank-one and centering terms are vectors or `n x rank`
   arrays, so this costs little memory.
2. Accumulate sparse products in float64 while reading float32 data. Check
   first whether SciPy copies `S.data` to float64 for a float32 matrix times
   a float64 vector. Such a copy would forfeit the memory float32 saves, so a
   kernel reading float32 values into float64 accumulators would be needed.
3. Run ARPACK in float64 on the float32-stored operator by returning float64
   products, isolating solver precision from product precision.
4. Use the float64 mean, and a float64 sum of `y`, in the centering term.

For each ablation, report on the same brain subsets (50k, 100k, 400k, and
800k cells, prepared and selected as scp-bench does) the per-component
loading and score errors against float64, both maximum and median. Also
report median wall time and peak RSS at 400k, and singular values and total
variance against float64. A change is worth keeping if it brings SCP's
float32 residual error to Scanpy's or below where Scanpy fits (50k and 100k),
keeps the median error from growing with cell count, and keeps float32 peak
memory clearly below float64's; at 800k they were 28.9 and 37.4 GiB. If
ablation 1 alone closes the gap, it is the cheapest fix. Add a regression
test that fails at 1.0's error on a synthetic residual input where the
rank-one term dominates the represented values. Changes that only reorder
float32 arithmetic belong under other changes in the changelog, not fixes.

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
- A highly variable gene selection function built on
  `TransformedMatrix.variable_statistics`, including per-batch fitting and
  rank combination.
- Dask-backed inputs and block-aware transformations.
- Additional SVD solvers if they preserve deterministic and dtype contracts.
