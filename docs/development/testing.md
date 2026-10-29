# Testing strategy

Correctness is established at several levels because agreement of final PCA
plots alone is not enough to validate an implicit transform.

## Standard commands

```bash
uv sync --extra test --extra docs
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build --strict
```

CI runs all four. `ruff format` owns layout, so run `uv run ruff format .` to
apply it rather than wrapping lines by hand.

Normal tests are offline. External R and Python tooling used to regenerate
pinned reference artifacts is provenance, not a runtime test dependency.

## Count validation and log-construction memory

Validate count-check boundaries with invalid, fractional, and precision-limited
values in later blocks, preserving error precedence and input arrays. Tests
also cover canonical borrowing, copied noncanonical inputs, explicit zeros,
empty rows, extended precision, and the integer tolerance boundary.

Count-margin tests cover both axes, CSR matrix and array inputs, tiny and
default blocks, float32, float64, signed, and unsigned integer counts, empty
dimensions and support, explicit zeros, input preservation, integer
accumulator overflow, totals beyond float32 range, and genuine float64
overflow. Float32 counts must fit the same factors, representations, PCA, and
correspondence axes as an exact float64 copy, at either storage dtype.

For all five log builders and both storage dtypes, compare bounded construction
with independent dense oracles and verify that fitted factors agree exactly
with float64 storage. Tiny block targets exercise multi-block inputs, rows
longer than the target, and row sums across empty rows and blocks with no
stored entries. Check late ratio overflow, float64 ratios exceeding float32
range, read-only input arrays, int32/int64 input structure, owned output
structure at the required index width (including the int64 shape boundary),
and reuse of the builder's sparse matrix by the public transform. Residual
builders get the same index-width and ownership checks.
Compare each public log PCA with a dense SVD of the requested-dtype
representation using float64-fitted factors, including aligned
scores/loadings, mean, singular values, total variance, and variance ratios.
Returned-operator isolation checks also apply.

For allocation diagnostics, create input before starting tracemalloc and
reset tracing separately for validation and each construction stage. Use
canonical float32 counts with a fixed number of positive entries per row at
two nnz scales (for example one and two million), size factors varying across
rows, and a varying vector prior for Dirichlet CLR. Report the median of three
separately traced runs and distinguish retained output from temporary
allocation: peak allocation above retained output should stay roughly constant
as nnz doubles. Cross-check with sampled RSS and process high-water marks in
fresh workers with tracing disabled. Keep all thread limits at one;
generate/load input outside timed regions and report three-repeat timings for
validation, transform construction, and PCA independently. These measurements
describe small synthetic inputs rather than predicting full-data RSS, and RSS
figures include allocator behavior across repeated calls.

## Manual residual-memory probe

Memory-sensitive residual-construction changes should be checked with the
paired subprocess probe before merge. Point it at a baseline checkout and the
candidate checkout:

```bash
uv run python benchmarks/residual_memory.py \
  --baseline-source ../sparse_count_pca-main \
  --candidate-source .
```

The probe creates one deterministic sparse count matrix, saves it once, and
uses fresh workers for every residual family, clipping state, and calculation
dtype. Each worker loads that saved CSR and imports its selected source tree
before the parent establishes the RSS baseline. The default timed and sampled
boundary is residual-representation construction alone. BLAS-related thread
variables are fixed at one for every worker. Clipped cases use
`clip=sqrt(n_obs / 30)` and `clip_max_nnz_ratio=1.0`; `--clip` can override the
threshold for a custom synthetic input. The threshold is the numeric form of
the `clip="seurat"` default, passed explicitly to stay comparable with baseline
trees that predate the clipped default; the unclipped cases likewise pass
`clip=None`.

Use `--phase operator` to build the residual representation before the memory
baseline and isolate construction of the centered linear operator and its
summary statistics. Model, residual, dtype, and clipping filters keep large
manual runs focused. This example generates exactly 71.8 million stored values:

```bash
uv run python benchmarks/residual_memory.py \
  --baseline-source ../sparse_count_pca-main \
  --candidate-source . \
  --phase operator \
  --n-obs 100000 \
  --n-vars 30000 \
  --nnz-per-row 718 \
  --model poisson \
  --residual pearson \
  --dtype float64 \
  --clip-state none
```

The output reports sampled incremental peak RSS and the process-lifetime high
water mark as a cross-check. It intentionally has no machine-independent pass
threshold: compare the candidate with the baseline and with its own unclipped
configuration. Increase `--repeats` or the matrix dimensions when a change
needs a stronger signal. The high-water-mark delta is zero when the residual
build does not exceed the earlier import/load high-water mark; in that case,
use the sampled peak. Each worker has a five-minute timeout by default; adjust
it with `--timeout` for unusually large inputs.

## Executable documentation examples

Each user guide has a companion script under `examples/`, named after the guide
it belongs to. These scripts are the canonical source for the guides' complete
example workflows. They use Jupytext's percent format, so they remain ordinary
Python files while also defining notebook cells that readers can open directly
in Jupyter or VS Code.

Pytest discovers and executes every example script. A cold MkDocs build
independently converts each script to a notebook, executes its cells in order,
and renders the result to a Markdown fragment under `docs/examples/`. Those
fragments are excluded from the site's own pages and are pulled into the
matching guide with a `pymdownx.snippets` include, so a guide's complete
example is always code that ran for its current inputs. The fragments are build
artifacts and are not edited or committed.

The renderer caches each notebook using the example source, package source,
documentation dependency lockfile, and rendering hook. This keeps
Markdown-only live reloads fast while invalidating outputs when executable
inputs change. A clean CI checkout has no cache and executes every notebook.
Force the same behavior locally with:

```bash
DOCS_FORCE_EXAMPLES=1 uv run mkdocs build --strict
```

Adding a guide therefore means adding its companion script; renaming one means
renaming both the script and the guide's include.

The documentation build also rejects `$$` display delimiters that survive as
plain HTML paragraphs. Within Markdown lists, display-math blocks must use the
four-space continuation indentation required for Arithmatex to process them.

## Dense transform oracles

Every transform should be compared entry by entry with an independent explicit
dense implementation on small matrices and on committed real-data fixtures.
Tests must cover structural zeros, observed nonzeros, masking, supported
models, and both calculation dtypes where behavior differs.

The deterministic PBMC3k-derived
[real-data fixtures](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/real_data_reference)
include observed-depth and equal-depth variants. Simulated and real-data tests
share the same dense formula implementations so the larger fixture extends,
rather than replaces, the small-matrix oracle contracts.

## Operator parity

For each sparse-plus-low-rank representation, compare:

- materialized values with the dense oracle;
- `matvec` with dense `A @ x`;
- `rmatvec` with dense `A.T @ y`;
- matrix-matrix products where exposed;
- centered means and Frobenius-norm statistics.

Statistics tests cross CSR row-block boundaries for rank zero and both CSR
and dense column-batch boundaries for ranks one and three, with both operator
dtypes and centered and uncentered operators. Include empty, sparse, exactly-half-supported, and
dense-majority columns, constant and nearly constant columns, and signed
means with severe cancellation. Verify that dense batches respect their
represented-value target, with a one-column exception when a row block alone
exceeds that target. Cancellation-fallback and numerical-zero tests are also
part of the contract. A small normalized-log1p PCA comparison against
Scanpy checks aligned scores and loadings, singular values, total variance,
and explained-variance ratios, while rejecting dense reconstruction. Rank-zero
tests also cover explicit stored zeros, read-only borrowed data, skipped mean
work for uncentered operators, and subtraction rounding in float32.

Transpose products are independent contracts. A correct forward product does
not establish a correct adjoint.

### Operator batch-size measurements

The dense-batch target was chosen from a sweep over deterministic
2,000 × 1,500 and 4,000 × 3,000 count matrices with 20% of genes stored in 75%
of rows and the others in 2% (498,000 and 1,992,000 stored entries). Unclipped
Poisson Pearson residuals (rank one) were measured in float32 and float64. All
thread limits were one, representation construction was outside the timed
region, and CSR block targets were held fixed. Times are medians of five
repeats with randomized batch-target order, measuring operator construction
rather than PCA or transformation. Nonnegative rank-zero representations do
not use dense batches, so they do not inform this target.

Representative results for the larger fixture:

| Dtype | 100,000 values | 262,144 values | 1,048,576 values |
| --- | ---: | ---: | ---: |
| float32 | 113 ms | 112 ms | 110 ms |
| float64 | 122 ms | 118 ms | 118 ms |

The sweep also included targets of 16,384, 65,536, and 4,194,304 values. The
262,144-value target stayed within about 2% of the fastest tested target
across both sizes and dtypes, and bounds each float64 dense buffer at 2 MiB
except for the indivisible one-column case. Larger targets multiply that
buffer budget for small additional speed gains; smaller targets incur more
reconstruction and sparse-slicing calls. Means dominated the remaining setup
time.

For a nominal 1,000,000-cell × 20,000-gene input at 10% density, there are
about two billion stored entries. Assuming similar row depths, the default
mean and norm targets produce roughly 500-row and 50-row blocks, respectively.
Fewer than 4,000 genes can be dense-majority at that overall density, so a
typical norm block's direct reconstruction fits in one batch, while mean
blocks require several. Uniformly sparse genes do not use direct dense
batching; dense-majority means of nonzero-rank representations still incur
per-column `math.fsum` calls, which a larger batch target does not eliminate.

These small local measurements support a conservative default, not an optimum
for every machine or support distribution, nor a prediction of full-data
runtime. Batch targets do not affect the numerical tolerances but may change
low-order bits of factor products and reductions.

## PCA and subspace equality

Compare singular values, scores, loadings, and explained variance with a dense
SVD. Preserve component order and account for the joint sign ambiguity of each
score/loading pair. Repeated or nearly repeated singular values can make the
individual axes unstable; inspect subspace diagnostics when investigating such
disagreements without rotating the outputs to satisfy the agreement checks.

### Numerical agreement targets

For PCA implementation comparisons on identical counts and equivalent transforms,
use the following relative-error targets. They are development and benchmark
checks, not a public guarantee for every dataset or solver tolerance. Preserve
stricter checks where they verify a more specific invariant.

| Metric | float32 | float64 |
| --- | ---: | ---: |
| Maximum relative singular-value error | `1e-5` | `1e-12` |
| Sign-aligned score relative Frobenius error | `1e-5` | `1e-12` |
| Sign-aligned loading relative Frobenius error | `1e-5` | `1e-12` |

Select the threshold by calculation/storage precision, not the dtype of the
returned singular-value array. Evaluate comparison arithmetic in float64.
For reference singular values `s`, require `max(abs(s_candidate - s) / abs(s))`
to meet the target. A zero reference requires an exactly zero candidate.

Let `L` contain genes by components and `S` contain cells by components. For
each component in its returned order, choose `-1` if the dot product of the
candidate and reference loading vectors is negative, and `+1` otherwise.
Multiply that candidate loading column and its score column by the same sign.
Compare each resulting matrix using
`||candidate - reference||F / ||reference||F`. A zero reference norm requires
zero error. Use `<=` for acceptance. Components are neither reordered nor mixed
by rotations: individual gene weights are part of the comparison. An entrywise
`allclose` with an absolute allowance is a different metric and need not enforce
these relative targets.

Principal angles remain useful diagnostics, without a separate pass/fail
threshold. Variance statistics and transform/operator formulas retain their
own checks. Same-input, same-environment, same-settings repeatability requires
bitwise equality separately; alignment is not part of that check.

These targets are informed by the paired UMI subsampling, permutation, and
archived Scanpy comparisons in the companion repository's
`projects/scp-tolerance-study/results/findings.md`. They express numerical
agreement, not conservative nearest-neighbor stability; the study's all-cell
neighborhood criterion is stricter, especially at float32. Keep that analysis
outside CI. Use the small dense-SVD and Scanpy tests for routine agreement
checks, with the shared helper in `tests/_oracles.py`.

## External parity tests

Pinned references cover:

- Scanpy AnnData conventions;
- controlled SCTransform v2 residual equality;
- current count-scale PFlog values;
- the historical composition-scale shifted CLR formula;
- classical correspondence analysis through independent R implementations;
- real-data dense oracles for every public transform family.

Every artifact directory includes provenance, generation commands, pinned
versions or commits, and integrity checks where appropriate.

## Boundary and ownership tests

Validation tests cover invalid counts, masks, dimensions, parameters, dtypes,
zero totals, zero-variance transforms, clipping thresholds, and support-growth
limits. Named clipping thresholds are covered against their numeric
equivalents, across both clipping modes, entry points, residual families, and
observation subsets. Ownership tests mutate caller-owned matrices after fitting
and verify that transformed values cannot change.

Structure-sharing tests run every support-preserving transform at both storage
dtypes and both index widths. Default transforms must own their support;
`share_structure=True` must borrow indices and indptr at either input width,
including int64 support on matrices int32 could index, and fitted values must
survive mutation of input data and deletion of the input object. After entry
removal or a direct index write, a shared transform must refuse `pca` and
`materialize`; after entry removal, all four product directions must also
refuse. Value changes and owned transforms must pass these checks. A retained
operator from a sharing transform must not alias the input or the transform.
One-step analyses must borrow only when they do not return an operator.
Residual sharing tests cover both residual types, both clipping modes,
thresholds with and without support growth, and both storage dtypes. They
require borrowing exactly when the support is unchanged and identical values to
an owned fit. Forced zero corrections, from extreme counts split across tiny
support blocks and from values that underflow only in float32, must be removed
from an owned copy without touching the read-only input. Owned clipping that
grows support must add its corrections to read-only input support without
modifying or retaining it. Further cases cover canonicalization copies, AnnData
layers and masks, read-only inputs, which keep their flags, and rejection of
non-boolean flags.

Tests that compare against an unclipped dense oracle or a pinned external
reference must pass `clip=None` explicitly, because residual transforms clip
by default.

When adding a test, give every test function a docstring describing the
behavior it verifies. Every Python file, including examples and oracle scripts,
must have a module docstring.

The exhaustive verification requirements remain normative in the
[specification](specification.md#verification-contracts).
