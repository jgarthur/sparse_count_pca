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
before the parent establishes the RSS baseline. The timed and sampled boundary
is residual-representation construction alone. BLAS-related thread variables
are fixed at one for every worker. Clipped cases use
`clip=sqrt(n_obs / 30)` and `clip_max_nnz_ratio=1.0`; `--clip` can override the
threshold for a custom synthetic input. That is the numeric form of the
`clip="seurat"` default, passed explicitly so a baseline tree predating the
clipped default stays comparable; the unclipped cases likewise pass
`clip=None`.

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

Transpose products are independent contracts. A correct forward product does
not establish a correct adjoint.

## PCA and subspace equality

Compare singular values, explained variance, and reconstructed subspaces with a
dense SVD. Individual singular vectors are sign-ambiguous, and repeated or
nearly repeated singular values make basis vectors within a subspace ambiguous;
tests should use sign normalization or subspace comparisons as appropriate.

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

Tests that compare against an unclipped dense oracle or a pinned external
reference must pass `clip=None` explicitly; the residual default clips.

When adding a test, give every test function a docstring describing the
behavior it verifies. Every Python file, including examples and oracle scripts,
must have a module docstring.

The exhaustive verification requirements remain normative in the
[specification](specification.md#verification-contracts).
