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
limits. Ownership tests mutate caller-owned matrices after fitting and verify
that transformed values cannot change.

When adding a test, give every test function a docstring describing the
behavior it verifies. Every Python file, including examples and oracle scripts,
must have a module docstring.

The exhaustive verification requirements remain normative in the
[specification](specification.md#verification-contracts).
