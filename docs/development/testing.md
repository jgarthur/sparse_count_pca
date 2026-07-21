# Testing strategy

Correctness is established at several levels because agreement of final PCA
plots alone is not enough to validate an implicit transform.

## Standard commands

```bash
uv sync --extra test --extra docs
uv run python -m pytest
uv run ruff check .
uv run mkdocs build --strict
```

Normal tests are offline. External R and Python tooling used to regenerate
pinned reference artifacts is provenance, not a runtime test dependency.

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
- current count-shift PFlog values;
- the historical proportion-shifted CLR formula;
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
