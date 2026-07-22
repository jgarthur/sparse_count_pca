# sparse-count-pca

Single-cell transcriptomics data reaches the user as a large, very sparse
matrix of integer counts. Many normalization methods assign nonzero values to
observed zeros, making the transformed matrix dense and potentially increasing
entry storage by an order of magnitude or more.

Most workflows do not need every normalized entry at once, because their
primary consumer is PCA. `sparse-count-pca` represents supported count
transforms exactly as a sparse matrix plus a small low-rank term. This
factorization provides efficient matrix-vector and matrix-matrix products
without materializing the full transformed matrix. The package uses those
products to compute PCA of the column-centered transform through SciPy's
`LinearOperator` interface and truncated SVD.

The implementation is pure Python and is tested against independent
dense-matrix implementations and pinned external reference outputs. It covers
PCA of model residuals, fixed-count shifted logs, shifted centered log-ratio
(CLR) coordinates, proportion-shifted CLR coordinates, and classical
correspondence analysis.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install sparse-count-pca
```

## Quick start

The primary API writes Scanpy-compatible PCA outputs to an `AnnData` object:

```python
import sparse_count_pca as scp

scp.residual_pca(
    adata,
    layer="counts",
    n_comps=50,
    model="poisson",
    residual="pearson",
)
```

Scores are written to `adata.obsm["X_pca"]`, component vectors to
`adata.varm["PCs"]`, and variance statistics and parameters to
`adata.uns["pca"]`.

[Run a complete generated-data example](docs/getting-started.md)

## Supported transforms

| Transform | Public specification | Main use |
| --- | --- | --- |
| Residual | `Residual(model=..., residual=...)` | Pearson or deviance residual PCA under Poisson, binomial, or scaled-NB models |
| Fixed-count shifted log | `ShiftedLog(count_shift=...)` | PCA of `log1p(X / count_shift)` without library-size normalization |
| Fixed-count shifted CLR | `ShiftedCLR(count_shift=...)` | Within-observation log ratios; includes the PFlog parameterization of Booeshaghi et al. |
| Proportion-shifted CLR | `ProportionShiftedCLR(composition_shift=...)` | Historical fixed-composition-shift CLR formula |
| Correspondence analysis | separate one-step API | Classical contingency-table ordination |

`scaled_nb` is a package-specific name for a residual model whose expected
counts scale with each cell's total count and whose supplied per-gene `alpha`
values add negative-binomial overdispersion. The package does not estimate
`alpha`. The scaled-NB correspondence-analysis extension is experimental; see
the [residual PCA](docs/guides/residual-pca.md) and
[correspondence-analysis](docs/guides/correspondence-analysis.md) guides.

[Choose a transform](docs/choosing-a-transform.md)

## How it works

Supported dense transforms have an exact representation

```text
transformed matrix = sparse matrix + low-rank baseline.
```

The factorization combines sparse multiplication with small dense low-rank
products. Selecting PCA variables preserves this form, and column centering
adds one rank-one term. A `scipy.sparse.linalg.LinearOperator` exposes forward
and transpose matrix-vector and matrix-matrix products to
`scipy.sparse.linalg.svds`, currently using the ARPACK solver.

Optional Pearson or deviance residual clipping is also represented exactly;
symmetric clipping may add sparse corrections for zero-count entries.
"Exact" refers to the represented transform at the chosen floating-point
dtype. Truncated SVD is still a numerical iterative calculation. The current
backend owns an in-memory CSR correction; backed inputs are not yet processed
out of core.

[Read the mathematical explanation](docs/concepts/sparse-plus-low-rank.md)

## Choose an interface

### One-step AnnData API

Use functions such as `residual_pca`, `shifted_clr_pca`, and
`correspondence_analysis` to write results directly into an `AnnData` object.
Counts may come from `.X`, a layer, or `.raw.X`.

### One-step matrix API

Use the corresponding `_matrix` functions on dense or sparse matrices outside
AnnData. Non-CSR sparse inputs are converted to CSR. CSR inputs with duplicate
or unsorted indices, or with explicitly stored zeros, are copied before
canonicalization; already
[canonical CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.has_canonical_format.html)
input can be borrowed without mutation.

```python
result = scp.residual_pca_matrix(
    counts,
    n_comps=20,
    model="poisson",
    residual="deviance",
)

result.scores       # observations by components
result.components   # components by variables
result.loadings     # variables by components
```

### Two-step transform API

Fit once when transformed values must be inspected or several PCA masks should
share one normalization:

```python
transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
)

cell = transformed.materialize(obs=10)
subset = transformed.materialize(obs=[10, 3], var=[25, 2, 8])
result = transformed.pca(n_comps=50, mask_var="highly_variable")
```

[Transform once and reuse](docs/guides/transform-reuse.md)

## Important distinctions

- **A gene mask usually selects PCA columns, not normalization inputs.** For
  residual, shifted-log, and shifted-CLR PCA, normalization is fitted on the
  full chosen count matrix before `mask_var` selects variables for PCA. Slice
  the count matrix first if excluded genes should not affect cell totals, gene
  proportions, or CLR row means. In correspondence analysis, the mask instead
  defines the contingency table itself; expected counts depend on that table's
  margins, so the margins are recomputed after masking.
- **Scaled-NB residuals are not a drop-in SCTransform v2 implementation.** Do
  not expect default outputs to match. The compatibility guide describes the
  controlled conditions under which equality can be tested.
- **Shift domain matters.** A fixed raw-count shift and a fixed shift after
  library-size division are different CLR transforms. The PFlog normalization
  proposed by [Booeshaghi et al. in preprint version 4 (June 22,
  2026)](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4)
  uses the fixed-count form.
- **Precision is a computation choice.** Keep the recommended `float64`
  default for accuracy and parity testing. Explicit `float32` reduces memory
  but changes the representation passed to ARPACK.
- **Residual clipping is optional and exact.** Symmetric clipping of Pearson or
  deviance residuals can alter negative residuals for zero-count entries. The
  representation must then store those corrections, increasing sparse
  support. Clipping only the positive tail does not expand sparse support.

See [normalization, masking, and centering](docs/concepts/normalization-masking-and-centering.md),
[clipping and precision](docs/concepts/clipping-and-precision.md), and the
[compatibility reference](docs/reference/compatibility.md).

## Related work

The [`cleartools`](https://github.com/cleartools) projects provide dedicated
Rust-backed tooling for shifted CLR / PFlog workflows, including the Python
[`scclr`](https://github.com/cleartools/scclr) package.

[`10XGenomics/scan-rs`](https://github.com/10XGenomics/scan-rs) is a Rust
library used by Cell Ranger and contains similar matrix-representation and
normalization machinery.

## Documentation

- [Getting started](docs/getting-started.md)
- [Choosing a transform](docs/choosing-a-transform.md)
- [User guides](docs/guides/residual-pca.md)
- [Concepts](docs/concepts/sparse-plus-low-rank.md)
- [Compatibility](docs/reference/compatibility.md)
- [API reference](docs/reference/api/anndata.md)
- [Architecture](docs/development/architecture.md)
- [Normative specification](docs/development/specification.md)

## Development

```bash
uv sync --extra test --extra docs
uv run python -m pytest
uv run ruff check .
uv run mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for project invariants and documentation
expectations.
