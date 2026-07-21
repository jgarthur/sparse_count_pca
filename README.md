# sparse-count-pca

Single-cell transcriptomics data reaches the user as a large, very sparse
matrix of integer counts. Many normalization methods assign nonzero values to
observed zeros, making the transformed matrix dense and potentially increasing
entry storage by an order of magnitude or more, depending on the original
density and dtypes.

Most workflows do not need every normalized entry at once: their primary
consumer is PCA. `sparse-count-pca` represents supported count transforms
exactly as a sparse matrix plus a small low-rank term, then computes centered
truncated SVD through SciPy's `LinearOperator` interface without materializing
the full transformed matrix.

The implementation is pure Python and is tested against independent dense
oracles and pinned external references. It covers residual PCA, fixed-count
shifted log and shifted CLR, proportion-shifted CLR, Dirichlet-prior log and CLR
transforms, and correspondence analysis.

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
| Fixed-count shifted CLR | `ShiftedCLR(count_shift=...)` | Within-observation log ratios; includes current PFlog parameterization |
| Proportion-shifted CLR | `ProportionShiftedCLR(composition_shift=...)` | Historical fixed-composition-shift CLR formula |
| Dirichlet log / CLR | `DirichletLog(...)`, `DirichletCLR(...)` | Log or CLR coordinates of posterior-mean compositions under an explicit prior |
| Correspondence analysis | separate one-step API | Classical contingency-table ordination and an experimental scaled-NB variant |

[Choose a transform](docs/choosing-a-transform.md)

## How it works

Supported dense transforms have an exact representation

```text
transformed matrix = sparse correction + low-rank baseline.
```

Selecting PCA variables preserves this form, and column centering adds one
rank-one term. The package implements forward and transpose products with the
resulting matrix and passes a `scipy.sparse.linalg.LinearOperator` to
`scipy.sparse.linalg.svds` with the ARPACK solver.

"Exact" refers to the represented transform and clipping behavior at the
chosen floating-point dtype. Truncated SVD is still a numerical iterative
calculation. The current backend owns an in-memory CSR correction; backed
inputs are not yet processed out of core.

[Read the mathematical explanation](docs/concepts/sparse-plus-low-rank.md)

## Choose an interface

### One-step AnnData API

Use functions such as `residual_pca`, `shifted_clr_pca`, and
`correspondence_analysis` to write results directly into an `AnnData` object.
Counts may come from `.X`, a layer, or `.raw.X`.

### One-step matrix API

Use the corresponding `_matrix` functions outside AnnData:

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

- **Masking and normalization are separate.** For PCA transforms,
  normalization is fitted on the full chosen count matrix before `mask_var`
  selects PCA variables. Correspondence analysis instead recomputes table
  margins after masking.
- **Scaled-NB residuals are not default SCTransform v2.** Controlled equality
  requires aligned means, dispersions, variance floors, clipping, and
  centering.
- **Shift domain matters.** A fixed raw-count shift and a fixed shift after
  library-size division are different CLR transforms. Current PFlog uses the
  fixed-count form.
- **Precision is a computation choice.** `float64` is the default. Explicit
  `float32` changes the representation passed to ARPACK; it is not merely
  compact output storage.
- **Clipping remains exact.** Symmetric clipping can change transformed zeros
  and expand sparse support; upper-only clipping cannot for the supported
  residual models.

See [normalization, masking, and centering](docs/concepts/normalization-masking-and-centering.md),
[clipping and precision](docs/concepts/clipping-and-precision.md), and the
[compatibility reference](docs/reference/compatibility.md).

## Related work

The [`cleartools`](https://github.com/cleartools) projects provide dedicated
Rust-backed tooling for shifted CLR / PFlog workflows, including the Python
[`scclr`](https://github.com/cleartools/scclr) package.

[`10XGenomics/scan-rs`](https://github.com/10XGenomics/scan-rs) is a Rust
library used by Cell Ranger and contains related matrix-representation and
normalization machinery. `sparse-count-pca` is an independent pure-Python
implementation; no output-parity claim with `scan-rs` is currently made.

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
