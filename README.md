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
PCA of GLM model residuals, shifted centered log-ratio
(CLR) coordinates, classical correspondence analysis, and several more
experimental options.

## Installation

Python 3.10 or newer is required.

```bash
pip install sparse-count-pca
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

Cell embeddings/scores are written to `adata.obsm["X_pca"]`, component vectors to
`adata.varm["PCs"]`, and variance statistics and parameters to
`adata.uns["pca"]`.

[Run a complete generated-data example](docs/getting-started.md)

## Supported transforms

| Transform | AnnData function | Main use |
| --- | --- | --- |
| Residual PCA | `residual_pca(...)` | Pearson or deviance residual PCA under Poisson, binomial, or scaled-NB (see note below) |
| Count-scale shifted log | `shifted_log_pca(...)` | PCA of `log1p(X / count_shift)` without library-size normalization |
| Count-scale shifted CLR | `shifted_clr_pca(...)` | Within-observation log ratios; includes the PFlog parameterization of Booeshaghi et al. |
| Composition-scale shifted CLR | `proportion_shifted_clr_pca(...)` | CLR after a fixed shift on the composition scale |
| Correspondence analysis | `correspondence_analysis(...)` | Classical contingency-table ordination |

[Compare transform assumptions, formulas, provenance, and API maturity](docs/transforms.md)

Already using Scanpy's Pearson-residual preprocessing? See
[coming from Scanpy](docs/reference/compatibility.md#coming-from-scanpy) for
what each call maps to, and
[what this package does not do](docs/transforms.md#what-this-package-does-not-do)
for the transforms it deliberately omits.

`scaled_nb` is a package-specific name for a negative-binomial model that
scales overdispersion inversely with cell depth, as in the sSeq model from
[Yu, Huber, and Vitek (2013)](https://doi.org/10.1093/bioinformatics/btt143). If
`s_i = cell_total_i / mean_cell_total`, the overdispersion for gene
`j` in cell `i` is `alpha_j / s_i`. At positive overdispersion, Pearson residuals
from this model correspond with
[SCTransform](https://doi.org/10.1186/s13059-019-1874-1) only when every cell has the
same total count, with additional model-fitting and post-processing choices
matched. See the [residual-PCA model definition](docs/guides/residual-pca.md#choose-the-count-model)
and its [SCTransform comparison](docs/guides/residual-pca.md#relationship-to-sctransform).

There is also an experimental scaled-NB extension to correspondence analysis; see the
[correspondence-analysis guide](docs/guides/correspondence-analysis.md).


## How it works

Supported dense transforms have an exact representation,

```text
transformed matrix = sparse matrix + low-rank baseline,
```

which allows efficient factored matrix multiplication without materializing the
actual matrix. Selecting PCA variables preserves this form, and column centering
adds one rank-one term. This package uses `scipy.sparse.linalg.LinearOperator` to
expose matrix-vector and matrix-matrix products to
`scipy.sparse.linalg.svds`, currently using the ARPACK solver. See
[sparse plus low rank](docs/concepts/sparse-plus-low-rank.md) for more details.

Pearson or deviance residual matrices are often clipped to exclude extreme
outliers. The package also represents this clipping exactly, although clipping
large negative residuals at observed zeros can increase the stored sparse
support. See [clipping and precision](docs/concepts/clipping-and-precision.md).

## API structure

- **One-step AnnData API.** Functions such as `residual_pca`,
  `shifted_clr_pca`, and `correspondence_analysis` write results directly into
  an `AnnData` object, as in the quick start above. Counts may come from `.X`,
  a layer, or `.raw.X`. See [AnnData workflows](docs/guides/anndata-workflows.md).
- **One-step matrix API.** The corresponding `_matrix` functions take dense or
  sparse matrices outside AnnData and return a result object.
- **Two-step transform API.** `transform` fits a normalization once, so
  transformed values can be inspected in bounded slices, or several PCA masks
  can share one fitted state. See
  [transform once and reuse](docs/guides/transform-reuse.md).

The matrix functions return their results instead of writing them:

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

All three paths canonicalize counts identically. Non-CSR sparse input is
converted to CSR, and CSR input with duplicate or unsorted indices, or with
explicitly stored zeros, is copied before canonicalization. Already
[canonical CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.has_canonical_format.html)
input is borrowed without mutation.

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
  uses the count-scale form, recommended by the authors over the previous
  proportion-scale shift.
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
Rust-backed tooling for shifted CLR (PFlog) workflows, including the Python
[`scclr`](https://github.com/cleartools/scclr) package.

[`10XGenomics/scan-rs`](https://github.com/10XGenomics/scan-rs) is a Rust
library used by Cell Ranger and contains similar matrix-representation and
normalization machinery.

## Development

```bash
uv sync --extra test --extra docs
uv run python -m pytest
uv run ruff check .
uv run mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for project invariants and documentation
expectations.
