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

Cell embeddings/scores are written to `adata.obsm["X_pca"]`, component vectors
to `adata.varm["PCs"]`, and variance statistics and parameters to
`adata.uns["pca"]`.

[Run a complete example with simulated data](https://sparse-count-pca.readthedocs.io/en/latest/getting-started/)

## Supported transforms

| Transform | AnnData function | Main use |
| --- | --- | --- |
| Residual PCA | `residual_pca(...)` | Pearson or deviance residual PCA under Poisson, binomial, or scaled-NB (see note below) |
| Count-scale shifted CLR | `shifted_clr_pca(...)` | Within-observation log ratios; includes the PFlog parameterization of Booeshaghi et al. (2026) |
| Composition-scale shifted CLR | `proportion_shifted_clr_pca(...)` | CLR after a fixed shift on the composition scale |
| Correspondence analysis | `correspondence_analysis(...)` | Classical contingency-table ordination |

[Compare transform assumptions, formulas, provenance, and API
maturity](https://sparse-count-pca.readthedocs.io/en/latest/transforms/)

Already using Scanpy's Pearson-residual preprocessing? See
[coming from Scanpy][scanpy-compatibility] for what each call maps to, and
[what this package does not do][omitted-transforms] for the transforms it
deliberately omits.

`scaled_nb` is a package-specific name for a negative-binomial model that
scales overdispersion inversely with cell depth, as in the sSeq model from
[Yu, Huber, and Vitek (2013)](https://doi.org/10.1093/bioinformatics/btt143). It
is not an implementation of
[SCTransform](https://doi.org/10.1186/s13059-019-1874-1); its Pearson residuals
agree with SCTransform's only under conditions that ordinary data does not meet.
See the [scaled-NB null model][scaled-nb-model].

There is also an experimental scaled-NB extension to correspondence analysis;
see the
[correspondence-analysis guide][correspondence-analysis-guide].

## How it works

Supported dense transforms have an exact representation,

```text
transformed matrix = sparse matrix + low-rank baseline,
```

which allows efficient factored matrix multiplication without materializing the
actual matrix. Selecting PCA variables preserves this form, and column centering
adds one rank-one term. This package uses `scipy.sparse.linalg.LinearOperator`
to expose matrix-vector and matrix-matrix products to
`scipy.sparse.linalg.svds`, currently using the ARPACK solver. See
[sparse plus low rank][sparse-plus-low-rank] for more details.

## API structure

- **One-step AnnData API.** Functions such as `residual_pca`,
  `shifted_clr_pca`, and `correspondence_analysis` write results directly into
  an `AnnData` object, as in the quick start above. Counts may come from `.X`
  or a layer. See [AnnData workflows][anndata-workflows].
- **One-step matrix API.** The corresponding `_matrix` functions take dense or
  sparse matrices outside AnnData and return a result object:

  ```python
  result = scp.residual_pca_matrix(counts, ...)

  result.scores       # observations by components
  result.components   # components by variables
  result.components.T # variables by components
  ```

- **Two-step transform API.** `transform` fits a normalization once, so
  transformed values can be inspected in bounded slices, or several PCA masks
  can share one fitted state. See
  [transform once and reuse][transform-reuse].

## Important distinctions

- **A gene mask usually selects PCA columns, not normalization inputs.** Slice
  the count matrix first if excluded genes should not affect cell totals, gene
  proportions, or CLR row means. Correspondence analysis is the exception: its
  mask defines the contingency table, so margins are recomputed after masking.
  See [normalization, masking, and centering][normalization-masking-centering].
- **Scaled-NB residuals are not a drop-in SCTransform v2 implementation.** Do
  not expect default outputs to match. See the
  [compatibility reference][sctransform-compatibility] for the conditions under
  which equality can be tested.
- **Symmetric residual clipping can expand sparse support**, because it alters
  negative residuals at zero-count entries that the low-rank baseline would
  otherwise carry. Upper-only clipping cannot. See
  [clipping and precision][clipping-and-precision].
- **Precision is a computation choice.** Keep the recommended `float64`
  default for accuracy and parity testing. Explicit `float32` reduces memory
  but changes the representation passed to ARPACK.

## Related work

The [`cleartools`](https://github.com/cleartools) projects provide sparse
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
uv run ruff format --check .
uv run mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for project invariants and documentation
expectations.

[anndata-workflows]: https://sparse-count-pca.readthedocs.io/en/latest/guides/anndata-workflows/
[clipping-and-precision]: https://sparse-count-pca.readthedocs.io/en/latest/concepts/clipping-and-precision/
[correspondence-analysis-guide]: https://sparse-count-pca.readthedocs.io/en/latest/guides/correspondence-analysis/
[normalization-masking-centering]: https://sparse-count-pca.readthedocs.io/en/latest/concepts/normalization-masking-and-centering/
[omitted-transforms]: https://sparse-count-pca.readthedocs.io/en/latest/transforms/#what-this-package-does-not-do
[scaled-nb-model]: https://sparse-count-pca.readthedocs.io/en/latest/reference/scaled-nb-model/
[scanpy-compatibility]: https://sparse-count-pca.readthedocs.io/en/latest/reference/compatibility/#coming-from-scanpy
[sctransform-compatibility]: https://sparse-count-pca.readthedocs.io/en/latest/reference/compatibility/#sctransform-v2
[sparse-plus-low-rank]: https://sparse-count-pca.readthedocs.io/en/latest/concepts/sparse-plus-low-rank/
[transform-reuse]: https://sparse-count-pca.readthedocs.io/en/latest/guides/transform-reuse/
