# PCA of sparse count transforms without the dense matrix

`sparse-count-pca` computes PCA and related spectral analyses of count
transforms that would ordinarily be dense. Supported transforms are represented
exactly as a sparse matrix plus a small low-rank term, then decomposed through a
SciPy linear operator without materializing the full transformed matrix.

The package is designed for sparse single-cell and spatial-transcriptomics
counts, while its matrix interfaces also apply to general nonnegative count
tables.

## Minimal example

The primary interface writes Scanpy-compatible PCA results to an existing
`AnnData` object:

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

[Install and run a complete example](getting-started.md)

## Choose your path

- **I want residual PCA.** Start with the
  [residual PCA guide](guides/residual-pca.md).
- **I want shifted-log or shifted-CLR PCA.** See
  [shifted log and CLR](guides/shifted-log-and-clr.md).
- **I want correspondence analysis.** See the
  [correspondence-analysis guide](guides/correspondence-analysis.md).
- **I want to inspect a transform or reuse it with several masks.** Use the
  [two-step transform workflow](guides/transform-reuse.md).
- **I need to work carefully with layers, raw counts, masks, or result keys.**
  See [AnnData workflows](guides/anndata-workflows.md).
- **I am not sure which transform to use.** Read
  [choosing a transform](choosing-a-transform.md).

## Core idea

Many count transformations assign nonzero values to entries that were zero in
the observed matrix. The transformed matrix is therefore dense even when the
input is sparse. For the supported methods, those zero baselines factor into a
small number of row and column vectors:

```text
transformed matrix = sparse correction + low-rank baseline
```

Ordinary PCA centering adds one more rank-one term. Matrix-vector products with
that representation are exact, so SciPy's truncated SVD can operate on it
without allocating every transformed entry. The benefit depends on the input
density, dtype, transform, and clipping behavior.

[Read the sparse-plus-low-rank explanation](concepts/sparse-plus-low-rank.md)

## Documentation

- [Getting started](getting-started.md) provides an installation path and a
  complete small analysis.
- [Guides](guides/residual-pca.md) show task-oriented workflows.
- [Concepts](concepts/sparse-plus-low-rank.md) explain the mathematical and
  numerical ideas.
- [Compatibility](reference/compatibility.md) states controlled equalities and
  important non-equivalences with related tools.
- [API reference](reference/api/anndata.md) documents exact signatures and
  public objects from their Python docstrings.

## For contributors and reviewers

The [architecture guide](development/architecture.md) maps the implementation,
the [package specification](development/specification.md) is normative, and the
[testing guide](development/testing.md) describes the oracle and parity
strategy.
