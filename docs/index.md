# PCA and related analyses of sparse counts without dense matrices

`sparse-count-pca` computes PCA and related spectral analyses directly from
sparse count matrices, avoiding the dense normalized or residual matrices these
analyses would ordinarily require. For supported methods, the matrix being
analyzed is represented exactly as a sparse matrix plus a small low-rank term,
then decomposed through a SciPy linear operator without materializing every
entry.

The package is designed for sparse single-cell and spatial-transcriptomics
counts, while its matrix interfaces also apply to general tables of counts.

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

- **I want PCA on Pearson or deviance residuals.** Start with the
  [residual PCA guide](guides/residual-pca.md).
- **I want PCA on shifted logs or centered log-ratio (CLR) coordinates.** See
  [shifted log and CLR](guides/shifted-log-and-clr.md).
- **I want correspondence analysis.** See the
  [correspondence-analysis guide](guides/correspondence-analysis.md).
- **I want to inspect normalized values or reuse one normalization with several
  PCA masks.** Use the
  [two-step transform workflow](guides/transform-reuse.md).
- **I need to choose a count source, select variables, or control output keys.**
  See [AnnData workflows](guides/anndata-workflows.md).
- **I want to compare the supported analyses.** Read
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
without allocating every transformed entry. Runtime and memory benefits depend
on input density, dtype, transform, and clipping behavior.

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

The [architecture guide](development/architecture.md) explains the
implementation, the [package specification](development/specification.md)
defines expected behavior, and the [testing guide](development/testing.md)
describes the oracle and parity strategy.
