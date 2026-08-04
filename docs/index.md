# PCA and correspondence analysis of sparse counts without dense matrices

`sparse-count-pca` computes PCA and correspondence analysis directly from
sparse count matrices, avoiding the dense normalized or residual matrices these
analyses would ordinarily require.

The package is designed primarily for sparse single-cell and
spatial transcriptomics counts stored in AnnData, but also provides a matrix
interface for general tables of counts.

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

- **I already use Scanpy's Pearson-residual preprocessing.** See
  [coming from Scanpy](reference/compatibility.md#coming-from-scanpy) for what
  each call maps to and where the models differ.
- **I need to pick a transform.** The [transform catalog](transforms.md)
  compares distinctions that affect interpretation, formulas, provenance, and
  exact interface names.
- **I know which analysis I want.** Guides cover
  [residual PCA](guides/residual-pca.md),
  [shifted CLR](guides/shifted-clr.md), and
  [correspondence analysis](guides/correspondence-analysis.md);
  [AnnData workflows](guides/anndata-workflows.md) covers count sources,
  variable selection, and output keys.
- **I want to inspect normalized values or reuse one normalization with several
  PCA masks.** Use the
  [two-step transform workflow](guides/transform-reuse.md).

## Core idea

Many count transformations assign nonzero values to entries that were zero in
the observed matrix, so the transformed matrix is dense even when the input is
sparse. For the supported methods, those normalized zero entries factor into a
small low-rank term, leaving a sparse correction plus a low-rank baseline.
Column centering adds one more rank-one term, and SciPy's truncated SVD
decomposes the result without allocating every entry.

[Read the sparse-plus-low-rank explanation](concepts/sparse-plus-low-rank.md)

## For contributors and reviewers

The [architecture guide](development/architecture.md) explains the
implementation, the [package specification](development/specification.md)
defines expected behavior, and the [testing guide](development/testing.md)
describes the oracle and parity strategy.
