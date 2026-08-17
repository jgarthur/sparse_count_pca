# AnnData functions

These functions mutate an `AnnData` object by default and return `None`. Pass
`copy=True` to return a modified copy instead.

## Residual PCA

::: sparse_count_pca.residual_pca

<!-- TODO(log1p): Add the log1p_norm_pca AnnData API entry. -->

## Shifted-CLR PCA

::: sparse_count_pca.shifted_clr_pca

::: sparse_count_pca.proportion_shifted_clr_pca

## Dirichlet PCA

These prior-based transforms have no external reference implementation
pinned; see the [transform catalog](../../transforms.md) for validation status.

::: sparse_count_pca.dirichlet_log_pca

::: sparse_count_pca.dirichlet_clr_pca

## Correspondence analysis

::: sparse_count_pca.correspondence_analysis
