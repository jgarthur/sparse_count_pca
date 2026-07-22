# AnnData functions

These functions mutate an `AnnData` object by default and return `None`. Pass
`copy=True` to return a modified copy instead.

## Residual PCA

::: sparse_count_pca.residual_pca

## Shifted-log and CLR PCA

::: sparse_count_pca.shifted_log_pca

::: sparse_count_pca.shifted_clr_pca

::: sparse_count_pca.proportion_shifted_clr_pca

## Experimental Dirichlet PCA

These prior-based transforms are experimental and are not part of the
recommended starting workflows.

::: sparse_count_pca.dirichlet_log_pca

::: sparse_count_pca.dirichlet_clr_pca

## Correspondence analysis

::: sparse_count_pca.correspondence_analysis
