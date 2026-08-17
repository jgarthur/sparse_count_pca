# Transform specifications

Use these immutable specifications with `transform` to separate normalization
from materialization and PCA.

::: sparse_count_pca.transform

::: sparse_count_pca.Transform
    options:
      members: false

::: sparse_count_pca.Residual
    options:
      members: false

<!-- TODO(log1p): Add the Log1pNormalized transform specification. -->

::: sparse_count_pca.ShiftedCLR
    options:
      members: false

::: sparse_count_pca.ProportionShiftedCLR
    options:
      members: false

The following prior-based transform specifications have no external reference
implementation pinned; see the [transform catalog](../../transforms.md) for validation status.

::: sparse_count_pca.DirichletLog
    options:
      members: false

::: sparse_count_pca.DirichletCLR
    options:
      members: false

## Fitted matrix

::: sparse_count_pca.TransformedMatrix
    options:
      members:
        - materialize
        - pca
