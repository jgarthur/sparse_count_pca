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

::: sparse_count_pca.ShiftedLog
    options:
      members: false

::: sparse_count_pca.ShiftedCLR
    options:
      members: false

::: sparse_count_pca.ProportionShiftedCLR
    options:
      members: false

The following prior-based transform specifications are experimental and are
not part of the recommended starting workflows.

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
