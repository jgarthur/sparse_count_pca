# Correspondence analysis

Classical correspondence analysis (CA) analyzes dependence in a contingency
table. It decomposes a standardized Pearson-residual matrix without ordinary
PCA column centering, then reports coordinates scaled by the row and column
masses.

## Matrix workflow

```python
import sparse_count_pca as scp

result = scp.correspondence_analysis_matrix(counts, n_comps=2)

result.row_principal_coordinates
result.column_principal_coordinates
result.principal_inertias
result.inertia_ratio
```

`total_inertia` equals Pearson chi-squared divided by the grand total. Standard
coordinates are not stored; derive them by dividing principal coordinates by
the corresponding singular values.

## AnnData workflow

```python
scp.correspondence_analysis(
    adata,
    layer="counts",
    n_comps=2,
)
```

The default keys are:

| Location | Contents |
| --- | --- |
| `adata.obsm["X_ca"]` | row principal coordinates |
| `adata.varm["CA"]` | column principal coordinates |
| `adata.uns["ca"]` | singular values, inertias, masses, and parameters |

Passing `key_added="my_ca"` uses that exact key in all three mappings.

## Variable masks change the table

Unlike PCA transforms, a CA variable mask is applied before fitting. Row and
column totals, masses, expected values, and inertia are recomputed from the
selected contingency table. Removing a variable and masking it therefore have
the same margin-defining role for CA, but not for residual or CLR PCA.

## Experimental scaled-NB residual ordination

Passing `model="scaled_nb"` replaces the Poisson variance with the package's
exposure-scaled negative-binomial variance:

```python
result = scp.correspondence_analysis_matrix(
    counts,
    n_comps=2,
    model="scaled_nb",
    alpha=overdispersion,
)
```

This is a correspondence-like residual ordination, not classical
correspondence analysis. It emits a warning and records `experimental=True`.
The reported `total_inertia` is scaled-NB Pearson-residual inertia; it does not
have the classical chi-square or barycentric interpretation. `scaled_nb` is a
package-specific model name; its mean and variance are defined in the
[residual-PCA guide](residual-pca.md#choose-the-count-model).
