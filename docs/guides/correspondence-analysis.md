# Correspondence analysis

Classical correspondence analysis (CA) analyzes dependence in a contingency
table. It decomposes a standardized Pearson-residual matrix without ordinary
PCA column centering, then reports coordinates scaled by the row and column
masses.

This guide's [complete example](#complete-example) is
[`examples/correspondence_analysis.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/correspondence_analysis.py).

## Matrix workflow

```python
import sparse_count_pca as scp

result = scp.correspondence_analysis_matrix(counts, n_comps=2)

result.row_principal_coordinates     # observations by components
result.column_principal_coordinates  # variables by components
result.principal_inertias            # squared singular values, one per axis
result.inertia_ratio                 # each axis's share of total inertia
```

The principal inertias are the squared singular values, also called the
eigenvalues of the decomposition, so they are amounts of inertia rather than
proportions; `inertia_ratio` holds the proportions, dividing each principal
inertia by `total_inertia`. Only the requested `n_comps` axes are returned, so
those ratios sum to at most one, reaching one only when the returned axes
exhaust the table's inertia.

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
depth-scaled negative-binomial variance:

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
have the classical chi-square interpretation. `scaled_nb` is a
package-specific model name; its mean and variance are defined in the
[residual-PCA guide](residual-pca.md#choose-the-count-model).

## Complete example

--8<-- "examples/correspondence_analysis.md"
