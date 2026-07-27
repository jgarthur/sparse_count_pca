# Comparing transforms

This page compares what the package's transforms represent and highlights
differences that affect interpretation. It is not a general method-selection
guide: choosing a normalization requires scientific and empirical judgments
that extend beyond this package.

For broader comparisons, see
[Ahlmann-Eltze and Huber (2023), *Comparison of transformations for single-cell
RNA-seq data*](https://doi.org/10.1038/s41592-023-01814-1), and
[Booeshaghi et al., *Normalization for sampled count data*](https://doi.org/10.1101/2022.05.06.490859),
which develops and benchmarks the shifted-CLR/PFlog approach. These papers
have different scopes and should be read as methodological context, not as a
single decision rule.

For complete formulas, API mappings, provenance, and validation status, see
the [transform catalogue](transforms.md).

| Transform | What it represents | Important distinction |
| --- | --- | --- |
| Residual PCA | PCA of deviations from a count-model expectation based on cell depth and gene abundance. | The model may be Poisson, binomial, or scaled-NB, with Pearson or deviance residuals. |
| Count-scale shifted log | PCA of `log1p(x / count_shift)` on the raw-count scale. | This does **not** perform library-size normalization before taking logs. |
| Count-scale shifted CLR | Within-cell log-ratio coordinates after adding the same raw-count shift to every gene. | PFlog in [Booeshaghi et al.](https://doi.org/10.1101/2022.05.06.490859) is obtained with `count_shift = 1 / (4 * alpha)`. |
| Composition-scale shifted CLR | CLR coordinates with a fixed shift after dividing by each cell total. | Its effective raw-count shift varies by cell depth. |
| Correspondence analysis | Classical row and column coordinates for a contingency table. | It does not apply ordinary PCA column centering; a mask defines new table margins. |

## Minimal residual-PCA example

The getting-started material uses Poisson Pearson residual PCA as a compact
example of the primary AnnData interface:

```python
scp.residual_pca(
    adata,
    layer="counts",
    n_comps=50,
    model="poisson",
    residual="pearson",
)
```

This example demonstrates the API; it is not a recommendation that Poisson
Pearson residuals are optimal for a particular dataset.

Use scaled negative-binomial residuals only when you have defensible
nonnegative per-gene overdispersion values. The package does not estimate those
values for you. `scaled_nb` is a package-specific label; see the
[residual-PCA guide](guides/residual-pca.md#choose-the-count-model) for its
formula.

## Questions that separate the methods

### Do you want model residuals or log-ratio coordinates?

Residual transforms compare observed counts with an explicit expected count
model. CLR transforms instead compare log abundances within each observation.
They answer different questions even when their PCA plots look similar.

### Is the shift on the count scale or composition scale?

`ShiftedCLR` adds one fixed raw-count shift to every entry.
`ProportionShiftedCLR` first divides each row by its total and then adds a fixed
composition shift. These are not alternative parameterizations of one formula.

### Do selected variables define normalization or only PCA?

Residual, log, and CLR transforms fit their normalization state on the full
selected count matrix before `mask_var` chooses PCA variables.
Correspondence analysis is different: its variable mask creates a new
contingency table and recomputes the margins.

See [normalization, masking, and centering](concepts/normalization-masking-and-centering.md)
for the consequences.

### Do you need transformed values?

Use a one-step PCA function when you only need scores, components, and variance
statistics. Use [`transform`](guides/transform-reuse.md) when you need to inspect
bounded slices, materialize selected values, or run PCA repeatedly with
different masks.
