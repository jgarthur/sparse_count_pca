# Choosing a transform

No transform is a universal default for every scientific question. Choose the
model whose normalization assumptions and coordinates match the analysis you
intend to interpret.

For complete formulas, API mappings, provenance, and validation status, see
the [transform catalogue](transforms.md).

| Transform | Use it when | Important distinction |
| --- | --- | --- |
| Residual PCA | You want PCA after removing a count-model expectation based on cell depth and gene abundance. | Choose a Poisson, binomial, or scaled-NB model and Pearson or deviance residuals. |
| Fixed-count shifted log | You want PCA of `log1p(x / count_shift)` on the raw-count scale. | This does **not** perform library-size normalization before taking logs. |
| Fixed-count shifted CLR | You want within-cell log-ratio coordinates after adding the same raw-count shift to every gene. | [PFlog in Booeshaghi et al. v4](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4) is obtained with `count_shift = 1 / (4 * alpha)`. |
| Proportion-shifted CLR | You need the historical formula with a fixed shift after dividing by each cell total. | Its effective raw-count shift varies by cell depth. |
| Correspondence analysis | You want classical row and column coordinates for a contingency table. | It does not apply ordinary PCA column centering; a mask defines new table margins. |

## A practical starting point

For sparse single-cell counts where the scientific aim is to remove an
independence-model expectation before PCA, begin with Poisson Pearson residual
PCA:

```python
scp.residual_pca(
    adata,
    layer="counts",
    n_comps=50,
    model="poisson",
    residual="pearson",
)
```

That recommendation is a workflow starting point, not a claim that Poisson
Pearson residuals are optimal for every dataset.

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
