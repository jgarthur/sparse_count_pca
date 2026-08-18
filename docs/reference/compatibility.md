# Compatibility and related methods

This page separates controlled equalities from broad similarities. Unless a
condition is stated explicitly, related methods should not be described as
interchangeable.

## Coming from Scanpy

Scanpy ships its own Pearson-residual preprocessing, so the first practical
question is which call this package replaces.

| If you currently call | Closest equivalent here | What differs |
| --- | --- | --- |
| `sc.experimental.pp.normalize_pearson_residuals_pca` | `scp.residual_pca(model="poisson")` | Scanpy uses a negative-binomial variance with one shared `theta`; `model="poisson"` is its `theta → ∞` limit. See below. |
| `sc.experimental.pp.normalize_pearson_residuals`, to inspect residual values | `scp.transform(adata, scp.Residual(...))`, then `materialize` | Scanpy materializes dense residuals into `.X` or the selected layer or `obsm` representation and stores only normalization settings in `.uns`; the two-step API keeps the transform implicit and returns bounded slices on request. |
| `sc.experimental.pp.highly_variable_genes(flavor="pearson_residuals")` | keep using it | This package consumes `adata.var["highly_variable"]` and does not select variables. |
| `sc.pp.normalize_total` + `sc.pp.log1p` + `sc.pp.pca` | `scp.log1p_norm_pca(target_sum=...)` | Transformed values match exactly for both the default median target and an explicit `target_sum`, and that parity is [tested against Scanpy](https://github.com/jgarthur/sparse_count_pca/blob/main/tests/test_log1p_scanpy.py). This package additionally accepts supplied size factors, keeps the transform implicit rather than writing normalized values into `.X`, rejects zero-total cells, and offers no `exclude_highly_expressed` or log `base`. |

Nothing in this package requires materializing the residual matrix, and it adds
deviance residuals and the binomial and scaled-NB variance models, none of
which Scanpy offers. Clipping defaults still differ: Scanpy clips residuals at
`sqrt(n_obs)` by default, while this package defaults to the lower
`clip="seurat"` threshold, `sqrt(n_obs / 30)`. Pass `clip="scanpy"` for
Scanpy's threshold or `clip=None` for no clipping.

### The shared-`theta` model is not offered

Scanpy computes residuals against

```text
Var = mu + mu**2 / theta,
```

with `theta` constant across genes and cells (default 100), following
[Lause, Berens, and Kobak (2021)](https://doi.org/10.1186/s13059-021-02451-7).
This package's `scaled_nb` instead scales overdispersion inversely with cell
depth, which in Scanpy's parameterization is
`1 / theta_ij = alpha_j * mean_total / total_i`.
Matching the two would require `alpha_j = total_i / (mean_total * theta)` for
every cell at once, which is possible only when every cell has the same total
count — the same condition as the
[SCTransform equality oracle](#sctransform-v2) below. Supplying
`alpha = 1 / theta` therefore reproduces Scanpy's residuals on equal-depth
counts. The residuals generally differ at unequal depths; they approach
agreement only when cell depths are nearly equal or the negative-binomial
contribution to the variance is negligible.

With `theta → ∞` the two models agree exactly: Scanpy's variance becomes `mu`,
which is `model="poisson"` here.

Both statements are tested directly against Scanpy in
[`tests/test_residual_scanpy.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/tests/test_residual_scanpy.py):
one case compares residuals, singular values, and component vectors at
`theta=np.inf`, and another compares the equal-depth finite-`theta` case with
and without clipping.

## Scanpy PCA conventions

Where the APIs overlap, the AnnData PCA functions follow current Scanpy
conventions for selecting `.X` or a layer, resolving a default highly-variable
mask, honoring `copy`, and writing scores, component vectors, and metadata.

There are intentional differences:

- residual and clipping parameters are specific to this package;
- masked variables receive `NaN` component values rather than zero;
- normalization is computed before the PCA-only variable mask;
- backed sparse count arrays are currently loaded into memory for fitting;
- only ARPACK is supported;
- this package defaults to `float64`, whereas Scanpy PCA defaults to `float32`.

These outputs are nevertheless placed at the default Scanpy keys so scores can
feed downstream neighbors and embedding workflows.

## SCTransform v2

The package's scaled-NB residual transform is related to SCTransform v2, but only
demonstrated to be equivalent under very specific conditions:

- the comparison is specifically between Pearson residuals
- this package receives `alpha = 1 / theta`, where infinite `theta` uses the Poisson limit `alpha = 0`
- every cell has equal depth, or all genes have `alpha = 0`
- SCTransform's final intercept implies the same empirical per-gene mean
- SCTransform uses `min_variance=0` rather than its default variance floor
- clipping bounds agree

The repository includes a pinned
[real-data equality oracle](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/real_data_reference)
generated with `sctransform` 0.4.3 and smaller algebraic identity tests.

The default `clip="seurat"` with `clip_mode="symmetric"` reproduces
SCTransform's default clip range of \(\pm\sqrt{n_\mathrm{obs}/30}\). That is a
match of one number, not of the method: the name carries no part of
SCTransform's variance model, regularization, or variance floor, and it applies
unchanged to deviance residuals and to the Poisson and binomial models.

## Shifted CLR, PFlog, and cleartools

The count-scale shifted-CLR transform is

```text
clr(x + count_shift).
```

The PFlog normalization proposed in
[Booeshaghi et al., preprint version 4 (June 22,
2026)](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4) is obtained
with `count_shift = 1 / (4 * alpha)`.

That `alpha` is a dataset-wide overdispersion under a negative-binomial
size-factor model, not the per-gene, depth-scaled `alpha` of
[the scaled-NB null model](scaled-nb-model.md). Because the shift is its
reciprocal, a more overdispersed dataset takes a smaller `count_shift`.

The follow-up [`cleartools/scclr`](https://github.com/cleartools/scclr) package
describes the same PFlog formula and represents it as sparse values plus a
per-cell mean before implicit PCA. It provides Rust-backed Python tooling built
on the [`runorm`](https://github.com/cleartools/runorm) normalization crate.

PFlog is the transform itself — proportional fitting, then `log1p`, then
per-cell centering — and `runorm` expresses the shift as a proportional-fitting
target rather than as a pseudocount.

This package tests transformed values against a separately maintained dense
PFlog oracle. That formula-level parity does not imply identical defaults,
metadata, dtype behavior, clipping behavior, or AnnData ownership semantics
across packages.

`ProportionShiftedCLR` is a different historical formula with a fixed shift
after library-size division, from v3 of the same preprint.

## scan-rs and Cell Ranger

[`10XGenomics/scan-rs`](https://github.com/10XGenomics/scan-rs) is a Rust
library of single-cell analysis algorithms used as a dependency of Cell
Ranger. Its
[`AdaptiveMat`](https://github.com/10XGenomics/scan-rs/blob/4468ba8a6434538b6eb91c29e141b6756f977e7d/sqz/src/mat.rs)
and
[`normalization`](https://github.com/10XGenomics/scan-rs/blob/4468ba8a6434538b6eb91c29e141b6756f977e7d/scan-rs/src/normalization.rs)
sources are relevant prior art for matrix representations and normalization.

`sparse-count-pca` is an independent pure-Python implementation built around
SciPy's `LinearOperator`. No output-parity or drop-in-compatibility claim with
Cell Ranger or `scan-rs` is currently made.

## Classical correspondence analysis

With `model="poisson"`, the correspondence-analysis API implements classical
CA: total inertia equals Pearson chi-squared divided by the grand total, and
the reported row and column coordinates are principal coordinates.

**Coordinate scaling differs from R's `ca` package.** `ca::ca` returns standard
coordinates in `rowcoord` and `colcoord`, while `row_principal_coordinates` and
`column_principal_coordinates` here are principal coordinates — standard
coordinates scaled by the singular values. Comparing the two directly will show
a per-axis scale difference that is not a disagreement about the analysis.
Divide this package's coordinates by `singular_values` to recover standard
coordinates. Bioconductor `corral`, which this package is
[tested against](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/corral_reference),
reports both, and both are checked.

Reported `principal_inertias` are the squared singular values, the amounts of
inertia conventionally also called eigenvalues; `inertia_ratio` holds their
proportions of `total_inertia`. Software that reports only percentages of
inertia is reporting the latter.

With `model="scaled_nb"` and at least one effective positive overdispersion, the
API is an experimental residual ordination whose inertia does not have the
classical chi-square or barycentric interpretation. If every supplied
`alpha < 1e-8`, the implementation uses the exact Poisson/classical-CA limit,
although the mode still emits a warning and records its experimental status in
result metadata. `scaled_nb` is a package-specific model name, not a claim of
parity with an external named method.
