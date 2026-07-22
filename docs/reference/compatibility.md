# Compatibility and related methods

This page separates controlled equalities from broad similarities. Unless a
condition is stated explicitly, related methods should not be described as
interchangeable.

## Scanpy PCA conventions

The AnnData PCA functions follow current Scanpy conventions for selecting `.X`,
a layer, or `.raw.X`; resolving a default highly-variable mask; honoring
`copy`; and writing scores, component vectors, and metadata.

There are intentional differences:

- residual and clipping parameters are specific to this package;
- masked variables receive `NaN` component values rather than zero;
- normalization is fitted before the PCA-only variable mask;
- backed sparse count arrays are currently loaded into memory for fitting;
- only ARPACK is supported.

These outputs are nevertheless placed at the default Scanpy keys so scores can
feed downstream neighbors and embedding workflows.

## SCTransform v2

The package's scaled-NB residual transform is related to, but not generally
identical to, default SCTransform v2. A controlled equality oracle aligns when:

- every cell has equal depth, so SCTransform's fixed log-library-size offset
  matches the exposure-scaled mean structure used here;
- SCTransform's final intercept implies the same empirical per-gene mean;
- this package receives `alpha = 1 / theta`, with infinite `theta` mapped to
  `alpha = 0`;
- SCTransform uses `min_variance=0` rather than its default variance floor;
- clipping bounds and clipping-before-centering order agree.

Default SCTransform v2 regularizes fitted intercepts and enables a variance
floor, either of which can change residuals. The equality conditions therefore
describe a test oracle, not default end-to-end equivalence.

The repository includes a pinned
[real-data equality oracle](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/real_data_reference)
generated with `sctransform` 0.4.3 and smaller algebraic identity tests.

## Shifted CLR, PFlog, and cleartools

The fixed-count shifted-CLR transform is

```text
clr(x + count_shift).
```

The PFlog normalization proposed in
[Booeshaghi et al., preprint version 4 (June 22,
2026)](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4) is obtained
with `count_shift = 1 / (4 * alpha)`. The follow-up
[`cleartools/scclr`](https://github.com/cleartools/scclr) project describes the
same PFlog formula and represents it as sparse values plus a per-cell mean
before implicit PCA. It provides Rust-backed Python tooling; related
`cleartools` projects provide other language interfaces.

This package tests transformed values against a separately maintained dense
PFlog oracle. That formula-level parity does not imply identical defaults,
metadata, dtype behavior, clipping behavior, or AnnData ownership semantics
across packages.

`ProportionShiftedCLR` is a different historical formula with a fixed shift
after library-size division. It should not be labeled as the PFlog formulation
from Booeshaghi et al. preprint v4.

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

With `model="scaled_nb"`, the API is an experimental residual ordination. Its
inertia does not have the classical chi-square or barycentric interpretation.
The mode emits a warning and records its experimental status in result
metadata. `scaled_nb` is a package-specific model name, not a claim of parity
with an external named method.
