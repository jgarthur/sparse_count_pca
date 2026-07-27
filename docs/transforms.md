# Transform catalogue

This page defines the transformations currently exposed by
`sparse-count-pca`, shows how each one is called, and records its scientific
lineage and validation. For a shorter comparison of their assumptions, see
[Comparing transforms](choosing-a-transform.md). For exact input validation,
masking, dtype, and output contracts, use the [API reference](reference/api/anndata.md)
and [normative specification](development/specification.md).

The status labels below describe maturity within the current package. The
project is pre-1.0, so **supported** does not yet promise a frozen API.

## At a glance

| Transform | AnnData function | Matrix function | Two-step specification | Status |
| --- | --- | --- | --- | --- |
| Pearson or deviance residuals | `residual_pca` | `residual_pca_matrix` | `Residual` | Supported; `scaled_nb` is package-specific |
| Fixed-count shifted log | `shifted_log_pca` | `shifted_log_pca_matrix` | `ShiftedLog` | Supported |
| Fixed-count shifted CLR | `shifted_clr_pca` | `shifted_clr_pca_matrix` | `ShiftedCLR` | Supported; includes PFlog v4 |
| Proportion-shifted CLR | `proportion_shifted_clr_pca` | `proportion_shifted_clr_pca_matrix` | `ProportionShiftedCLR` | Supported for historical reproducibility |
| Dirichlet log | `dirichlet_log_pca` | `dirichlet_log_pca_matrix` | `DirichletLog` | Experimental |
| Dirichlet CLR | `dirichlet_clr_pca` | `dirichlet_clr_pca_matrix` | `DirichletCLR` | Experimental |
| Correspondence analysis | `correspondence_analysis` | `correspondence_analysis_matrix` | — | Supported; `scaled_nb` mode is experimental |

All PCA functions use observations in rows and variables in columns. The
two-step specifications are passed to `transform` when transformed values must
be inspected or one fitted transformation must be reused.

## Notation

Let \(X_{ij}\) be the count for observation \(i\) and variable \(j\). Define

```math
n_i = \sum_j X_{ij},
\qquad
N = \sum_i n_i,
\qquad
p_j = \frac{\sum_i X_{ij}}{N},
\qquad
\mu_{ij} = n_i p_j.
```

For log-ratio transforms, \(G\) is the number of variables and

```math
\operatorname{clr}(y)_j
= \log y_j - \frac{1}{G}\sum_k \log y_k.
```

## Pearson and deviance residuals

### Formula

Pearson residuals are

```math
R_{ij} = \frac{X_{ij}-\mu_{ij}}{\sqrt{V_{ij}}}.
```

The supported variance models are:

| `model` | Variance \(V_{ij}\) |
| --- | --- |
| `"poisson"` | \(\mu_{ij}\) |
| `"binomial"` | \(\mu_{ij}(1-p_j)\) |
| `"scaled_nb"` | \(\mu_{ij}(1+\alpha_j\bar n p_j)\), where \(\bar n\) is mean observation depth |

For deviance residuals,

```math
R_{ij}
= \operatorname{sign}(X_{ij}-\mu_{ij})
  \sqrt{d(X_{ij},\mu_{ij})},
```

where \(d\) is the elementwise model deviance. In particular,

```math
d_{\mathrm{Pois}}(x,\mu)
= 2\left[x\log\frac{x}{\mu}-(x-\mu)\right]
```

and

```math
d_{\mathrm{Bin}}(x,n,\mu)
= 2\left[
  x\log\frac{x}{\mu}
  +(n-x)\log\frac{n-x}{n-\mu}
  \right].
```

For `scaled_nb`, let

```math
\widetilde\alpha_{ij}
= \frac{\alpha_j}{n_i/\bar n}.
```

Then

```math
d_{\mathrm{sNB}}(x,\mu,\widetilde\alpha)
= 2\left[
  x\log\frac{x}{\mu}
  - \frac{1+\widetilde\alpha x}{\widetilde\alpha}
    \log\frac{1+\widetilde\alpha x}{1+\widetilde\alpha\mu}
  \right],
```

with the Poisson limit used when \(\alpha_j\) is numerically zero.

### Interfaces and parameters

Use `residual_pca`, `residual_pca_matrix`, or
`Residual(model=..., residual=...)`. The transform-defining parameters are:

- `model`: `"poisson"`, `"binomial"`, or `"scaled_nb"`;
- `residual`: `"pearson"` or `"deviance"`;
- `alpha`: nonnegative scalar or per-variable values required by `scaled_nb`;
- optional `clip`, `clip_mode`, and `clip_max_nnz_ratio`.

The package does not estimate `alpha`.

### Origin and validation

Residual PCA under the multinomial count model follows the work of
[Townes et al. (2019)](https://doi.org/10.1186/s13059-019-1861-6), including
the use of Pearson or deviance residuals as a fast approximation to GLM-PCA.
[Lause, Berens, and Kobak (2021)](https://doi.org/10.1186/s13059-021-02451-7)
develop the analytic Pearson-residual formulation for single-cell UMI counts.
Negative-binomial Pearson-residual normalization is also related to
[Hafemeister and Satija (2019)](https://doi.org/10.1186/s13059-019-1874-1),
but this package's `scaled_nb` model is its own exposure-scaled
parameterization, not an implementation of SCTransform.

Tests compare the matrix-free result with independent dense oracles for every
model/residual combination and with pinned external values from the
[Townes `null_residuals()` implementation](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/townes_reference).
A separate
[controlled SCTransform equality oracle](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/real_data_reference)
tests only the conditions under which the two parameterizations coincide.

### Caveats

Residual normalization is fitted before `mask_var` selects PCA variables.
Clipping is optional, occurs before PCA centering, and can increase sparse
support. See the [residual-PCA guide](guides/residual-pca.md),
[masking concept page](concepts/normalization-masking-and-centering.md), and
[compatibility reference](reference/compatibility.md#sctransform-v2).

## Fixed-count shifted log

### Formula

For a positive raw-count shift \(a\),

```math
Z_{ij}=\log\left(1+\frac{X_{ij}}{a}\right).
```

This is `log(X + a)` with the constant `log(a)` removed. Ordinary PCA column
centering makes those two gauges equivalent for PCA.

### Interfaces and parameters

Use `shifted_log_pca`, `shifted_log_pca_matrix`, or
`ShiftedLog(count_shift=a)`. `count_shift` is required and has no implicit
default.

### Origin and validation

This is the package's explicit raw-count parameterization of an elementary
shifted-log transform; no unique upstream method is claimed. Tests compare all
operator products, materialized values, and PCA outputs with a direct dense
implementation.

### Caveats

This transform does **not** divide by observation totals and is not the usual
Scanpy `normalize_total` followed by `log1p` workflow. See
[Shifted log and CLR](guides/shifted-log-and-clr.md#fixed-count-shifted-log).

## Fixed-count shifted CLR

### Formula

For a positive raw-count shift \(a\),

```math
Z_{ij}
= \log(X_{ij}+a)
  - \frac{1}{G}\sum_k\log(X_{ik}+a)
= \operatorname{clr}(X_i+a\mathbf 1)_j.
```

The PFlog normalization in Booeshaghi et al. preprint v4 is obtained with
\(a=1/(4\alpha)\), or equivalently by row-centering
`log1p(4 * alpha * X)`.

### Interfaces and parameters

Use `shifted_clr_pca`, `shifted_clr_pca_matrix`, or
`ShiftedCLR(count_shift=a)`. `count_shift` is required and positive.

### Origin and validation

The centered log-ratio transformation comes from compositional data analysis;
see [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x).
The current PFlog parameterization is documented by
[Booeshaghi et al., preprint version 4 (June 22, 2026)](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4)
and the follow-up [`cleartools/scclr`](https://github.com/cleartools/scclr)
implementation.

Tests compare transformed values with an independently written dense
[count-shifted PFlog oracle](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/shifted_clr_reference).

### Caveats

The shift is fixed on the raw-count scale. It is not interchangeable with a
shift added after library-size division, and formula parity with another
package does not imply identical defaults or outputs. See
[PFlog and cleartools compatibility](reference/compatibility.md#shifted-clr-pflog-and-cleartools).

## Proportion-shifted CLR

### Formula

For a positive composition-scale shift \(\tau\),

```math
Z_i
= \operatorname{clr}\left(\frac{X_i}{n_i}+\tau\mathbf 1\right)
= \operatorname{clr}(X_i+n_i\tau\mathbf 1).
```

Its effective raw-count shift is therefore different for every observation.

### Interfaces and parameters

Use `proportion_shifted_clr_pca`, `proportion_shifted_clr_pca_matrix`, or
`ProportionShiftedCLR(composition_shift=tau)`. `composition_shift` is required
and positive.

### Origin and validation

This historical PFlog formula is retained for reproducibility with an earlier
manuscript revision. Tests use an independent dense oracle tied to the pinned
[upstream revision](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/proportion_shifted_clr_reference).

### Caveats

This transform is invariant to deterministic rescaling of each observation,
unlike fixed-count shifted CLR. It should not be described as the current
PFlog v4 formula.

## Dirichlet log and Dirichlet CLR

### Formula

Let \(A>0\) be the total prior concentration, let \(q_j>0\) be prior
proportions with \(\sum_j q_j=1\), and set prior counts \(a_j=Aq_j\). The
posterior-mean composition is

```math
\widehat\pi_{ij}=\frac{X_{ij}+a_j}{n_i+A}.
```

Dirichlet log analyzes

```math
Z_{ij}=\log\widehat\pi_{ij},
```

while Dirichlet CLR analyzes

```math
Z_i=\operatorname{clr}(X_i+a).
```

Fixed-count shifted CLR is the uniform-prior special case with
\(A=G\,\mathtt{count\_shift}\).

### Interfaces and parameters

Use `dirichlet_log_pca`, `dirichlet_log_pca_matrix`, or `DirichletLog`; and
`dirichlet_clr_pca`, `dirichlet_clr_pca_matrix`, or `DirichletCLR`.
`concentration` defaults to `1.0`; `prior_proportions=None` uses a uniform
prior. AnnData interfaces also accept an `adata.var` key.

### Origin and validation

These are package-defined prior-count generalizations. The CLR coordinate
system follows [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x),
but no parity with a named external Dirichlet-normalization package is claimed.
Tests compare transformed values, operator products, and PCA outputs with
independent dense calculations.

### Caveats

Both APIs are experimental. The prior is fitted on the full variable universe
before `mask_var` selects PCA columns. Different concentrations and prior
compositions encode scientific assumptions, not merely numerical settings.

## Correspondence analysis

### Formula

Classical correspondence analysis decomposes the standardized independence
residual matrix

```math
Z_{ij}
= \frac{1}{\sqrt N}\frac{X_{ij}-\mu_{ij}}{\sqrt{\mu_{ij}}}.
```

It does not apply ordinary PCA column centering. Row and column coordinates are
then scaled by their corresponding masses.

The experimental `scaled_nb` mode replaces the Poisson variance in this
standardization with

```math
V_{ij}=\mu_{ij}(1+\alpha_j\bar n p_j).
```

### Interfaces and parameters

Use `correspondence_analysis` or `correspondence_analysis_matrix` with
`model="poisson"` for classical CA. There is no two-step transform
specification because CA has its own coordinates, inertia, and mass-weighting
contract. `model="scaled_nb"` additionally requires nonnegative `alpha`.

### Origin and validation

Correspondence analysis is a classical contingency-table method. Its use for
single-cell count matrices and its relationship to PCA pipelines are described
by [Hsu and Culhane (2023)](https://doi.org/10.1038/s41598-022-26434-1).
Package outputs are checked against the pinned Bioconductor
[`corral` reference fixture](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/corral_reference),
including singular values and row and column coordinates.

### Caveats

A CA variable mask defines a new contingency table, so margins and masses are
recomputed after masking. This deliberately differs from the PCA transform
APIs. The `scaled_nb` extension is a package-specific residual ordination: its
inertia has no classical chi-square or barycentric interpretation. See the
[correspondence-analysis guide](guides/correspondence-analysis.md) and
[compatibility reference](reference/compatibility.md#classical-correspondence-analysis).

## What validation means here

Independent dense oracles establish that the sparse-plus-low-rank operators
represent the documented formulas and that their PCA or CA results agree with
dense calculations on test matrices. Pinned external fixtures test selected
relationships to prior implementations. Neither kind of test makes methods
scientifically interchangeable outside the stated conditions. The
[testing guide](development/testing.md) describes the oracle hierarchy and
tolerances.
