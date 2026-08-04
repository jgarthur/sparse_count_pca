# Transform catalog

This page defines the transformations `sparse-count-pca` exposes, shows how
each one is called, and records citations and validation. It notes differences
that affect interpretation, but it is not a method-selection guide. For exact
input validation, masking, dtype, and output contracts, see the
[API reference](reference/api/anndata.md) and
[normative specification](development/specification.md).

## At a glance

| Transform | AnnData function | Matrix function | Two-step specification | Status |
| --- | --- | --- | --- | --- |
| Pearson or deviance residuals | `residual_pca` | `residual_pca_matrix` | `Residual` | Supported; see notes on `scaled_nb` |
| Count-scale shifted CLR | `shifted_clr_pca` | `shifted_clr_pca_matrix` | `ShiftedCLR` | Supported; includes the PFlog parameterization |
| Composition-scale shifted CLR | `proportion_shifted_clr_pca` | `proportion_shifted_clr_pca_matrix` | `ProportionShiftedCLR` | Supported |
| Dirichlet log | `dirichlet_log_pca` | `dirichlet_log_pca_matrix` | `DirichletLog` | Package-defined |
| Dirichlet CLR | `dirichlet_clr_pca` | `dirichlet_clr_pca_matrix` | `DirichletCLR` | Package-defined |
| Correspondence analysis | `correspondence_analysis` | `correspondence_analysis_matrix` | — | Supported; `scaled_nb` mode is experimental |

All PCA functions use observations in rows and variables in columns. A two-step
specification is passed as the `method` argument of `transform`, which returns
a `TransformedMatrix` for inspecting transformed values or running PCA more
than once from one fitted normalization.

Status labels describe scientific validation, not API stability:

- **Supported**: verified against an independent dense implementation and the
  external fixtures named below, where available.
- **Package-defined**: verified independently, but specific to this package's
  Dirichlet prior-count formulation.
- **Experimental**: verified independently, but extended beyond the available
  external reference, as with `scaled_nb` correspondence analysis.

External fixtures test selected relationships to prior implementations; they do
not make methods scientifically interchangeable outside the stated conditions.
The [testing guide](development/testing.md) describes the oracle hierarchy and
tolerances.

## What this package does not do

The table above is the complete set of transforms. In particular, there is no
library-size-normalized log transform and no plain shifted-log transform of raw
counts. A raw-count shifted log without a depth model is deliberately not
exposed because it invites comparisons driven by sequencing depth.

That workflow is deliberately out of scope rather than merely unimplemented.
Dividing by a row total and taking `log1p` maps zero to zero, so the normalized
matrix stays sparse and only PCA centering makes it dense. Existing sparse PCA
implementations already handle that rank-one correction, so the transform gains
nothing from this package's representation. The transforms here are the ones
whose normalized matrix is dense *before* centering.

Two further log-normalization recipes are deliberately absent. Seurat's "CLR"
is not a CLR coordinate transform and adds a data-dependent shift rule. The
Ahlmann-Eltze and Huber normalized Anscombe log divides by a size factor before
adding \(1/(4\alpha)\), which is a cell-specific raw-count shift
\(r_i/(4\alpha)\) rather than the count-scale shift used here.

For a mapping from the Scanpy functions you may be replacing, see
[coming from Scanpy](reference/compatibility.md#coming-from-scanpy).

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

These are plug-in quantities computed from the observed margins, not latent
parameters, so \(\mu_{ij}\) is a fitted null mean and \(p_j\) a fitted
proportion. Following the residual literature, this page drops the hats.

For log-ratio transforms, \(G\) is the number of variables and

```math
\operatorname{clr}(y)_j
= \log y_j - \frac{1}{G}\sum_k \log y_k.
```

Each transform below makes two independent choices: a **shift rule**, which
makes zeros positive, and a **coordinate rule**, which decides whether PCA sees
ordinary logs or log-ratio coordinates. A single literature name need not fix
both — PFlog named a composition-scale shift before it named a count-scale one
— which is why the interfaces here name the shift domain explicitly.

## Pearson and deviance residuals

### Formula

Pearson residuals are

```math
R_{ij} = \frac{X_{ij}-\mu_{ij}}{\sqrt{V_{ij}}}.
```

The supported null models are:

| `model` | Null model for \(X_{ij}\) | Variance \(V_{ij}\) |
| --- | --- | --- |
| `"poisson"` | \(\operatorname{Poisson}(\mu_{ij})\) | \(\mu_{ij}\) |
| `"binomial"` | \(\operatorname{Binomial}(n_i,p_j)\) | \(\mu_{ij}(1-p_j)\) |
| `"scaled_nb"` | negative binomial with mean \(\mu_{ij}\) and dispersion \(\widetilde\alpha_{ij}\) below | \(\mu_{ij}(1+\alpha_j\bar n p_j)\), where \(\bar n\) is mean observation depth |

All three models use the same null mean \(\mu_{ij}=n_i p_j\), which is the
exact maximum-likelihood mean under `scaled_nb` as well as Poisson.

In `scaled_nb`, \(\alpha_j\) is the supplied gene overdispersion: larger values
mean more overdispersion. The overdispersion for \(X_{ij}\) is

```math
\widetilde\alpha_{ij}
= \frac{\alpha_j}{s_i},
\qquad
s_i = \frac{n_i}{\bar n},
```

so a gene's overdispersion is scaled inversely with relative cell depth. This
is the model's definition, not an approximation to a constant-overdispersion
model. The package does not estimate \(\alpha_j\). See
[the scaled-NB null model](reference/scaled-nb-model.md) for the full
parameterization and the maximum-likelihood derivation.

For deviance residuals,

```math
R_{ij}
= \operatorname{sign}(X_{ij}-\mu_{ij})
  \sqrt{d(X_{ij},\mu_{ij})},
```

where \(d\) is the elementwise deviance for the selected count distribution.
The [normative specification](development/specification.md#deviance-residuals)
gives the model-specific formulas and numerical limits.

### Interfaces and parameters

Use `residual_pca`, `residual_pca_matrix`, or
`Residual(model=..., residual=...)`. The transform-defining parameters are:

- `model`: `"poisson"`, `"binomial"`, or `"scaled_nb"`;
- `residual`: `"pearson"` or `"deviance"`;
- `alpha`: nonnegative scalar or per-variable values required by `scaled_nb`; not currently estimated in this package
- optional `clip`, `clip_mode`, and `clip_max_nnz_ratio`; see
  [clipping and precision](concepts/clipping-and-precision.md).

### Origin and validation

Related work includes residual PCA under the multinomial count model in
[Townes et al. (2019)](https://doi.org/10.1186/s13059-019-1861-6), the
comparison of that model with an offset version of negative-binomial
regression in
[Lause, Berens, and Kobak (2021)](https://doi.org/10.1186/s13059-021-02451-7),
and negative-binomial Pearson-residual normalization in
[Hafemeister and Satija (2019)](https://doi.org/10.1186/s13059-019-1874-1).

The `binomial` model is the per-variable marginal of that multinomial, as used by Townes et al.:
variable \(j\) has \(n_i\) trials and success probability \(p_j\), giving variance \(\mu_{ij}(1-p_j)\).

This package's `scaled_nb` model is its own depth-scaled overdispersion
parameterization, sharing the inverse-size-factor dispersion scaling of sSeq;
see [the scaled-NB null model](reference/scaled-nb-model.md). It is not an
implementation of SCTransform, and matches its Pearson residuals only under the
controlled conditions given in the
[residual-PCA comparison](guides/residual-pca.md#relationship-to-sctransform).

Pinned external values from the
[Townes `null_residuals()` implementation](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/townes_reference)
cover the four Poisson and binomial combinations, Pearson and deviance for
each; `scaled_nb` has no counterpart there.
A separate
[controlled SCTransform equality oracle](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/real_data_reference)
tests only the conditions under which the two NB parameterizations coincide.

### Caveats

Residual normalization is fitted before `mask_var` selects PCA variables.
Clipping is optional, occurs before PCA centering, and can increase sparse
support. See the [residual-PCA guide](guides/residual-pca.md),
[masking concept page](concepts/normalization-masking-and-centering.md), and
[compatibility reference](reference/compatibility.md#sctransform-v2).

## Count-scale shifted CLR

### Formula

For a positive raw-count shift \(a\),

```math
Z_{ij}
= \log(X_{ij}+a)
  - \frac{1}{G}\sum_k\log(X_{ik}+a)
= \operatorname{clr}(X_i+a\mathbf 1)_j
```

The PFlog normalization in the June 22, 2026
[Booeshaghi et al. preprint](https://doi.org/10.1101/2022.05.06.490859) is
exactly this transform with \(a=1/(4\alpha)\), equivalently evaluated by
row-centering `log1p(4 * alpha * X)`.

### Interfaces and parameters

Use `shifted_clr_pca`, `shifted_clr_pca_matrix`, or
`ShiftedCLR(count_shift=a)`. `count_shift` is required and positive.

### Origin and validation

The centered log-ratio transformation comes from compositional data analysis;
see [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x).
The PFlog parameterization is documented by Booeshaghi et al. and the
follow-up [`cleartools/scclr`](https://github.com/cleartools/scclr)
implementation.

Its dense oracle follows the
[pinned upstream count-scale PFlog formula](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/shifted_clr_reference).

### Caveats

The shift is fixed on the raw-count scale. It is not interchangeable with a
shift added after library-size division, and formula parity with another
package does not imply identical defaults or outputs. See
[PFlog and cleartools compatibility](reference/compatibility.md#shifted-clr-pflog-and-cleartools).

## Composition-scale shifted CLR

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

This composition-scale formula matches the June 10, 2026 revision of the same
Booeshaghi et al. preprint; the current
revision instead uses the count-scale formula above. Its dense oracle is tied
to that
[pinned earlier upstream revision](https://github.com/jgarthur/sparse_count_pca/tree/main/tests/proportion_shifted_clr_reference).

### Caveats

This transform is invariant to deterministic rescaling of each observation,
unlike count-scale shifted CLR. It should not be described as the current
PFlog formula / count-scale shifted CLR, which is recommended by the original authors.

## Dirichlet log and Dirichlet CLR

### Formula

Let \(A>0\) be the total prior concentration, let \(q_j>0\) be prior
proportions with \(\sum_j q_j=1\), and set prior counts \(a_j=Aq_j\). Each
observation's composition has a conjugate Dirichlet prior, and its counts are
multinomial given that composition:

```math
\pi_i\sim\operatorname{Dirichlet}(a_1,\dots,a_G),
\qquad
X_i\mid\pi_i\sim\operatorname{Multinomial}(n_i,\pi_i).
```

The posterior is \(\operatorname{Dirichlet}(X_{i1}+a_1,\dots,X_{iG}+a_G)\), so
the posterior-mean composition is

```math
\widehat\pi_{ij}=\frac{X_{ij}+a_j}{n_i+A}.
```

Dirichlet log analyzes

```math
Z_{ij}=\log\widehat\pi_{ij},
```

while Dirichlet CLR analyzes the same posterior mean in log-ratio coordinates,

```math
Z_i=\operatorname{clr}(\widehat\pi_i)=\operatorname{clr}(X_i+a).
```

The two right-hand sides above agree because the posterior denominator
\(n_i+A\) does not depend on \(j\), so it contributes a constant to every log
that the CLR row mean subtracts away.

The two are not interchangeable, however: they carry the same
within-observation log ratios but give different PCA decompositions, which is
why both are offered.

Count-scale shifted CLR is the uniform-prior special case: with
\(q_j=1/G\) the prior counts are \(a_j=A/G\), so
\(A=G\cdot\mathtt{count\_shift}\) gives \(a_j=\mathtt{count\_shift}\) and
\(\operatorname{clr}(X_i+a)=\operatorname{clr}(X_i+\mathtt{count\_shift}\,\mathbf 1)\).

### Interfaces and parameters

Use `dirichlet_log_pca`, `dirichlet_log_pca_matrix`, or `DirichletLog`; and
`dirichlet_clr_pca`, `dirichlet_clr_pca_matrix`, or `DirichletCLR`.
`concentration` defaults to `1.0`; `prior_proportions=None` uses a uniform
prior. AnnData interfaces also accept an `adata.var` key.

### Caveats

These are package-defined prior-count generalizations, with no external
reference implementation to pin. The prior is fitted on the full variable
universe before `mask_var` selects PCA columns.

## Correspondence analysis

### Formula

Classical correspondence analysis applies a singular value decomposition to the
standardized independence residual matrix

```math
Z_{ij}
= \frac{1}{\sqrt N}\frac{X_{ij}-\mu_{ij}}{\sqrt{\mu_{ij}}}.
```

It does not apply ordinary PCA column centering. Row and column coordinates are
then scaled by their corresponding masses.

The standardization uses the Poisson variance \(V_{ij}=\mu_{ij}\), so
\(Z_{ij}\) is exactly the Poisson Pearson residual of the
[residual section above](#pearson-and-deviance-residuals) divided by
\(\sqrt N\). The two analyses differ in what they do with that matrix, not in
the matrix itself: CA omits column centering, and rescales the singular vectors
by the row and column masses.

The experimental `scaled_nb` mode substitutes the same depth-scaled
negative-binomial variance used there,

```math
V_{ij}=\mu_{ij}(1+\alpha_j\bar n p_j).
```

It fits one dispersion-depth relationship over the whole matrix. For multiple
batches or strongly different depth regimes, analyze batches separately or use
a model that represents those differences.

Both row and column outputs are principal coordinates: each singular vector is
divided by the square root of its row or column mass, giving the standard
coordinates, and then scaled by the singular values. Standard coordinates are
not stored; divide the principal coordinates by the singular values to recover
them.

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
including singular values, total inertia, and both principal and standard
coordinates for rows and columns.

### Caveats

A CA variable mask defines a new contingency table, so margins and masses are
recomputed after masking. This deliberately differs from the other transform
APIs. The `scaled_nb` extension is experimental. See the
[correspondence-analysis guide](guides/correspondence-analysis.md) and
[compatibility reference](reference/compatibility.md#classical-correspondence-analysis).



## Further reading

[Ahlmann-Eltze and Huber (2023), *Comparison of transformations for single-cell
RNA-seq data*](https://doi.org/10.1038/s41592-023-01814-1) benchmarks
transformations and comments on the rationale of different methods.

[Booeshaghi et al. (2026), *Normalization for sampled count
data*](https://doi.org/10.1101/2022.05.06.490859) proposes the
shifted-CLR/PFlog approach, with strong theoretical and empirical evidence in
its favor.

[Hsu and Culhane (2023), *Correspondence analysis for dimension reduction, batch
integration, and visualization of single-cell RNA-seq
data*](https://doi.org/10.1038/s41598-022-26434-1) discusses correspondence
analysis in the context of omics data.
