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
| Size-factor-normalized log1p | `log1p_norm_pca` | `log1p_norm_pca_matrix` | `Log1pNormalized` | Supported; matches Scanpy's `normalize_total` + `log1p` |
| Count-scale shifted CLR | `shifted_clr_pca` | `shifted_clr_pca_matrix` | `ShiftedCLR` | Supported; includes the PFlog parameterization |
| Composition-scale shifted CLR | `proportion_shifted_clr_pca` | `proportion_shifted_clr_pca_matrix` | `ProportionShiftedCLR` | Supported |
| Dirichlet log | `dirichlet_log_pca` | `dirichlet_log_pca_matrix` | `DirichletLog` | Numerically verified only |
| Dirichlet CLR | `dirichlet_clr_pca` | `dirichlet_clr_pca_matrix` | `DirichletCLR` | Numerically verified only |
| Correspondence analysis | `correspondence_analysis` | `correspondence_analysis_matrix` | — | Supported; `scaled_nb` mode is numerically verified only |

All PCA functions use observations in rows and variables in columns. A two-step
specification is passed as the `method` argument of `transform`, which returns
a `TransformedMatrix` for inspecting transformed values or running PCA more
than once from one fitted normalization.

Status describes scientific validation, not API stability. Every transform is
verified against an independently written dense implementation of its formula,
so the label says whether anything beyond that is available:

- **Supported**: also pinned against a named prior implementation, cited in the
  section below.
- **Numerically verified only**: implemented as specified, but with no external
  implementation pinned — either none exists, or the method extends past the
  available reference. Numeric agreement with a formula is not scientific
  corroboration of the method.

Pinned fixtures test selected relationships to prior implementations; they do
not make methods scientifically interchangeable outside the stated conditions.
The [testing guide](development/testing.md) describes the oracle hierarchy and
tolerances.

## What this package does not do

The table above is the complete set of transforms.

Size-factor-normalized log1p is a boundary case for the package's
representation. Dividing by a size factor and taking `log1p` maps zero to
zero, so the normalized matrix stays sparse and only PCA centering makes it
dense — a rank-one correction that iterative sparse PCA implementations
already handle. It is included anyway, as most common count normalization.

Two log-normalization recipes are deliberately absent. Seurat's "CLR"
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
- `alpha`: nonnegative scalar or per-variable overdispersion required by
  `scaled_nb`; variance under the model increases with `alpha`. Not currently
  estimated in this package
- `clip`, `clip_mode`, and `clip_max_nnz_ratio`. `clip` defaults to the named
  `"seurat"` threshold \(\sqrt{n_\mathrm{obs}/30}\); `"scanpy"` names
  \(\sqrt{n_\mathrm{obs}}\), a float sets the threshold directly, and `None`
  disables clipping. See
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
Clipping is on by default, occurs before PCA centering, and can increase sparse
support. See the [residual-PCA guide](guides/residual-pca.md),
[masking concept page](concepts/normalization-masking-and-centering.md), and
[compatibility reference](reference/compatibility.md#sctransform-v2).

## Size-factor-normalized log1p

### Formula

For positive per-observation size factors \(s_i\),

```math
Z_{ij}
= \log\!\left(1+\frac{X_{ij}}{s_i}\right)
= \log(X_{ij}+s_i) - \log s_i.
```

With a target total \(t\), the size factors are \(s_i = n_i/t\). The default
target is the median observation total, so the size factors have median one.

Zeros map to zero, so this is the one transform in the catalog whose
normalized matrix is sparse before PCA centering; its representation has rank
zero. There is no row centering and no log-ratio coordinate rule.

### Interfaces and parameters

Use `log1p_norm_pca`, `log1p_norm_pca_matrix`, or
`Log1pNormalized(target_sum=..., size_factors=...)`. Exactly one
normalization recipe applies:

- `target_sum`: positive total to which each observation is normalized;
  `target_sum=None`, the default, uses the median observation total;
- `size_factors`: positive per-observation divisors, or an `adata.obs` key on
  the AnnData interface, used without rescaling. Mutually exclusive with an
  explicit `target_sum`.

### Origin and validation

This is the standard library-size normalization followed by `log1p`, as
implemented by Scanpy's `pp.normalize_total` and `pp.log1p`.
Transformed values are
[tested directly against those Scanpy functions](https://github.com/jgarthur/sparse_count_pca/blob/main/tests/test_log1p_scanpy.py)
for both the median and explicit `target_sum` recipes, and against an
independent dense formula oracle.

### Caveats

Cell totals and the median target are computed before `mask_var` selects PCA
variables; slice the count matrix first if excluded genes should not count
toward totals.

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
`ShiftedCLR(count_shift=a)`. `count_shift` is required and positive. It is
added to the raw counts, the same amount for every observation and variable, so
larger values shrink the resulting log ratios toward zero.

### Origin and validation

The centered log-ratio transformation comes from compositional data analysis;
see [Aitchison (1982)](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x).
The PFlog parameterization is documented by Booeshaghi et al. and the follow-up
[`cleartools/scclr`](https://github.com/cleartools/scclr) implementation, which
is built on the [`runorm`](https://github.com/cleartools/runorm) crate. Its
`alpha` is a dataset-wide overdispersion under a negative-binomial size-factor
model, not the per-gene `alpha` of the residual transforms above; see
[PFlog and cleartools compatibility](reference/compatibility.md#shifted-clr-pflog-and-cleartools).

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
and positive. It is added to each observation's proportions rather than to its
counts, so its equivalent raw-count shift \(n_i\tau\) differs per observation.

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
`concentration` is the total prior concentration \(A\), measured in counts: it
is the number of prior pseudo-counts spread over the variables, so larger values
shrink each observation harder toward the prior composition. It defaults to
`1.0`. `prior_proportions` is the composition \(q\) those pseudo-counts are
distributed across; its values are proportions rather than counts, must be
strictly positive, and must sum to one. `prior_proportions=None` uses a uniform
prior. AnnData interfaces also accept an `adata.var` key.

### Caveats

These are prior-count generalizations with no external reference implementation
to pin. The prior is fitted on the full variable universe before `mask_var`
selects PCA columns.

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

It applies one depth-dispersion relationship across the whole matrix, so it
does not represent batch structure. Fitting per batch is not a small adjustment
to that: the mean depth and every gene proportion would become batch-specific,
so the null model itself would differ by batch and the residuals would no
longer be on a common scale. This package does not fit per batch.

Both row and column outputs are principal coordinates: each singular vector is
divided by the square root of its row or column mass, giving the standard
coordinates, and then scaled by the singular values. Standard coordinates are
not stored; divide the principal coordinates by the singular values to recover
them.

### Interfaces and parameters

Use `correspondence_analysis` or `correspondence_analysis_matrix` with
`model="poisson"` for classical CA. There is no two-step transform
specification because CA has its own coordinates, inertia, and mass-weighting
contract. `model="scaled_nb"` additionally requires the nonnegative
overdispersion parameter `alpha`.

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
