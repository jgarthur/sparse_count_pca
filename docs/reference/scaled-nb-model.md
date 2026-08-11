# The scaled-NB null model

`model="scaled_nb"` fits a negative-binomial null whose overdispersion scales
inversely with observation depth. This page defines that model, shows that its
maximum-likelihood null mean is exactly the Poisson one, and relates it to
prior work.

To call it, see the
[transform catalog](../transforms.md#pearson-and-deviance-residuals) and the
[residual-PCA guide](../guides/residual-pca.md#choose-the-count-model).

## Notation

Let \(X_{ij}\) be the count for observation \(i\) and variable \(j\), with
\(m\) observations and \(G\) variables. Define

```math
n_i = \sum_j X_{ij},
\qquad
N = \sum_i n_i,
\qquad
p_j = \frac{\sum_i X_{ij}}{N},
\qquad
\mu_{ij} = n_i p_j,
```

and the relative depth and mean depth

```math
s_i = \frac{n_i}{\bar n},
\qquad
\bar n = \frac{N}{m},
\qquad
S = \sum_i s_i = m.
```

## The model

For a supplied per-variable overdispersion \(\alpha_j \ge 0\), the
overdispersion of \(X_{ij}\) is

```math
\widetilde\alpha_{ij} = \frac{\alpha_j}{s_i},
```

so a variable's overdispersion is scaled inversely with relative depth. Writing
variable \(j\)'s depth-normalized mean as \(\lambda_j\), the model for
\(\alpha_j > 0\) is

```math
X_{ij}\sim
\operatorname{NB}\left(
    r_{ij}=\frac{s_i}{\alpha_j},
    q_j=\frac{1}{1+\alpha_j\lambda_j}
\right),
```

where \(\operatorname{NB}(r,q)\) has probability mass proportional to
\(q^{r}(1-q)^{x}\), and \(\widetilde\alpha_{ij} = 1/r_{ij}\). Here \(r_{ij}\) is
the size of the negative binomial, so the supplied \(\alpha_j\) is a dispersion
rather than a size: an estimate reported on the size scale, such as an
SCTransform `theta`, must be inverted before it is passed as `alpha`. Then

```math
\operatorname{E}(X_{ij})=s_i\lambda_j,
\qquad
\operatorname{Var}(X_{ij})
=s_i\lambda_j(1+\alpha_j\lambda_j).
```

The package does not estimate \(\alpha_j\).

The model applies one depth-dispersion relationship across the whole matrix, so
it does not represent batch structure. Where batches differ substantially in
sequencing depth, the shared mean depth represents none of them well.

Fitting per batch is not a small adjustment to that. The mean depth and every
gene proportion become batch-specific, so the null model itself differs by
batch and the resulting residuals are no longer on a common scale. This package
does not fit per batch.

## The maximum-likelihood null mean

Conditional on the size factors and fixed overdispersion, `scaled_nb` has the
same maximum-likelihood mean as the Poisson model.

The derivation justifies using \(\mu_{ij}=n_i p_j\) as the fitted mean in the
[Pearson residual formula](../transforms.md#pearson-and-deviance-residuals) for
`scaled_nb`, rather than only for Poisson.

All observations share the same NB probability parameter \(q_j\) (distinct from
the variable proportion \(p_j\) above). Because independent negative-binomial
variables with common \(q_j\) are closed under addition,

```math
T_j=\sum_i X_{ij}
\sim
\operatorname{NB}\left(\frac{S}{\alpha_j},q_j\right).
```

For fixed \(\alpha_j\), the joint likelihood over \(i\) is a one-parameter
exponential family in the natural parameter \(\log(1-q_j)\), with sufficient
statistic \(T_j\):

```math
\prod_i\Pr(X_{ij}=x_{ij})
= h(x)
  \exp\left\{
    T_j\log(1-q_j)+\frac{S}{\alpha_j}\log q_j
  \right\},
```

where \(h\) collects the terms free of \(\lambda_j\). For \(T_j>0\) the
maximum-likelihood estimate therefore matches the sufficient statistic to its
expectation, and \(\operatorname{E}(T_j)=S\lambda_j\), so

```math
\widehat{\lambda}_j=\frac{T_j}{S}.
```

At \(T_j=0\) the log-likelihood is strictly decreasing in \(\lambda_j\), so the
maximum is the boundary estimate \(\widehat{\lambda}_j=0\), which the same
formula gives. A variable with zero total count contributes a zero residual
column in any case, so it carries no variance and no coefficient.

Since \(S=m\),

```math
\widehat{\lambda}_j
=\frac{\sum_i X_{ij}}{m}
=\bar n p_j,
```

and therefore

```math
\widehat{\mu}_{ij}
=s_i\widehat{\lambda}_j
=n_i p_j
=\mu_{ij}.
```

Poisson and `scaled_nb` thus share the same exact fitted null mean \(n_i p_j\),
but differ in variance, likelihood, deviance, and residuals. The result extends
to \(\alpha_j=0\) by the Poisson limit.

## Relationship to prior work

The inverse-size-factor dispersion form matches the scaling used by
[Yu, Huber, and Vitek (2013)](https://doi.org/10.1093/bioinformatics/btt143),
whose normalization lets the size factor affect the dispersion as well as the
expected value, so that mean and variance scale linearly with it. That paper
obtains its size factors by the DESeq median-of-ratios method rather than the
depth ratio \(s_i=n_i/\bar n\) used here.

This differs from the negative-binomial size-factor model considered by
[Lause, Berens, and Kobak (2021)](https://doi.org/10.1186/s13059-021-02451-7)
in the context of Pearson residuals, where the overdispersion does not scale
inversely with \(s_i\). Its NB probability parameter therefore varies across
observations, the closure argument above does not apply, and the Poisson fitted
mean is only approximate.

`scaled_nb` is not an implementation of SCTransform. For the conditions under
which the two agree, see the
[residual-PCA comparison](../guides/residual-pca.md#relationship-to-sctransform)
and the [compatibility reference](compatibility.md#sctransform-v2).
