# Log-transform design

Status: implemented.

This document records the mathematical and API rationale for log and CLR PCA
in `sparse-count-pca`. Before this design was implemented, `shifted_clr` used a
composition-scale shift. The canonical implementation now uses a fixed
raw-count shift, while the historical formula is available explicitly as
`proportion_shifted_clr`.

## Implemented decisions

The repository should expose a small set of mathematically explicit PCA
transforms, not a public normalization-construction DSL.

1. Make `shifted_clr` mean a **fixed raw-count shift**:

   $$
   \operatorname{clr}(x_c + a\mathbf 1), \qquad a>0.
   $$

2. Rename the current implementation to `proportion_shifted_clr`, because it
   uses a **fixed composition-scale shift**:

   $$
   \operatorname{clr}\left(\frac{x_c}{s_c}+\tau\mathbf 1\right)
   =
   \operatorname{clr}(x_c+s_c\tau\mathbf 1).
   $$

3. Keep `dirichlet_log` and `dirichlet_clr`. They already provide the
   gene-specific prior-count generalization using total prior concentration
   and a prior composition.

4. Add `shifted_log` as the uncentered-coordinate partner of count-shifted
   CLR. For PCA, define its zero-baseline gauge as

   $$
   \log\left(1+\frac{x_{cg}}{a}\right).
   $$

   It has the same column-centered PCA as $\log(x_{cg}+a)$ and is exactly
   sparse.

5. Do not expose `PFlog` or `logPF` as canonical function names. Those names
   have referred to different formulas. Document the current paper recipe as a
   parameterization of `shifted_clr` or `shifted_log`.

6. Do not implement Seurat "CLR" or the Ahlmann--Eltze--Huber
   normalization-before-shift recipe.

7. Defer exact negative-binomial Anscombe PCA. It is straightforward to
   represent as a sparse matrix, but adding it now would force decisions about
   dispersion inputs and estimation without helping resolve the current
   shifted-CLR ambiguity.

The semantic correction was made before the first stable release so ambiguous
behavior did not become a compatibility contract.

## Why this is the right boundary for this package

The broad catalog separates two choices:

- how zeros are made positive, and
- which coordinates represent the resulting positive vector.

That distinction should guide the implementation and documentation, but it
does not require a public API such as `log_normalize(divisor=..., shift=...,
coordinates=...)`. The repository is an AnnData-first spectral-analysis tool,
not a general normalization framework. Its public functions should continue
to name complete, reproducible transforms whose sparse-plus-low-rank PCA can
be implemented exactly.

The useful conceptual rule is:

> A shift rule determines the positive abundance estimate. A coordinate rule
> determines whether PCA sees ordinary logs, log-closed proportions, or CLR
> coordinates. These are independent choices even when two recipes share a
> literature name.

Post-log row scaling, proportional fitting, gene standardization, clipping,
variable-gene selection, residualization, and PCA remain separate pipeline
stages. In particular, historical PF--log--PF should not be folded into a
shifted-log transform.

## The scale of the shift must be in the name and metadata

For a positive divisor $d_c$, suppose a method forms

$$
v_{cg}=\frac{x_{cg}}{d_c}+\delta_{cg}.
$$

Then its effective raw-count shift is

$$
a_{cg}=d_c\delta_{cg},
$$

because

$$
\log v_{cg}=\log(x_{cg}+a_{cg})-\log d_c.
$$

This is the source of the current naming problem. A constant shift after
library-size division is not a constant count pseudocount when cell depths
vary.

For this repository, use these terms consistently:

| Term | Formula | Effective raw-count shift |
| --- | --- | --- |
| count shift | $x+a$ | $a$ |
| composition shift | $x/s+\tau$ | $s\tau$ |
| Dirichlet prior counts | $x+\lambda\pi$ | $\lambda\pi_g$ |
| post-size-factor shift | $x/r+q$ | $rq$ |

Use `shift`, `count_shift`, `composition_shift`, or `prior_counts` in code and
metadata. Avoid bare `pseudocount` when its scale is not explicit.

## Proposed public transform family

Every transform continues to have an AnnData function and a matrix function.
The signatures below omit the common PCA, AnnData-selection, and output
arguments.

### 1. Count-shifted log PCA

```python
shifted_log_pca_matrix(X, *, count_shift, ...)
shifted_log_pca(adata, *, count_shift, ...)
```

Canonical transformed values:

$$
z_{cg}=\log\left(1+\frac{x_{cg}}{a}\right),
\qquad a=\texttt{count_shift}>0.
$$

This is a zero-at-zero gauge of $\log(x+a)$. The two differ only by the
constant $\log a$, which ordinary PCA column centering removes. Defining the
zero-baseline form as canonical keeps the represented uncentered matrix and
the reported column means unambiguous.

Special cases include:

- `count_shift=1`: raw `log1p` PCA;
- `count_shift=1/(4*alpha)`: the shifted-log approximation motivated by the
  NB Anscombe transform.

This method does not normalize library sizes. That is a statistical property,
not a sparse-representation limitation.

### 2. Count-shifted CLR PCA

```python
shifted_clr_pca_matrix(X, *, count_shift, ...)
shifted_clr_pca(adata, *, count_shift, ...)
```

Definition:

$$
z_{cg}
=
\log(x_{cg}+a)
-
\frac{1}{G}\sum_h\log(x_{ch}+a),
\qquad a=\texttt{count_shift}>0.
$$

Equivalently,

$$
z_{cg}
=
\log\left(1+\frac{x_{cg}}{a}\right)
-
\frac{1}{G}\sum_h\log\left(1+\frac{x_{ch}}{a}\right).
$$

This is the canonical scalar count-shift CLR. It supports zero-total cells,
because no division by observed cell depth occurs.

The current PFlog formula is obtained with

$$
a=\frac{1}{4\alpha}.
$$

The implementation may evaluate the sparse entries as
`log1p(4 * alpha * x)`. The missing additive constant cancels under CLR.

Do not add a `pflog_pca` alias. Documentation should show the formula above
and identify the paper/version to which it refers.

### 3. Proportion-shifted CLR PCA

```python
proportion_shifted_clr_pca_matrix(X, *, composition_shift, ...)
proportion_shifted_clr_pca(adata, *, composition_shift, ...)
```

Definition:

$$
z_c
=
\operatorname{clr}\left(\frac{x_c}{s_c}+\tau\mathbf 1\right)
=
\operatorname{clr}(x_c+s_c\tau\mathbf 1),
\qquad \tau=\texttt{composition_shift}>0.
$$

This is the formula currently implemented by `shifted_clr`. It has a fixed
composition regularizer and a cell-specific raw-count shift $s_c\tau$. It is
exactly invariant to deterministic rescaling $x_c\mapsto k_cx_c$, but it is
undefined for zero-total cells.

Keep it because it is a coherent transform, has a pinned historical oracle,
and reproduces the June 10 formulation. Its explicit name prevents it from
being mistaken for current count-shift PFlog.

### 4. Dirichlet log-closure PCA

Keep the existing API:

```python
dirichlet_log_pca_matrix(
    X,
    *,
    concentration=1.0,
    prior_proportions=None,
    ...,
)
dirichlet_log_pca(adata, ...)
```

For $a_g=\lambda\pi_g$ and $A=\sum_g a_g=\lambda$,

$$
z_{cg}
=
\log\frac{x_{cg}+a_g}{s_c+A}.
$$

This is the log of a Dirichlet posterior-mean composition. The existing
parameterization is good:

- `concentration` is total prior count $\lambda$;
- `prior_proportions` is the shrinkage target $\pi$;
- `None` means a uniform prior over the full normalization gene universe.

The name `dirichlet_log` is preferable to a generic `log_closed` name because
it states why the shift vector and denominator have those values.

### 5. Dirichlet CLR PCA

Keep the existing API:

```python
dirichlet_clr_pca_matrix(
    X,
    *,
    concentration=1.0,
    prior_proportions=None,
    ...,
)
dirichlet_clr_pca(adata, ...)
```

Definition:

$$
z_c=\operatorname{clr}(x_c+\lambda\pi).
$$

The posterior denominator disappears under CLR:

$$
\operatorname{clr}\left(
\frac{x_c+\lambda\pi}{s_c+\lambda}
\right)
=
\operatorname{clr}(x_c+\lambda\pi).
$$

This is the generalized gene-specific count-shift CLR. The scalar
`shifted_clr` API is exactly its uniform-prior special case:

$$
\boxed{
\texttt{count\_shift}=a
\quad\Longleftrightarrow\quad
\texttt{concentration}=Ga,
\ \texttt{prior\_proportions}=\text{uniform}
}
$$

That identity should be implemented through shared builder code and locked in
by a unit test.

## Required API migration

The current public call

```python
shifted_clr_pca(..., pseudocount=c)
```

means a composition shift. Reusing the same keyword while changing its scale
would silently change scientific results. Do not do that.

Instead:

- rename the old implementation and keyword to
  `proportion_shifted_clr_pca(..., composition_shift=c)`;
- make the corrected function use
  `shifted_clr_pca(..., count_shift=a)`;
- require the shift keyword explicitly for both methods;
- remove the ambiguous default and the bare `pseudocount` keyword;
- record `shift_domain` in result metadata.

Because this is a pre-1.0 correction, a clean break is preferable to a long
deprecation cycle. Most importantly, an old call with no shift argument or
with `pseudocount=` must fail rather than run a different formula silently.

Suggested transform metadata:

```python
{
    "transform": "shifted_clr",
    "shift_domain": "count",
    "count_shift": a,
    "normalization_n_vars": G,
    ...,
}
```

and

```python
{
    "transform": "proportion_shifted_clr",
    "shift_domain": "composition",
    "composition_shift": tau,
    "effective_count_shift": "cell_total * composition_shift",
    "normalization_n_vars": G,
    ...,
}
```

Dirichlet metadata should additionally identify the prior-count domain:

```python
{
    "transform": "dirichlet_clr",
    "shift_domain": "dirichlet_prior_counts",
    "concentration": concentration,
    "prior_proportions": prior,
    "normalization_n_vars": G,
    ...,
}
```

## Coordinate choices that should and should not be exposed

### Keep both log closure and CLR

For $v_c=x_c+a$, log closure and CLR differ by a cell-specific scalar:

$$
\log\frac{v_{cg}}{\sum_hv_{ch}}
-
\operatorname{clr}(v_c)_g
=
\frac{1}{G}\sum_h\log v_{ch}
-
\log\sum_hv_{ch}.
$$

They contain the same within-cell log-ratios, but ordinary PCA is not
invariant to adding a different multiple of the all-ones feature vector to
each cell. Therefore `dirichlet_log` and `dirichlet_clr` are genuinely
different PCA transforms and both should remain.

### Do not add ILR PCA

An ILR basis is not needed for this package. After CLR and ordinary column
centering, all rows lie in the feature-contrast subspace. Multiplication by a
complete orthonormal ILR basis is an isometry on that subspace, so it preserves
cell scores and nonzero singular values. It would add dense-basis machinery
without adding a new spectral analysis. Gene-space loadings from CLR are also
easier to interpret.

### Do not add a public combinator API

Do not expose `Divisor`, `AfterDivision`, `OnCountScale`, `Log`, `CLR`, or
similar public strategy objects now. Those abstractions are useful in the
documentation and perhaps internally, but they create a much larger API and
many combinations that the package would then need to define, validate, and
test.

## Exact sparse-plus-low-rank representations

Let

$$
S_{cg}=\log\left(1+\frac{x_{cg}}{a_g}\right).
$$

Since $S_{cg}=0$ wherever $x_{cg}=0$, $S$ has the same support as the count
matrix.

The proposed transforms fit the existing `SparseLowRankMatrix` exactly:

| Transform | Exact representation before PCA column centering | Low-rank factor rank |
| --- | --- | ---: |
| shifted log | $S$ | 0 |
| count-shifted CLR | $S-\operatorname{rowmean}(S)\mathbf 1^\top$ | 1 |
| proportion-shifted CLR | same form, with $a_c=s_c\tau$ | 1 |
| Dirichlet log | $S+\mathbf 1\log(a)^\top-\log(s+A)\mathbf 1^\top$ | at most 2 |
| Dirichlet CLR | $S-\operatorname{rowmean}(S)\mathbf 1^\top+\mathbf 1(\log a-\operatorname{mean}\log a)^\top$ | at most 2 |

The current generic representation and operator already support all of these:

$$
M=S+UV^\top.
$$

No additional matrix representation is needed. The `shifted_log` tests
exercise the rank-zero case directly.

The gene-offset term in a count-shifted log is absent because the canonical
transform is `log1p(x / count_shift)`. For Dirichlet log closure, the row
denominator is scientifically meaningful and cannot be discarded before PCA.

## Internal implementation

### Share the log-correction builders

The shared private helper accepts strictly positive per-gene prior counts and
returns

```python
S.data = np.log1p(X.data / prior_counts[X.indices])
```

with copied CSR indices and indptr. Count-shifted CLR and both Dirichlet
transforms use it.

The builder layout is:

```text
_log_transforms.py
    validate_positive_scalar(...)
    validate_dirichlet_prior(...)
    log1p_count_correction(...)
    build_shifted_log_representation(...)
    build_shifted_clr_representation(...)
    build_proportion_shifted_clr_representation(...)
    build_dirichlet_log_representation(...)
    build_dirichlet_clr_representation(...)
```

The implemented organization keeps builders in `_log_transforms.py` and public
entry points in `_log_pca.py` and `_dirichlet_pca.py`. The important point is
to have one implementation of `log1p(x / prior_count)` and one validation
path.

### Shared implicit-PCA path

All transform families delegate solver validation, operator construction,
zero-variance checking, SVD, and variance bookkeeping to:

```python
compute_pca_from_representation(
    representation,
    n_comps,
    *,
    transform_label,
    params,
    dtype,
    solver,
    random_state,
    tol,
    return_operator,
)
```

Transform-specific code is responsible for:

- canonicalizing and validating counts;
- validating transform parameters;
- constructing the representation over the full normalization universe;
- applying the feature mask only after construction;
- assembling transform metadata.

The common helper owns only spectral mechanics. This keeps model semantics out
of a generic enum and avoids copy-and-paste execution paths.

### Shared result type

Use `PCAResult` as the sole public PCA dataclass. Its `operator` annotation is
`SparseLowRankLinearOperator | None`. Do not retain a residual-specific alias;
all PCA families use the same result contract.

Separate result classes are unnecessary because every PCA family reports the
same decomposition and variance contract.

### Keep the normalization universe rule

For AnnData calls, the input matrix after `layer`/`raw` selection and any
upstream permanent gene filtering defines the normalization universe.
`mask_var` selects columns for PCA only after:

- CLR row means are computed over all normalization genes;
- Dirichlet prior proportions are validated and normalized over all genes;
- cell totals and posterior denominators are computed from all genes.

This is already the intended behavior and must remain so during refactoring.
Store `normalization_n_vars` in every affected result.

## Reference and test coverage

The current fixed-count formula is covered by
[`tests/shifted_clr_reference`](../../tests/shifted_clr_reference/). The
historical composition-shift formula is covered separately by
[`tests/proportion_shifted_clr_reference`](../../tests/proportion_shifted_clr_reference/).
Both references record their upstream formula and revision, while the normal
test suite remains offline.

Current PFlog transformed values are checked directly as

$$
\operatorname{center}_{\text{row}}\left[\log(1+4\alpha x)\right].
$$

The defining tests verify all of the following:

1. Count-shifted CLR equals a dense direct implementation.
2. Every count-shifted CLR row sums to zero within floating-point tolerance.
3. Count-shifted CLR permits a zero-total row.
4. Proportion-shifted CLR rejects a zero-total row.
5. Proportion-shifted CLR is invariant to multiplying each row by a positive
   scalar; count-shifted CLR is not claimed to be.
6. With $G$ genes, `shifted_clr(count_shift=a)` equals
   `dirichlet_clr(concentration=G*a, uniform prior)`.
7. `shifted_log(count_shift=a)` has the same centered dense matrix as
   $\log(x+a)$.
8. Sparse support is unchanged for all unclipped log-correction matrices.
9. Representation ranks are 0, 1, or at most 2 as listed above.
10. Feature masking happens after full-universe normalization.
11. Old `pseudocount=` calls fail instead of silently changing scale.
12. Metadata records the shift domain and exact parameterization.

Dense-operator, singular-value, component-subspace, zero-variance, float32,
and float64 checks cover each builder.

## Exact Anscombe: sparse, but deferred

The exact NB variance-stabilizing transform can be written stably as

$$
g_\alpha(x)
=
\frac{2}{\sqrt\alpha}\operatorname{asinh}(\sqrt{\alpha x})
=
\frac{1}{\sqrt\alpha}\operatorname{acosh}(1+2\alpha x),
\qquad \alpha>0.
$$

It has the Poisson-limit transform

$$
g_0(x)=2\sqrt{x}.
$$

Because $g_\alpha(0)=0$, applying it only to `X.data` produces an exact CSR
matrix with unchanged support. In this repository it would be a rank-zero
`SparseLowRankMatrix`; sparse-plus-low-rank feasibility is not a concern.

Nevertheless, do not add `nb_anscombe_pca` in the semantic-correction work.
Before adding it, decide and document:

- whether `alpha` is scalar or per gene;
- whether the package estimates `alpha` or only consumes supplied values;
- how AnnData-aligned dispersion arrays and masks behave;
- whether this raw-count VST is useful without a separate depth model;
- how its purpose differs from the package's scaled-NB residual PCA.

If it is later added, call it `nb_anscombe_pca`, implement the stable `asinh`
form with the exact Poisson limit, and keep it separate from
`shifted_log(count_shift=1/(4*alpha))`. The latter is only a large-count log
approximation, not the exact Anscombe transform.

## Other methods intentionally out of scope

### Seurat "CLR"

Do not implement it. It is not a CLR coordinate transform, is not needed for
the paper reconciliation, and adds a data-dependent shift rule with little
value to this package.

### Ahlmann--Eltze--Huber normalized Anscombe log

Do not implement it. Dividing by a size factor and then adding
$1/(4\alpha)$ gives a cell-specific raw-count shift $r_c/(4\alpha)$. It is
neither the fixed-count approximation adopted here nor needed as a
compatibility target.

### Arbitrary size-factor and library-target logs

Transforms such as

$$
\log\left(1+\frac{x_{cg}}{r_c}\right)
\quad\text{or}\quad
\log\left(1+K\frac{x_{cg}}{s_c}\right)
$$

are exactly sparse and may be useful baselines. Defer them until the package
has an explicit policy for supplied versus estimated size factors. When added,
name the divisor in the function and metadata; do not describe their
post-division shift as a fixed raw-count pseudocount.

### ILR and post-log proportional fitting

Do not add them for the reasons above. ILR is spectrally redundant with a
complete CLR analysis, while post-log proportional fitting is a separate row
rescaling operation rather than a coordinate choice.

## Current implementation

The implementation has the following contracts:

- no canonical public name hides whether its shift is on the count or
  composition scale;
- current PFlog is reproducible as fixed-count `shifted_clr`;
- the historical formulation remains reproducible under an explicit name;
- uniform `dirichlet_clr` and scalar count-shifted CLR share implementation and
  agree exactly up to floating-point error;
- all transforms retain exact sparse-plus-low-rank representations;
- old ambiguous calls fail rather than silently change results;
- AnnData masks continue to be applied after full-universe normalization;
- documentation distinguishes exact Anscombe from its shifted-log
  approximation; and
- Seurat CLR and the AEH recipe do not enter the codebase.

Potential additions such as exact Anscombe or arbitrary size-factor logs
should be evaluated on representative data before receiving public APIs.

The package-level framing is:

> `sparse-count-pca` performs exact implicit PCA for named count transforms.
> For log-ratio methods, the shift domain and coordinate system are explicit:
> fixed count shifts, fixed composition shifts, and Dirichlet prior counts are
> different models, even when the literature has called more than one of them
> PFlog.
