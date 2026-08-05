# All-zero gene and cell notes

Working notes recording the decisions on empty genes and empty cells, to be
folded into the specification and guides. Not part of the built site.

## The rule for genes

**Exclude where the transformed value is undefined; keep where it is defined.**

One sentence covers every family. It splits along the mathematics rather than
along convenience, and it requires no change to any transform's formula. The
cell axis is treated separately below, because the same test does not split the
families there.

| Family | Value at an all-zero gene | Affects other genes? | Decision |
| --- | --- | --- | --- |
| Residual (Poisson, binomial, scaled-NB) | undefined, `0/0` | no | exclude from fit and PCA, `NaN` in `varm` |
| Correspondence analysis | undefined, zero mass | no | exclude from fit and PCA, `NaN` in `varm` |
| Count-scale shifted CLR | defined, `log(a) − m_i` | **yes** | keep, formula unchanged |
| Composition-scale shifted CLR | defined | **yes** | keep, formula unchanged |
| Dirichlet log, Dirichlet CLR | defined, `log(α_j)` | **yes** | keep, formula unchanged |
| Size-factor log norm (not implemented) | defined, exactly `0` | no | keep, zero-variance column |

## Why residual and CA exclude

Exclusion is provably lossless. An all-zero gene contributes nothing to any row
total, to the grand total, or to any other gene's proportion, so `n_i`, `S`, and
every retained `p_k` are bit-identical with or without it. Auto-exclusion
changes no number for any retained gene or any cell — only the reported
`pca_n_vars`.

Computed literally the residual is `0/0`. Leaving that as `NaN` does not
quarantine itself to one column: it propagates through the SVD and returns `NaN`
for every cell. The limit as `μ → 0` is `0` for both Pearson and deviance, so
defining the residual as `0` is also coherent and yields identical cell scores
with exactly zero loadings. We still exclude and write `NaN`, because a stored
`0` is indistinguishable from a genuine zero loading.

This replaces the previous hard error, which fired on a common and harmless
situation: subsetting cells without re-filtering genes.

## Why CLR and Dirichlet keep them

Redefining the composition as being over *detected* parts would be a
specification change. It breaks bit-compatibility with `runorm`/`scclr` and with
our own PFlog oracle (`tests/shifted_clr_reference/oracle.py`), and it is
counterintuitive in exactly the cell-subsample case that motivated the question,
where the gene is legitimately part of the declared feature space.

The cost, stated honestly: the CLR row-mean divisor counts all `n_vars` columns,
so `b` empty genes shrink the row-centering by `G_A / G`. Measured on 600 cells
by 1,887 genes, as the maximum principal angle between the top-10 PC subspaces
against an unpadded baseline:

| Empty fraction | `G_A/G` | Subspace rotation |
| ---: | ---: | ---: |
| 5% | 0.95 | 4.7° |
| 10% | 0.90 | 8.7° |
| 25% | 0.75 | 19.1° |
| 50% | 0.50 | 33.9° |

Real and measurable, but modest at realistic fractions — comparable to nudging
`count_shift`. The decision is to make it **discoverable rather than corrected**.

## Recorded metadata

- `n_empty_vars` — empty genes in the normalization universe. Present for every
  family. The dilution factor above is
  `(normalization_n_vars − n_empty_vars) / normalization_n_vars`.
- `n_empty_vars_excluded` — empty genes dropped from PCA. Residual and CA only;
  `0` elsewhere.

The two scopes differ deliberately. `n_empty_vars_excluded` records an *action*
over mask-selected genes, so a masked-out empty gene is not counted. Normalization
is fitted before `mask_var`, so for CLR and Dirichlet a masked-out empty gene
still dilutes; the universe-scoped `n_empty_vars` is the diagnostic that makes
that visible. Without it the dilution is invisible, which is the whole reason for
reporting anything.

## `varm` convention

- `NaN` — the gene was not in the fit (excluded by `mask_var`, or empty under
  residual/CA).
- `0` — the gene was in the fit and genuinely has zero loading.

This keeps `varm` self-describing and is why exclusion writes `NaN` rather than
adopting the define-as-zero alternative.

## Implementation notes

- Validate `n_comps` *after* auto-exclusion, since `pca_n_vars` shrinks.
- Give a clear error if exclusion empties the selection, as the analogue of the
  existing `mask_var selected zero genes`.
- CA applies its mask before fitting, so column masking can produce an empty
  *row*. `_correspondence.py:153` currently rejects that; see the open cell-axis
  question below.

## Column scaling, if added later

Not supported today. If a size-factor log-norm transform gains unit-variance
column scaling:

- Restrict scaling to that family. This restriction is **load-bearing**, not a
  taste judgment: a `var == 0` guard does nothing for CLR, where an empty gene's
  column is `log(a) − m_i` with finite nonzero variance — measured at the 1st
  percentile of real genes and correlated 0.973 with log depth. It would pass the
  guard and be *promoted* to unit variance, turning a pure depth artifact into a
  full-strength PCA variable.
- The guard is exact for log norm. `log1p(0)` is bitwise `0.0`, so the column is
  exactly constant and `var == 0` needs no tolerance. Retain and set to `0`,
  matching `sc.pp.scale`.
- The exact-zero case is the easy one. The dangerous case is a gene detected in a
  handful of cells: tiny nonzero variance, inflated enormously by scaling. That
  needs a variance floor or a `min_cells`-style guard. Scanpy has the same hole.

## What other packages do

| Package / path | All-zero gene | All-zero cell |
| --- | --- | --- |
| Scanpy `normalize_total` + `log1p` + `pca` | kept silently, loading `-0.` | warns `"Some cells have zero counts"`, divisor coerced to 1, finite meaningless embedding |
| Scanpy `experimental.pp.normalize_pearson_residuals` | silent all-`NaN` column, only a numpy `RuntimeWarning` | `NaN` row |
| Scanpy `experimental.pp.highly_variable_genes` | **explicitly dropped**, refilled with `0` (`_highly_variable_genes.py:170-173, 207-208`) | n/a |
| Seurat / sctransform | dropped by `min_cells = 5`, object shrinks (`vst.R:200-201`) | unguarded, `log10(0) = -Inf` (`utils.R:38-39`) |
| cleartools `runorm` / `scclr` | **kept, counted in the CLR divisor** `D = n_cols` (`lib.rs:11-12`) | zeroed deliberately, unit-tested (`lib.rs:425`, `:562`) |
| this package, previously | hard error | hard error |

No other package errors on either axis. Scanpy is internally inconsistent: its
Pearson-residual HVG drops empty genes while its Pearson-residual normalization
emits silent `NaN`. Our two decisions each match an existing precedent — residual
and CA exclusion follows Scanpy's Pearson HVG, CLR retention follows cleartools.

## All-zero cells

### There is no cross-cell coupling in any family

An empty cell contributes nothing to any column total or to the grand total, so
every `p_j` and every other cell's `n_i` are unchanged by its presence. CLR and
Dirichlet are row-wise operations, so they cannot change another row at all. The
only coupling is `O(1/n_obs)`, through PCA column centering and the `n − 1`
variance divisor.

So the "it is part of the declared composition" argument that kept empty genes in
CLR and Dirichlet **has no analogue on the cell axis**.

| Family | Empty-cell row |
| --- | --- |
| Residual | undefined, `0/0` (limit is the zero vector) |
| Correspondence analysis | undefined; zero mass, and coordinates divide by `sqrt(r_i)` |
| Count-scale shifted CLR | defined, exactly the zero vector |
| Composition-scale shifted CLR | undefined; the composition of an empty cell is `0/0` |
| Dirichlet log, Dirichlet CLR | defined, equals the prior — identical for every empty cell |

Every family is therefore either undefined or collapses to a single fixed
artifact point shared by all empty cells. Definedness splits the gene families; it
does not split the cell families. All of them land on "must not be in the fit."

### Decision: keep the hard error

Empty cells stay rejected in count canonicalization (`_counts.py:137`). We do
**not** mirror the gene-axis exclude-and-`NaN` treatment, for one measured reason:
the marker is not inert.

With scanpy 1.12.1, a `NaN` row in `obsm["X_pca"]` makes `sc.pp.neighbors`
succeed with no error and no warning, on both the exact path (`n_obs = 60`) and
the approximate pynndescent path (`n_obs = 6000`). Worse, 35 and 55 *other* cells
respectively acquired edges **to** the `NaN` cell, and the resulting
`connectivities` matrix contains no `NaN` at all — so nothing downstream can
detect the corruption.

The governing principle, so the asymmetry is not mistaken for inconsistency:

> **Exclude-and-mark only where the mark is inert.** `NaN` in `varm` is never
> consumed numerically. `NaN` in `obsm` is consumed, and silently corrupts other
> cells' neighborhoods.

The second reason is that nothing is lost. An empty cell carries no information
under any transform, so the error asks the user to delete a row of zeros, not
data. Document the remedy: `sc.pp.filter_cells(adata, min_counts=1)`, or
`adata = adata[adata.X.sum(axis=1).A1 > 0]`.

### The `.raw` asymmetry

AnnData slices `.raw` along `obs` but not along `var`. Verified: slicing genes
`(20, 10) → (20, 5)` leaves `.raw` at `(20, 10)`; slicing cells
`(20, 10) → (10, 10)` takes `.raw` to `(10, 10)` as well.

So "filter upstream and keep the originals in `.raw`" is available for genes and
is destructive for cells. That asymmetry is real, and it is part of why the
gene-axis error was worth removing. It does not rescue the cell axis, because
what gets destroyed there is a row of zeros.

### What other packages do with empty cells

| Package / path | Behavior |
| --- | --- |
| Scanpy `normalize_total` + `log1p` + `pca` | warns `"Some cells have zero counts"`, coerces the divisor to 1, cell receives a finite meaningless embedding |
| Scanpy `experimental.pp.normalize_pearson_residuals` | silent `NaN` row |
| Seurat `LogNormalize` | silent pass-through; the sparse column has no stored entries, so the division never executes (`preprocessing.R:3612`) |
| sctransform | unguarded, `log_umi = log10(0) = -Inf` (`utils.R:38-39`) |
| cleartools `runorm` / `scclr` | zeroed deliberately, unit-tested (`lib.rs:425`, `:562`) |
| this package | hard error (`_counts.py:137`) — retained |

Neither Scanpy nor Seurat filters by default: `CreateSeuratObject` defaults to
`min.cells = 0, min.features = 0` (`objects.R:418-419`), and the standard tutorial
passes `3` and `200` explicitly. In every other stack an empty cell either reaches
PCA silently or produces `-Inf`.

We are the only package that errors here, and on this axis that is the defensible
position. The alternatives on offer are a confident meaningless coordinate
(Scanpy, `runorm`), a `NaN` row that silently corrupts the neighbor graph, or
`-Inf`.
