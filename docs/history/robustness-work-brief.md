# Correctness and numerical robustness work brief

> Historical note: this brief predates the `sparse-count-pca` rename and the
> completed generic PCA refactor. See [architecture](../architecture.md) and
> the [package specification](../specification.md) for the current organization
> and API.

## Purpose

Harden the current residual-PCA implementation before expanding it to other
implicit transforms. The repository is a strong prototype, but several
correctness, numerical-stability, validation, and Scanpy-compatibility issues
should be resolved first.

This brief records decisions already made. Start future work by writing the
failing regression tests described below; do not reopen the settled API choices
unless implementation evidence requires it.

## Current baseline

- Public residual API: `residual_pca`, `residual_pca_matrix`; all PCA methods
  return `PCAResult`.
- Existing test suite: 86 tests passing at the time of review.
- Ruff passes.
- The reviewed working tree already contained user changes. Preserve them and
  inspect `git diff` before editing.

## Decisions already made

1. A centered transformed matrix with numerically zero variance should raise a
   clear `ValueError`; PCA directions are undefined in that case.
2. For scaled-NB residuals, select Poisson versus NB once per gene using
   `alpha_j`, then use that choice consistently for every cell and both zero and
   nonzero counts.
3. Change the representation/operator default from `float32` to `float64`.
   Continue supporting `float32` as an explicit low-memory, approximate mode.
4. Restrict the requested representation/output dtype to `float32` and
   `float64`.
5. Keep `NaN` loadings for genes excluded by `mask_var`; document this as an
   intentional difference from Scanpy.
6. Otherwise match cost-free Scanpy behavior: exact `key_added` keys, strict
   boolean mask validation, and harmonized standard metadata.
7. Retain exact clipping semantics and filter binary-search candidates with the
   direct strict product predicate before applying the support-growth guard.
8. Reuse the CSR row-support vector rather than constructing it again inside
   clipping.
9. Chunked support processing is a future scalability improvement, not part of
   this work.
10. Generic sparse-plus-low-rank architecture, PFlog support, and release work
    are explicitly deferred.

## Workstream 1: stable total variance and zero-variance handling

### Problem

`SparseLowRankLinearOperator.__init__` computes

```text
||S||^2 + 2 <S, uv^T> + ||uv^T||^2
```

and `frobenius_squared_centered()` subsequently subtracts
`n_obs * ||mean||^2`. These identities are correct but can catastrophically
cancel because the sparse correction and low-rank baseline may be individually
large while the residual matrix is small. Clamping a negative result to zero
hides the error.

Known reproduction:

```python
b = 10**6
D = np.tile([b, 2 * b, 3 * b, 4 * b], (1000, 1))
D[0, 0] += 1
D[0, 1] -= 1
```

With float64, the implementation reported `total_variance == 0`, while direct
materialization of the small operator gave approximately `1.5e-9`.

Exactly identical rows should yield a zero centered matrix. Currently ARPACK
may return arbitrary numerical PCs, infinite explained-variance ratios, or a
starting-vector error.

### Required work

- Develop an `O(nnz + n_obs + n_vars)` stable calculation based on squared
  actual residual values on stored support plus the implicit structural-zero
  baseline, rather than cancellation between representation components.
- Handle sparse and nearly dense columns carefully; do not replace one severe
  subtraction with another.
- Calculate/check centered total variance before invoking ARPACK.
- Define a scale-aware numerical-zero criterion.
- Raise a clear `ValueError` for zero centered variance.
- Ensure `total_variance` describes the same stored operator that ARPACK
  decomposes.

### Acceptance tests

- The high-count near-constant reproduction agrees with a dense calculation.
- Identical rows raise the documented error before ARPACK.
- Ordinary matrices continue to match dense total variance and explained-
  variance ratios.
- Cover both float64 and explicit float32.
- Include sparse, dense-support, clipped, and near-null cases.

## Workstream 2: consistent scaled-NB model selection

### Problem

The contract says genes with `alpha_j < ALPHA_EPS` use the Poisson limit.
Current deviance code uses gene-level `alpha_j` for the zero-count low-rank
factor but cell-adjusted `alpha_tilde = alpha_j / s_i` on nonzero support. This
can create a hybrid transform that is NB at zeros and Poisson at nonzeros.

The current size factor is mean-one:

```text
s_i = n_i / mean(n)
```

It is not the raw cell sum. Do not size-factor-normalize the count matrix and
then apply an ordinary NB likelihood to fractional counts.

### Required work

- Normalize/validate alpha once.
- Build a gene-level Poisson mask from `alpha_j` only.
- Apply that choice consistently in Pearson, zero-count deviance factors, and
  nonzero deviance evaluation.
- For positive alpha, evaluate the NB model even when `alpha_j / s_i` is small;
  numerical stability belongs in the formula, not in a cell-dependent model
  switch.
- Remove duplicated/discarded validation where practical, including the current
  zero-dimensional NumPy-scalar inconsistency.

### Acceptance tests

- Strongly unequal cell depths with alpha just above and below the threshold.
- The same gene never changes probability family based on count support.
- Independent likelihood comparisons for both branches.
- Pearson and deviance agree with their documented Poisson limits.

## Workstream 3: stable deviance evaluation

### Problem

The direct Poisson/binomial/NB formulas handle zero boundaries with `xlogy` but
can lose all precision when `x` and `mu` are large and close. A genuine small
positive deviance may become negative and then be silently clipped to zero; a
nearby case may also be spuriously inflated.

### Required work

- Use relative-error `log1p` forms near the mean.
- Add a series expansion for sufficiently small relative differences.
- Derive the analogous stable NB expression, including small positive
  dispersion without switching models.
- Clamp only negative values consistent with final-rounding noise. Materially
  negative results should trigger stable recomputation or an error.
- Keep well-conditioned paths simple and vectorized.

### Oracle and regression strategy

Use three layers:

1. Compare the direct formula and series formula over a well-conditioned
   overlap region.
2. Test continuity immediately on both sides of the branch threshold.
3. Compare huge, near-mean cases with arbitrary-precision `mpmath` values.

`mpmath` may be a test-only dependency. Alternatively, check in a small table
of high-precision constants with a documented regeneration script. Avoid using
NumPy `longdouble` as the cross-platform oracle.

Include exact equality, zero boundaries, differences of 1 and 10 at very large
means, values around the switch threshold, and the NB-to-Poisson limit.

## Workstream 4: input, mask, and dtype validation

### Required changes

- Reject complex and other non-real/non-numeric count data before any cast.
- Replace default-tolerance integer checking with an intentional rule, e.g.
  `np.allclose(data, np.rint(data), rtol=0, atol=<small explicit tolerance>)`,
  or exact equality if chosen.
- Require masks and `adata.var[mask_key]` columns to be genuinely boolean;
  reject integers, floats, `NaN`, and strings instead of coercing them.
- Accept only `float32` and `float64` for the representation/operator `dtype`.
- Change public defaults, docs, spec, and recorded parameters to `float64`.
- Treat raw-count input dtype separately from operator dtype: accept safely
  convertible real integer/count-like floating input, but never silently
  discard information.

### Regression cases

- Complex sparse and dense counts.
- Large fractional values such as `100000.4`.
- Nonfinite, negative, and explicit-zero sparse entries.
- Numeric/string/nullable masks.
- Rejection of requested float16, float128, integer, and complex operator
  dtypes.

## Workstream 5: Scanpy alignment and result metadata

### Required behavior

- For `key_added="foo"`, write scores, loadings, and metadata under the exact
  key `"foo"`, matching Scanpy. Do not create `obsm["X_foo"]`.
- Preserve `NaN` for excluded loadings and document this single intentional
  incompatibility.
- Match Scanpy's standard parameter names/meaning where applicable, including
  centering, mask/highly-variable selection, and layer metadata.
- Store package-specific residual parameters as a documented superset.
- Add reproducibility fields currently omitted where appropriate: `n_comps`,
  `random_state`, `tol`, `check_values`, resolved dtype, and package version.
- Ensure matrix-result and AnnData metadata tell the same story.

Update tests that currently lock in the `X_` prefix for custom keys.

## Workstream 6: clipping equality and support reuse

### Clipping

The binary search in `_clipped_zero_locations` should remain a candidate
generator. Before counting candidates or enforcing `clip_max_nnz_ratio`, filter
them with the exact requested predicate:

```python
u[rows] * v[cols] < -clip
```

Add a regression where the only structural-zero residual is exactly `-clip`;
it must not create a correction or trigger the growth guard.

### Support reuse

`build_residual_representation` creates

```python
rows = np.repeat(np.arange(n_obs), np.diff(X.indptr))
```

to map each `X.data` entry to its cell. `apply_clipping` currently constructs
the same length-`nnz` vector again while the first is live. Pass `rows`, or the
already calculated `uv_nonzero`, into clipping. This saves approximately
`sizeof(intp) * nnz` bytes without changing the algorithm.

Reuse or release other support-sized temporaries where obvious, but do not turn
this work into the deferred chunking refactor.

## Test-suite improvements across workstreams

- Add independent `scipy.stats` or arbitrary-precision likelihood oracles; do
  not duplicate production algebra or import its branch threshold into the
  oracle.
- Add randomized small count matrices across every supported model/residual.
- Add high-count, highly unequal-depth, near-null, and exact-boundary fixtures.
- Exercise the default float64 path and explicit float32 path.
- Retain the Townes reference tests; add an independent scaled-NB reference.
- Verify ARPACK singular-triplet residuals where useful.

## Suggested execution order

1. Add failing tests for all known reproductions.
2. Implement stable total variance and zero-variance detection.
3. Fix gene-wise Poisson/NB selection.
4. Implement stable deviance formulas and high-precision oracle tests.
5. Tighten count, mask, and dtype validation; switch defaults to float64.
6. Align Scanpy keys and metadata while retaining NaN excluded loadings.
7. Fix clipping equality and reuse the support vector.
8. Run the full suite, Ruff, and targeted numerical stress tests.
9. Update the package specification and `README.md` to match final behavior.

## Explicitly deferred work

### Generic implicit-PCA architecture

Later, extract PCA/SVD/statistics from `_compute_residual_pca` into a generic
sparse-plus-low-rank engine. Do not add another residual/model enum for
non-residual transforms. Rank one is sufficient for shifted CLR; general
`S + U V.T` can wait for a concrete rank-q use case.

### PFlog / shifted CLR

The full filtered gene universe should define the CLR row mean; apply
`mask_var` only afterwards for PCA. The supplied June 10, 2026 formulation and
the newer June 22 formulation use different pseudocount parameterizations, but
both are exactly sparse minus a row-centering rank-one term.

When implemented, compare against a pinned commit/formula from
<https://github.com/pachterlab/BHGP_2022> and the official `scclr`
implementation. Check transformed values/row centers, operator products,
singular values, and PCA subspaces. Normal tests should use checked-in small
reference results or optional regeneration tooling rather than network access
or a moving upstream checkout.

### Scalability

Chunked CSR support processing remains a future improvement. Record peak-memory
benchmarks before undertaking it.

### Release readiness

Before release, add a license and project metadata, tested dependency floors,
CI across supported Python versions, direct test dependencies, installation
instructions, stable documentation URLs, and citation/provenance information.
Separate the normative specification from historical implementation notes if
the current specification continues to grow.
