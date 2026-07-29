# Sparse plus low rank

## Why a sparse input can produce a dense transform

A sparse count matrix stores observed nonzero counts and omits zeros. Many
normalizations assign a nonzero transformed value to an observed zero. Pearson
residuals, deviance residuals, and shifted CLR coordinates are therefore
generally dense even when the count matrix is very sparse.

Other normalizations, including the common library-size normalization followed
by `log1p`, map zero to zero and remain sparse before PCA. PCA nevertheless
subtracts each column mean, which generally makes the centered matrix dense.
PCA implementations using iterative solvers such as IRLBA or ARPACK can already
apply this rank-one centering correction implicitly in matrix-vector products.
This package uses the same pattern and extends it to methods whose normalized
matrix is dense even before centering.

A dense `float64` matrix with `n_obs * n_vars` entries requires
`8 * n_obs * n_vars` bytes of storage. Assuming 32-bit sparse indices and row
pointers, and ignoring small object overhead, a CSR `float64` matrix with `nnz`
nonzero entries requires

```text
8 * nnz + 4 * nnz + 4 * (n_obs + 1) bytes
```

### A worked example

Take 500,000 cells by 20,000 genes at 5% density, so `nnz = 5 * 10**8`. The
transform's entry storage is then

| Representation | Bytes | |
| --- | --- | --- |
| Dense `float64` | `8 * 5e5 * 2e4` | 80.0 GB |
| CSR `float64` correction | `12 * 5e8 + 4 * 5e5` | 6.0 GB |
| Rank-`k` factors, `k = 2` | `8 * 2 * (5e5 + 2e4)` | 8.3 MB |

The low-rank factors are negligible at this scale, so the ratio is
approximately `n_obs * n_vars / (1.5 * nnz)`, or about 13× here. At 1% density
the same shapes give roughly 66×; at 20% density, roughly 3×.

Two caveats keep this from being a runtime memory figure. Symmetric residual
clipping can add entries to the sparse correction, and PCA allocates its own
working arrays and outputs. See
[what remains in memory](#what-remains-in-memory).

## Zero baselines factor

For every supported PCA transform, the values associated with observed zeros
can be written as a small sum of outer products. The complete transformed
matrix has the form

```text
A = S + U @ V.T
```

where `S` is a sparse correction matrix and `U @ V.T` has small rank. `S`
stores how observed nonzero counts differ from the factored zero baseline.

The details vary by transform:

- residual expectations factor through cell totals and gene parameters;
- the package's count-scale log transform computes
  `log1p(x / count_shift)` without library-size normalization; zeros remain
  zero, so only PCA centering adds a dense rank-one term;
- CLR subtracts a row-specific mean, adding a rank-one term.

The [package specification](../development/specification.md#sparse-plus-low-rank-representation)
gives the exact formulas.

## PCA centering is another low-rank term

PCA decomposes the column-centered matrix. After selecting the PCA variables,
the package computes the column mean `mu` and represents

```text
A_centered = S + U @ V.T - ones @ mu.T,
```

where the centering correction has rank one.

## LinearOperator and truncated SVD

The package exposes matrix-vector and transpose-matrix-vector products through
`scipy.sparse.linalg.LinearOperator`. Those products combine a sparse multiply
with small dense low-rank multiplies; they do not allocate all entries of `A`
or `A_centered`.

PCA uses `scipy.sparse.linalg.svds` with the ARPACK solver. SciPy accepts a
`LinearOperator`, so the transformed matrix and its transpose need not be
constructed explicitly. Returned components are ordered and oriented using the
scikit-learn PCA convention used by Scanpy.

"Exact" describes the represented matrix at the selected floating-point dtype.
The PCA calculation then has the ordinary numerical accuracy considerations of
other ARPACK-based workflows.

## What remains in memory

Avoiding the dense transform does not mean using no additional memory. During
fitting, the count matrix is converted to canonical in-memory CSR. The fitted
representation then retains:

- one sparse correction matrix;
- the low-rank factors;

PCA additionally allocates temporary arrays during the calculation and the
returned score, component, and variance arrays.

Backed sparse inputs are currently loaded into memory during canonicalization.
The two-step API can bound the output memory used for later materialization,
but fitting is not yet an out-of-core operation.

See [transform once, reuse several times](../guides/transform-reuse.md) for
bounded materialization and process-lifetime details.
