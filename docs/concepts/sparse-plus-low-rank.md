# Sparse plus low rank

## Why a sparse input can produce a dense transform

A sparse count matrix stores observed nonzero counts and omits zeros. Many
normalizations assign a nonzero transformed value to an observed zero. Pearson
residuals, deviance residuals, and shifted CLR coordinates are therefore
generally dense even when the count matrix is very sparse.

A dense `float64` matrix with `n_obs * n_vars` entries requires
`8 * n_obs * n_vars` bytes for its entries alone. CSR storage instead scales
mainly with the number of stored entries, plus indices and row pointers. The
ratio depends on input density, count and index dtypes, transform parameters,
and whether clipping expands support, so the package does not claim one
universal memory multiplier.

## Zero baselines factor

For every supported PCA transform, the values associated with observed zeros
can be written as a small sum of outer products. The complete transformed
matrix has the form

```text
T = S + U @ V.T
```

where `S` is a sparse correction matrix and `U @ V.T` has small rank. `S`
stores how observed nonzero counts differ from the factored zero baseline.

The details vary by transform:

- residual expectations factor through cell totals and gene parameters;
- fixed-count log corrections are nonzero only on observed support;
- CLR subtracts a row-specific mean, adding a rank-one term.

The [normative specification](../development/specification.md#sparse-plus-low-rank-representation)
gives the exact formulas.

## PCA centering is another low-rank term

Ordinary PCA decomposes the column-centered matrix. After selecting the PCA
variables, the package computes the column mean `mu` and represents

```text
T_centered = S + U @ V.T - ones @ mu.T.
```

The centering correction has rank one. It therefore preserves the same
sparse-plus-low-rank structure.

## LinearOperator and truncated SVD

The package exposes matrix-vector and transpose-matrix-vector products through
`scipy.sparse.linalg.LinearOperator`. Those products combine a sparse multiply
with small dense low-rank multiplies; they do not allocate all entries of `T`.

PCA currently calls `scipy.sparse.linalg.svds` with the ARPACK solver to obtain
the requested leading singular triplets. SciPy accepts a `LinearOperator`, so
the transformed matrix and its transpose need not be constructed explicitly.
The package then sorts the triplets, applies a deterministic sign convention,
and calculates PCA scores and variance statistics.

"Exact" describes the matrix representation and clipping semantics, up to the
selected floating-point dtype. The requested truncated SVD remains a numerical
iterative calculation subject to dtype, tolerance, and solver convergence.

## What remains in memory

Avoiding the dense transform does not mean using no memory. The current
implementation owns:

- a canonical in-memory CSR count matrix during fitting;
- a sparse correction and owned support snapshot;
- the small low-rank factors;
- solver work arrays and returned PCA outputs.

Backed sparse inputs are currently loaded into memory during canonicalization.
The two-step API can bound the output memory used for later materialization,
but fitting is not yet an out-of-core operation.

See [transform once, reuse several times](../guides/transform-reuse.md) for
bounded materialization and process-lifetime details.
