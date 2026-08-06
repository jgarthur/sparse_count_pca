# Clipping and numerical precision

## Clipping is opt-in and exact

Residual transforms do not clip outliers by default. Set a positive finite
`clip` threshold to enable clipping before PCA centering. The threshold is in
residual units, not counts, and applies to the uncentered residual values.

Two modes are available:

- `clip_mode="symmetric"` clips to `[-clip, clip]`;
- `clip_mode="upper"` clips only values above `clip`, leaving the lower tail
  untouched.

The implementation represents the clipped residual matrix exactly.

## Why symmetric clipping may grow support

Zero-count residuals are negative for the supported residual models. Symmetric
clipping can therefore change an entry that was represented by the low-rank
zero baseline. The changed entries must move into the sparse correction, which
can expand its support.

`clip_max_nnz_ratio` bounds that expansion. It is a multiple of the input count
matrix's stored nonzeros, so `2.0`, the default, permits the sparse part to grow
to just under twice that. The operation raises `RuntimeError` before
constructing a correction that meets or exceeds the limit. Use `1.0` to reject
any support growth or `None` to allow unlimited exact expansion.

Upper-only clipping leaves negative zero-count residuals unchanged and cannot
expand support for the implemented models.

## Calculation dtype

The default `dtype="float64"` controls the sparse correction, low-rank factors,
linear operator, and arrays passed to ARPACK. It is the recommended mode for
numerical accuracy and parity testing.

Explicit `dtype="float32"` is a lower-memory approximate calculation mode. It
changes the matrix presented to the SVD solver; it is not merely an output
storage conversion. No other representation dtype is supported.

## Degenerate inputs

The package rejects a transformed matrix whose centered variance is
numerically zero. This occurs when every observation has the same transformed
values across the selected variables—for example, when the same count row is
repeated for every cell. PCA directions are not defined in that situation. An
all-zero count matrix is rejected earlier because its cells have zero total
counts.

ARPACK also requires `n_comps` to be strictly smaller than both the number of
observations and the number of selected variables whose transformed column is
not identically zero.

For exact validation boundaries, see the
[clipping contract](../development/specification.md#clipping) and
[numerical implementation](../development/specification.md#numerical-implementation)
in the package specification.
