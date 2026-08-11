# Clipping and numerical precision

## Clipping is opt-in and exact

Residual transforms do not clip outliers by default. Set a positive finite
`clip` threshold to enable clipping. The threshold applies to the computed
residual values before column centering.

Two clipping modes are available:

- `clip_mode="symmetric"` (the default, when `clip` is set) clips to
  `[-clip, clip]`;
- `clip_mode="upper"` clips to `(-inf, clip]`, leaving the negative residuals
  untouched.

## Symmetric clipping may expand the sparse matrix support

For every supported model and residual type, residuals at zero-count entries
are negative (except for all-zero genes). Clipping those zero-count residuals
that fall below `-clip` cannot be done using the low-rank zero baseline alone.
Instead, clipping is performed by adjusting the sparse correction term, which
expands its support.

`clip_max_nnz_ratio` bounds that expansion. It is a multiple of the input count
matrix's stored nonzeros, so the default value `2.0` permits the sparse part to
grow to just under twice that. The operation raises `RuntimeError` if the
correction would meet or exceed the limit. Use `1.0` to reject any support
growth, `clip_mode="upper"` to avoid it by construction, or `None` to disable
the guard and allow unlimited exact expansion.

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
