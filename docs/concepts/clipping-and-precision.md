# Clipping and numerical precision

## Residual clipping is on by default and exact

Residual transforms clip by default. The threshold applies to the computed
residual values before column centering, and `clip` accepts four kinds of
value:

| `clip` | Threshold |
| --- | --- |
| `"seurat"` (default) | \(\sqrt{n_\mathrm{obs}/30}\) |
| `"scanpy"` | \(\sqrt{n_\mathrm{obs}}\) |
| positive finite float | that number |
| `None` | no clipping |

Here \(n_\mathrm{obs}\) is the number of rows — cells — of the count matrix
the transform is fitted on, taken as passed.

The two names come from the defaults of
[SCTransform](https://satijalab.org/seurat/reference/sctransform) and
[Scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals_pca.html)
respectively. A name resolves only a numeric threshold: it applies to every
residual family and model, not only to Pearson residuals, and it resolves to a
different number for a different dataset.

Two clipping modes are available:

- `clip_mode="symmetric"` (the default) clips to `[-clip, clip]`;
- `clip_mode="upper"` clips to `(-inf, clip]`, leaving the negative residuals
  untouched.

`clip` and `clip_mode` are independent: a name selects a threshold, not a
mode. `clip="seurat", clip_mode="upper"` means upper-only clipping at
\(\sqrt{n_\mathrm{obs}/30}\), not SCTransform's symmetric clipping.

### Reusing a specification across datasets

A `Residual` specification stores `clip` as passed and resolves a name at each
fit, so reusing one `Residual(clip="seurat")` on two datasets can produce two
different thresholds. The fitted transform, the PCA result, and any retained
operator record the resolved value as `params["clip_threshold"]` alongside the
request in `params["clip"]`.

```python
result = residual_pca_matrix(counts, n_comps=50)
result.params["clip"]            # "seurat"
result.params["clip_threshold"]  # sqrt(counts.shape[0] / 30)
```

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

Because the default clips symmetrically, both the growth and the guard are
reachable under default arguments. How much support grows depends on the
dataset. If a run raises on the guard, raise `clip_max_nnz_ratio`, switch to
`clip_mode="upper"`, or set `clip=None`.

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
