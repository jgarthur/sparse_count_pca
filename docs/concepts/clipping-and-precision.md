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

Here \(n_\mathrm{obs}\) is the number of observations in the count matrix the
transform is fitted on. It reflects any cell filtering you performed
beforehand, but not gene masking or later PCA variable selection.

The two names come from the defaults of
[SCTransform](https://satijalab.org/seurat/reference/sctransform) and
[Scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.experimental.pp.normalize_pearson_residuals_pca.html)
respectively. They are named numbers, nothing more: they apply to every
residual family and model, not only to Pearson residuals, and the same name
resolves to a different number for a different dataset. Pass `clip=None` to
recover unclipped residuals.

Two clipping modes are available:

- `clip_mode="symmetric"` (the default) clips to `[-clip, clip]`;
- `clip_mode="upper"` clips to `(-inf, clip]`, leaving the negative residuals
  untouched.

`clip` and `clip_mode` are independent. A name selects only the threshold and
never a mode, so `clip="seurat", clip_mode="upper"` means upper-only clipping
at \(\sqrt{n_\mathrm{obs}/30}\) — it does not adopt SCTransform's symmetric
clipping. The reverse holds too: `clip_mode` means the same thing for named and
numeric thresholds alike.

### Reusing a specification across datasets

A named threshold is symbolic in a `Residual` specification and numeric in a
fit. Reusing one `Residual(clip="seurat")` on two datasets resolves two
different thresholds, one per dataset's \(n_\mathrm{obs}\). The fitted
transform, the PCA result, and any retained operator carry the value frozen at
their own fit, recorded as `params["clip_threshold"]` alongside the request in
`params["clip"]`.

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

Because the defaults clip symmetrically, both the growth and the guard are
reachable on a call that never mentions clipping. This is deliberate: extreme
residuals need control, and the alternative — pairing the clipped default with
`clip_mode="upper"` — would leave the large negative residuals of deeply
sequenced cells untouched.

How much support actually grows is a property of the dataset, not of the API.
On a combined four-donor PBMC matrix of 22,111 cells filtered to at least 100
detected genes per cell and three cells per gene, the Seurat threshold is about
`27.148`, and symmetric Poisson Pearson clipping changes 157,739 positive and
84 negative residuals while adding no structural-zero entries at all. A smaller
or shallower dataset has a smaller threshold and can add many. Do not treat the
absence of growth as an API guarantee: if a run raises on the guard, raise
`clip_max_nnz_ratio`, switch to `clip_mode="upper"`, or set `clip=None`.

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
