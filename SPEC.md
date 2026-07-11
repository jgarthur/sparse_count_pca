# Sparse-plus-low-rank spectral analysis package spec

## Scope

The package computes spectral analyses of implicitly transformed sparse count
data without materializing dense transformed matrices.

The package is AnnData-first. The primary public API operates on `AnnData` and writes Scanpy-compatible PCA outputs. A lower-level sparse matrix API is provided for testing and non-AnnData use.

This version supports:

* Poisson Pearson residuals
* Poisson deviance residuals
* Binomial Pearson residuals
* Binomial deviance residuals
* size-factor-scaled negative-binomial Pearson residuals
* size-factor-scaled negative-binomial deviance residuals
* shifted CLR PCA
* classical correspondence analysis with principal coordinates

The package does **not** expose the usual standard negative-binomial residuals because their zero-count terms do not factorize into sparse-plus-rank-one form.

## Public API

```python
import sparse_residual_pca as srp

srp.residual_pca(adata, ...)
srp.residual_pca_matrix(X, ...)
srp.shifted_clr_pca(adata, ...)
srp.shifted_clr_pca_matrix(X, ...)
srp.correspondence_analysis(adata, ...)
srp.correspondence_analysis_matrix(X, ...)
srp.ResidualPCAResult
srp.CorrespondenceAnalysisResult
```

No public `pp` namespace for v1. The package should feel like a small tool rather than a full Scanpy replacement.

## Package layout

```text
src/
  sparse_residual_pca/
    __init__.py
    _anndata.py
    _matrix.py
    _operator.py
    _representation.py
    _residuals.py
    _shifted_clr.py
    _correspondence.py
    _clip.py
    _svd.py
    py.typed
tests/
  __init__.py
  conftest.py
  _oracles.py
  townes_reference/
    README.md
    generate_reference.R
    test_townes_reference.py
  shifted_clr_reference/
    README.md
    oracle.py
    LICENSE
  corral_reference/
    README.md
    generate_reference.R
    test_corral_reference.py
```

All package modules other than `__init__.py` are private. `tests/_oracles.py`
holds dense oracle helpers used by the test suite (`_materialize_dense_residual`,
`_dense_poisson_deviance`, `_dense_binomial_deviance`,
`_dense_scaled_nb_deviance`, `_compare_subspaces`) so test-only code is not
installed in the runtime package.

`__init__.py` exposes the residual PCA, shifted CLR PCA, and correspondence-analysis
functions and their result types.

```python
from ._anndata import residual_pca
from ._matrix import ResidualPCAResult, residual_pca_matrix

__all__ = ["ResidualPCAResult", "residual_pca", "residual_pca_matrix"]
```

The `src/` layout prevents repository-root imports from succeeding unless the
package has been installed into the active environment.

### Distribution contents

The wheel contains only `src/sparse_residual_pca`.

The source distribution contains package sources, ordinary tests, the
shifted CLR reference oracle and its license, `README.md`, `SPEC.md`, and
`pyproject.toml`. It excludes editor configuration, lock files, local
development scripts, and `tests/townes_reference`. The Townes reference files
are repository-only provenance and regression tooling.

## Dependencies

Runtime requirements:

```text
python >= 3.10
numpy
scipy
anndata
scikit-learn        # for sklearn.utils.extmath.svd_flip
```

The core package does not require Scanpy.

Testing extras:

```text
scanpy >= 1.10      # provides the modern PCA API with mask_var
pytest
ruff
```

Scanpy 1.10 is required for the Scanpy comparison tests because the modern
PCA API uses `mask_var` (with `use_highly_variable` deprecated in favor of
it), which matches the `mask_var` semantics this package implements.

For SciPy, any version with `scipy.sparse.linalg.svds(..., solver='arpack',
v0=v0)` and `LinearOperator` support is sufficient.

## Main AnnData function

```python
def residual_pca(
    adata,
    n_comps=50,
    *,
    layer=None,
    use_raw=False,
    mask_var=_empty,
    use_highly_variable=None,
    key_added=None,
    model="poisson",
    residual="pearson",
    alpha=None,
    clip=None,
    clip_mode="symmetric",
    clip_max_nnz_ratio=2.0,
    check_values=True,
    dtype="float64",
    solver="arpack",
    random_state=0,
    tol=0.0,
    copy=False,
):
    """Compute PCA of implicitly represented residuals and write Scanpy-style outputs."""
```

## Sparse matrix function

```python
def residual_pca_matrix(
    X,
    n_comps=50,
    *,
    model="poisson",
    residual="pearson",
    alpha=None,
    clip=None,
    clip_mode="symmetric",
    clip_max_nnz_ratio=2.0,
    check_values=True,
    dtype="float64",
    solver="arpack",
    random_state=0,
    tol=0.0,
    return_operator=False,
):
    """Compute PCA of implicitly represented residuals from a sparse count matrix."""
```

Return type:

```python
@dataclass
class ResidualPCAResult:
    scores: np.ndarray
    components: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    total_variance: float
    params: dict
    operator: object | None = None
```

Where:

```python
scores.shape == (n_obs, n_comps)
components.shape == (n_comps, n_vars_used)
loadings.shape == (n_vars_used, n_comps)
```

## Parameter reference

### `model`

Allowed values:

```python
"poisson"
"binomial"
"scaled_nb"
```

### `residual`

Allowed values:

```python
"pearson"
"deviance"
```

Only the `(model, residual)` combinations listed in `SUPPORTED` below are
valid.

### `alpha`

Required only for `model="scaled_nb"`; invalid for all other models.

For `residual_pca`:

```python
alpha: float          # broadcast scalar to all genes
alpha: str            # read per-gene values from adata.var[alpha], then apply mask_var
alpha: array-like     # shape (adata.n_vars,) before masking; mask_var is then applied
```

The AnnData API never accepts an already-masked `alpha` array, to keep `.var`
alignment unambiguous.

For `residual_pca_matrix`:

```python
alpha: float          # broadcast scalar
alpha: array-like     # shape (X.shape[1],)
```

Validation:

```python
if model == "scaled_nb" and alpha is None:
    raise ValueError("alpha is required for model='scaled_nb'")

if model != "scaled_nb" and alpha is not None:
    raise ValueError("alpha is only used for model='scaled_nb'")

if np.any(np.asarray(alpha) < 0):
    raise ValueError("alpha must be nonnegative")
```

If `alpha_j` is below `alpha_eps = 1e-8`, select the Poisson limit once for
that gene and use it for Pearson and deviance residuals at every cell. For a
positive `alpha_j` above the threshold, retain the scaled-NB family even when
the cell-adjusted `alpha_tilde_ij = alpha_j / s_i` is below the threshold.

### `clip`

For v1, accept only:

```python
clip=None       # no clipping
clip=float      # finite positive clipping threshold
```

Validation:

```python
if isinstance(clip, str):
    raise NotImplementedError(
        "String clip aliases are not supported in v1; "
        "pass a finite positive float."
    )
if clip is not None and (not np.isfinite(clip) or clip <= 0):
    raise ValueError("clip must be finite and positive or None")
```

Future versions may add string aliases such as `"sqrt_n_obs"` or
`"sqrt_n_obs_over_30"`, but v1 keeps the default as no clipping rather than
introduce a hidden Scanpy- or Seurat-style default.

### `clip_mode`

Allowed values:

```python
"symmetric"  # clip both tails to [-clip, clip]
"upper"      # clip only the upper tail to clip
```

The default is `"symmetric"`. `clip_mode` is validated even when `clip=None`.
See the Clipping section for mathematical semantics.

Validation:

```python
if clip_mode not in {"symmetric", "upper"}:
    raise ValueError("clip_mode must be one of: symmetric, upper")
```

### `clip_max_nnz_ratio`

`clip_max_nnz_ratio` is a resource guard for exact symmetric clipping:

```python
clip_max_nnz_ratio=2.0  # default
clip_max_nnz_ratio=1.0  # reject any sparse support growth
clip_max_nnz_ratio=None # allow unlimited exact support growth
```

A finite value must be at least `1.0`. The guard is inactive for upper
clipping because upper clipping does not alter zero-count residuals for any
supported model.

Validation:

```python
if clip_max_nnz_ratio is not None and (
    not np.isfinite(clip_max_nnz_ratio) or clip_max_nnz_ratio < 1
):
    raise ValueError(
        "clip_max_nnz_ratio must be finite and at least 1 or None"
    )
```

### `check_values`

`check_values=True` adds an integer-likeness check on nonzero entries when
`X.dtype` is a floating type:

```python
if np.issubdtype(X.dtype, np.floating):
    assert np.allclose(X.data, np.rint(X.data), rtol=0, atol=1e-8)
```

For integer dtypes this check is skipped because it is trivially satisfied.

`check_values=False` skips the integer-likeness check entirely, but NaN, inf,
and negative values are always rejected because the residual math is
undefined for them.

Reject complex and nonnumeric input before any cast. Integer and wider
floating inputs must round-trip exactly through `float64`; this prevents raw
count information from being silently discarded.

### `dtype`

Representation and operator dtype, default `"float64"`. Only `float32` and
`float64` are accepted. Explicit `float32` is a lower-memory approximate mode.

- Validation and computation of `n_i`, `p_j`, `mean`, total variance, and
  Frobenius norms are performed in `float64`.
- `S.data`, `u`, `v`, `mean` are cast to the requested `dtype` for storage
  and operator use.
- `_matvec`, `_rmatvec`, `_matmat`, `_rmatmat` outputs use the operator
  dtype.
- `singular_values`, `explained_variance`, `explained_variance_ratio`,
  `total_variance` are returned as `float64`.

`dtype` controls representation and operator output, not accumulation.

### `solver`

Only `"arpack"` is supported in v1. Any other value raises:

```python
raise NotImplementedError("Only solver='arpack' is supported in v1")
```

### `random_state` and `v0`

`random_state` seeds the ARPACK starting vector:

```python
rng = np.random.default_rng(random_state)
v0 = rng.standard_normal(min(A.shape))
```

### `tol`

Passed through to `scipy.sparse.linalg.svds(..., tol=tol)`.

### `n_comps`

Must satisfy:

```python
if not 1 <= n_comps < min(n_obs, n_vars_used):
    raise ValueError(
        "n_comps must satisfy 1 <= n_comps < min(n_obs, n_vars_used) "
        "when solver='arpack'"
    )
```

`n_vars_used` is `int(mask.sum())` after `mask_var` resolution. Do not
silently clamp.

### `copy`

```python
copy=False     # mutate adata in place, return None
copy=True      # operate on a copy, return the modified copy
```

```python
if copy:
    adata = adata.to_memory() if adata.isbacked else adata.copy()
    # ... write outputs ...
    return adata
else:
    # ... write outputs ...
    return None
```

For backed AnnData inputs, `copy=True` returns an in-memory object because
`AnnData.copy()` requires a destination filename in backed mode.

### `return_operator`

`residual_pca_matrix(..., return_operator=True)` populates `result.operator`
with the exact `ResidualLinearOperator` passed to ARPACK, post-clipping and
centered. It is intended for tests and custom downstream SVDs. The AnnData
API does not expose `return_operator` and never writes the operator into
`adata`.

### Centering

PCA is performed on column-centered residuals. `center=True` is fixed in the
public API. The internal operator carries a `center` flag for testing dense
operator products against an uncentered baseline, but users cannot set it.

## Count matrix selection

Input matrix selection follows this priority:

```python
if use_raw and layer is not None:
    raise ValueError("Specify only one of use_raw=True or layer=...")

if use_raw:
    if adata.raw is None:
        raise ValueError("use_raw=True, but adata.raw is None")

    # Use raw counts, but preserve the current adata.var_names feature space.
    # adata.raw may have more genes than adata because AnnData.raw is not sliced
    # along variables when adata is sliced.
    missing = adata.var_names.difference(adata.raw.var_names)
    if len(missing) > 0:
        raise ValueError(
            "use_raw=True requires all current adata.var_names to be present "
            "in adata.raw.var_names"
        )

    X = adata.raw[:, adata.var_names].X
    var = adata.var
    var_names = adata.var_names
else:
    X = adata.layers[layer] if layer is not None else adata.X
    var = adata.var
    var_names = adata.var_names
```

This means PCA loadings always align with `adata.var_names`, so `.varm[...]` can be written normally even when `use_raw=True`.

Recommended user behavior:

```python
srp.residual_pca(adata, layer="counts", ...)
```

`use_raw=True` is supported for Scanpy compatibility, but `layer="counts"` is preferred because `.raw` is not guaranteed to contain raw integer counts.

## Input validation

The selected count matrix is canonicalized and validated before any residual
math runs.

Canonicalization:

```python
if isinstance(X, (anndata.abc.CSRDataset, anndata.abc.CSCDataset)):
    X = X.to_memory()

if scipy.sparse.issparse(X):
    X = X.tocsr()
else:
    warnings.warn(
        "Dense input was converted to CSR. This may require substantial memory.",
        UserWarning,
    )
    X = scipy.sparse.csr_matrix(X)
```

CSR, CSC, COO, and other sparse formats are all accepted; internally they
are converted to CSR. AnnData-backed sparse datasets are loaded into memory
through their `to_memory()` method. Dense array-like inputs, including Zarr
arrays, are eagerly converted to CSR. Integer or float dtype is accepted, but
the data must represent counts. Dask arrays are not supported as lazy inputs.

Always-on checks (regardless of `check_values`):

```python
if not np.isfinite(X.data).all():
    raise ValueError("Input contains NaN or inf")
if (X.data < 0).any():
    raise ValueError("Input contains negative values")
```

If `check_values=True` and `X.dtype` is a floating type, additionally require
count-likeness on stored entries:

```python
if np.issubdtype(X.dtype, np.floating) and not np.allclose(
    X.data, np.round(X.data)
):
    raise ValueError(
        "Input contains non-integer values; pass check_values=False to skip"
    )
```

For integer dtypes the check is skipped because it is trivially satisfied.

Edge cases on the chosen count matrix:

```python
if (n == 0).any():
    raise ValueError("Cells with zero total counts are not supported")

if (p_j[mask] == 0).any():
    raise ValueError("Selected genes with zero total counts are not supported")
```

For `model="binomial"`:

```python
if (p_j[mask] >= 1).any():
    raise ValueError("Binomial residuals require 0 < p_j < 1")
```

For `model="scaled_nb"`:

```python
if np.any(np.asarray(alpha) < 0):
    raise ValueError("alpha must be nonnegative")
```

`mask_var` is the user's lever for excluding problematic genes; in v1 the
package does not silently drop them.

## Parameter estimation

Cell totals and gene proportions are estimated from the chosen count matrix
**before** applying `mask_var`:

```math
n_i = \sum_{j=1}^{G} X_{ij},
\qquad
M = \sum_i n_i,
\qquad
p_j = \frac{\sum_i X_{ij}}{M}.
```

If `mask_var` selects a subset of genes for PCA, residuals are computed only
for those genes, but `n_i` and `M` are based on the full chosen count
matrix. Thus selected-gene `p_j` values do not need to sum to one: the null
model is based on total RNA per cell, not on RNA among selected genes.

For binomial residuals, `n_i` is also the binomial trial count for cell `i`:

```math
X_{ij} \sim \mathrm{Binomial}(n_i, p_j).
```

There is no separate `n_trials` argument in v1.

For `model="scaled_nb"`, the per-gene overdispersion `alpha_j` is supplied
by the user; it is not estimated in v1. See the `alpha` entry under
Parameter reference for accepted forms.

For `residual_pca_matrix(X, ...)`, the entire `X` is treated as the
universe: `n_i`, `M`, and `p_j` are computed from `X` directly.

## Variable masking

Match Scanpy PCA behavior.

`_empty` is a private module-level sentinel used only as a default value to
distinguish "user did not pass `mask_var`" (apply Scanpy's default behavior)
from "user passed `mask_var=None`" (force all genes):

```python
_empty = object()
```

It is not exported and users should never pass it.

```python
def _resolve_mask_var(var, mask_var=_empty, use_highly_variable=None):
    if use_highly_variable is not None:
        warnings.warn(
            "use_highly_variable is deprecated; use mask_var instead. "
            "use_highly_variable=True is equivalent to mask_var='highly_variable'; "
            "use_highly_variable=False is equivalent to mask_var=None.",
            FutureWarning,
        )
        if mask_var is not _empty:
            raise ValueError("Cannot specify both mask_var and use_highly_variable")

        if use_highly_variable:
            mask_var = "highly_variable"
        else:
            mask_var = None

    if mask_var is _empty:
        if "highly_variable" in var:
            mask = require_boolean_mask(var["highly_variable"])
        else:
            mask = np.ones(var.shape[0], dtype=bool)

    elif mask_var is None:
        mask = np.ones(var.shape[0], dtype=bool)

    elif isinstance(mask_var, str):
        if mask_var not in var:
            raise KeyError(f"{mask_var!r} not found in adata.var")
        mask = require_boolean_mask(var[mask_var])

    else:
        mask = require_boolean_mask(mask_var)

    if mask.ndim != 1 or mask.shape[0] != var.shape[0]:
        raise ValueError("mask_var must be a boolean vector with length n_vars")

    if mask.sum() == 0:
        raise ValueError("mask_var selected zero genes")

    return mask
```

Important behavior:

* `mask_var=_empty`: use `.var['highly_variable']` if present; otherwise use all genes.
* `mask_var=None`: use all genes, even if `.var['highly_variable']` exists.
* `use_highly_variable=True`: deprecated alias for `mask_var='highly_variable'`.
* `use_highly_variable=False`: deprecated alias for `mask_var=None`.

No HVG selection is implemented in v1.

## Output keys

Match Scanpy PCA output conventions.

If `key_added is None`:

```python
obsm_key = "X_pca"
varm_key = "PCs"
uns_key = "pca"
```

If `key_added="resid_pca"`:

```python
obsm_key = varm_key = uns_key = "resid_pca"
```

Write:

```python
adata.obsm[obsm_key] = scores
adata.varm[varm_key] = loadings_full
adata.uns[uns_key] = {
    "variance": explained_variance,
    "variance_ratio": explained_variance_ratio,
    "singular_values": singular_values,
    "params": {
        "model": model,
        "residual": residual,
        "alpha": _serialize_alpha(alpha),
        "clip": clip,
        "clip_mode": clip_mode,
        "clip_max_nnz_ratio": clip_max_nnz_ratio,
        "zero_center": True,
        "layer": layer,
        "use_raw": use_raw,
        "mask_var": resolved_scanpy_mask_selector,
        "use_highly_variable": resolved_scanpy_hvg_flag,
        "mask_var_details": _serialize_mask_var(mask_var, mask),
        "solver": solver,
        "n_comps": n_comps,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": resolved_dtype,
        "package_version": __version__,
    },
}
```

`mask_var` and `use_highly_variable` match Scanpy's standard names and
meaning. `mask_var_details` additionally stores a compact description of the
resolved selection:

```python
params["mask_var_details"] = {
    "kind": "default" | "none" | "var_key" | "array",
    "key": "highly_variable" | <user-supplied key> | None,
    "n_vars_used": int(mask.sum()),
}
```

Examples:

```python
mask_var=_empty   # and adata.var has "highly_variable"
# -> {"kind": "default", "key": "highly_variable", "n_vars_used": ...}

mask_var=_empty   # and adata.var has no "highly_variable"
# -> {"kind": "default", "key": None, "n_vars_used": adata.n_vars}

mask_var=None
# -> {"kind": "none", "key": None, "n_vars_used": adata.n_vars}

mask_var="my_mask"
# -> {"kind": "var_key", "key": "my_mask", "n_vars_used": ...}

mask_var=<bool array>
# -> {"kind": "array", "key": None, "n_vars_used": ...}
```

For `alpha`, the serializer reduces it to a form safe for `.uns`:

```python
def _serialize_alpha(alpha):
    if alpha is None or isinstance(alpha, (int, float, str)):
        return alpha
    return np.asarray(alpha)  # array values are fine; arbitrary objects are not
```

If a variable mask is used, `loadings_full` has shape `(adata.n_vars, n_comps)`. Genes outside the mask receive `np.nan` to make non-used genes explicit (preferred over zeros, which can be mistaken for valid loadings).

## Sparse-plus-low-rank representation

The internal representation is:

```math
M = S + UV^\top
```

where:

* `S` is sparse
* `U` has shape `(n_obs, rank)`
* `V` has shape `(n_vars_used, rank)`
* rank zero is permitted

Residual transforms and shifted CLR currently produce rank-one representations.
The generic form also supports future transforms with multiple implicit
components. Row scaling, column scaling, and column selection preserve the
sparse-plus-low-rank form.

After clipping:

```math
R^{(t)} = S^{(t)} + uv^\top
```

After centering:

```math
A = R_c^{(t)}
  = S^{(t)} + uv^\top - \mathbf 1 \bar r^\top
```

where

```math
\bar r
=
\frac{1}{N}(S^{(t)})^\top \mathbf 1
+
\bar u v,
\qquad
\bar u = \frac{1}{N}\mathbf 1^\top u.
```

## Operator class

Use one class, not separate sparse-plus-rank-one and centered wrappers.

```python
class SparseLowRankLinearOperator(scipy.sparse.linalg.LinearOperator):
    """
    Represents

        A = S + U V^T - 1 mean^T

    if center=True, and

        A = S + U V^T

    if center=False.

    ResidualLinearOperator remains a compatibility wrapper for rank one.
    """
```

Fields and shapes (with `N = n_obs`, `G = n_vars_used`):

```python
S: scipy.sparse.csr_matrix     # shape (N, G)
left: np.ndarray               # shape (N, rank)
right: np.ndarray              # shape (G, rank)
mean: np.ndarray | None        # shape (G,) when center=True; None when center=False
center: bool
dtype: np.dtype
```

Dtype rules:

- `S.data`, `left`, `right`, `mean` are stored in the operator `dtype` (default
  `float64`).
- `_matvec(z)` and `_rmatvec(y)` return arrays of the operator `dtype`.
- Computations that build the representation, mean, and Frobenius norms are
  performed in `float64`; the cast to `dtype` happens at storage time.

Methods:

```python
_matvec(z)
_rmatvec(y)
_matmat(Z)
_rmatmat(Y)
mean_axis0()
frobenius_squared_uncentered()
frobenius_squared_centered()
```

Matrix-vector products:

```math
Az
=
S z
+
U(V^\top z)
-
\mathbf 1(\bar r^\top z)
```

```math
A^\top y
=
S^\top y
+
V(U^\top y)
-
\bar r(\mathbf 1^\top y)
```

If `center=False`, omit the mean terms.

## Supported residuals

Do not expose `"nb"` as a model. Only expose models whose zero-count residuals factorize.

Supported combinations:

```python
SUPPORTED = {
    ("poisson", "pearson"),
    ("poisson", "deviance"),
    ("binomial", "pearson"),
    ("binomial", "deviance"),
    ("scaled_nb", "pearson"),
    ("scaled_nb", "deviance"),
}
```

### Pearson residuals

For Pearson residuals:

```math
R_{ij}
=
\frac{X_{ij}-\mu_{ij}}{\sqrt{V_{ij}}},
\qquad
\mu_{ij}=n_i p_j.
```

Sparse entries on the support of `X` are:

```math
S_{ij}
=
\frac{X_{ij}}{\sqrt{V_{ij}}}
\qquad X_{ij}>0.
```

Low-rank zero-count term:

```math
L_{ij}=u_i v_j.
```

Implementation table:

```text
model: poisson
V_ij = n_i p_j
u_i = -sqrt(n_i)
v_j = sqrt(p_j)

model: binomial
V_ij = n_i p_j (1 - p_j)
u_i = -sqrt(n_i)
v_j = sqrt(p_j / (1 - p_j))

model: scaled_nb
s_i = n_i / mean(n)
V_ij = n_i p_j (1 + alpha_j mean(n) p_j)
u_i = -sqrt(n_i)
v_j = sqrt(p_j) / sqrt(1 + alpha_j mean(n) p_j)
```

### Deviance residuals

For deviance residuals:

```math
R_{ij}
=
\operatorname{sign}(X_{ij}-\mu_{ij})
\sqrt{d_j(X_{ij},\mu_{ij})}.
```

Zero-count term:

```math
L_{ij}
=
-\sqrt{d_j(0,\mu_{ij})}
=
u_i v_j.
```

Sparse entries on the support of `X` are:

```math
S_{ij}
=
R_{ij} - u_i v_j
\qquad X_{ij}>0.
```

Implementation table:

```text
model: poisson
d(0, mu_ij) = 2 n_i p_j
u_i = -sqrt(n_i)
v_j = sqrt(2 p_j)

model: binomial
d(0, mu_ij) = 2 n_i log(1 / (1 - p_j))
u_i = -sqrt(n_i)
v_j = sqrt(2 log(1 / (1 - p_j)))

model: scaled_nb
s_i = n_i / mean(n)
alpha_tilde_ij = alpha_j / s_i
d(0, mu_ij) = 2 n_i / (alpha_j mean(n)) * log(1 + alpha_j mean(n) p_j)
u_i = -sqrt(n_i)
v_j = sqrt(2 log(1 + alpha_j mean(n) p_j) / (alpha_j mean(n)))
```

For scaled NB deviance, if `alpha_j` is numerically zero, use the Poisson deviance limit.

## Numerical implementation

Production code should not use `scipy.stats.logpmf` for residual computation. Use direct formulas with stable primitives.

Recommended primitives:

```python
from scipy.special import xlogy
```

At zero-count boundaries, use:

```python
xlogy(x, x / mu)
```

instead of:

```python
x * np.log(x / mu)
```

because `xlogy(0, y)` returns zero.

Use:

```python
np.log1p(z)
```

for terms like:

```python
log(1 + z)
```

Use:

```python
-np.log1p(-p)
```

for:

```python
log(1 / (1 - p))
```

Near the mean, do not subtract the two leading terms in the direct formulas.
Write the Poisson relative-entropy contribution as:

```math
h(b, \delta)=(b+\delta)\log(1+\delta/b)-\delta
```

and evaluate it with:

```math
h(b,\delta)
=b\sum_{k=2}^{\infty}
\frac{(-1)^k(\delta/b)^k}{k(k-1)}
```

for small relative differences. Poisson deviance is `2 h(mu, x-mu)`;
binomial deviance is the sum of this contribution for successes and failures.
Use the `log1p` form outside the series region and test continuity at the
switch.

For scaled NB, use the corresponding near-mean series:

```math
\frac{d_{NB}}{2}
=\mu\sum_{k=2}^{\infty}
\frac{(-1)^k q^k}{k(k-1)}
\left(1-\rho^{k-1}\right),
\quad
q=\frac{x-\mu}{\mu},
\quad
\rho=\frac{\tilde\alpha\mu}{1+\tilde\alpha\mu}.
```

This remains in the NB family for every positive dispersion and approaches
the Poisson series continuously as dispersion tends to zero.

For deviance residuals, clamp only negative values consistent with final
float64 rounding. A materially negative result raises an error rather than
being silently converted to zero:

```python
r = np.sign(x - mu) * np.sqrt(d)
```

For scaled NB:

```python
alpha_tilde = alpha_j / s_i
```

but avoid explicitly forming a dense alpha matrix. For nonzero entries, compute `alpha_tilde` only on the sparse support.

Select Poisson versus NB once per gene from `alpha_j`. If `alpha_j` is below
`alpha_eps`, use the Poisson formulas for both zero and nonzero counts. If it
is above the threshold, use the NB formulas at every cell; do not switch on
`alpha_tilde`.

Recommended threshold:

```python
alpha_eps = 1e-8
```

or expose internally.

## Clipping

Clipping is an exact elementwise transformation. The supported modes are:

```math
\phi^{\mathrm{sym}}_t(x)=\min(t,\max(-t,x))
```

and

```math
\phi^{\mathrm{upper}}_t(x)=\min(t,x).
```

For every explicitly stored count entry:

```math
S^{(t)}_{ij}
=
\phi_t(R_{ij}) - u_i v_j.
```

For zero-count entries:

```math
R_{ij}=u_i v_j.
```

For every supported residual model, `u_i < 0` and `v_j > 0`, so zero-count
residuals are negative. Consequently:

- Upper clipping leaves every zero-count residual unchanged and never expands
  sparse support.
- Symmetric clipping may clip the lower tail at zero-count locations. Those
  corrections are included exactly in `S`.

Symmetric clipping may produce many zero-count corrections. The resource guard
is:

```python
clip_max_nnz_ratio=2.0
```

If the exact clipped representation would satisfy:

```python
(X.nnz + nnz_clipped_zeros) / X.nnz >= clip_max_nnz_ratio
```

raise `RuntimeError` before constructing the expanded matrix. A value of
`1.0` rejects any support growth. `None` disables the guard and always includes
the exact corrections.

The binary search used to locate possible lower-tail corrections is only a
candidate generator. Before counting corrections or applying the growth
guard, filter candidates with the direct strict predicate:

```python
u[rows] * v[cols] < -clip
```

Equality therefore does not clip. Reuse the CSR row-support vector already
constructed during residual evaluation when applying clipping and excluding
stored count locations.

The defaults are:

```python
clip=None
clip_mode="symmetric"
clip_max_nnz_ratio=2.0
```

For v1, `clip=None` means no clipping. Later versions may add aliases:

```python
clip="sqrt_n_obs"
clip="sqrt_n_obs_over_30"
```

but v1 should avoid hidden defaults.

## Solver

Only support ARPACK in v1.

```python
solver="arpack"
```

If another solver is requested:

```python
raise NotImplementedError("Only solver='arpack' is supported in v1")
```

Validate `n_comps` before calling ARPACK:

```python
if not 1 <= n_comps < min(n_obs, n_vars_used):
    raise ValueError(
        "n_comps must satisfy 1 <= n_comps < min(n_obs, n_vars_used) "
        "when solver='arpack'"
    )
```

Construct a deterministic starting vector from `random_state`:

```python
rng = np.random.default_rng(random_state)
v0 = rng.standard_normal(min(A.shape))
```

Then call:

```python
scipy.sparse.linalg.svds(A, k=n_comps, solver="arpack", tol=tol, v0=v0)
```

`svds` does not guarantee sorted singular values; sort descending afterwards.

## Sign flipping

Match sklearn PCA behavior after ARPACK.

```python
U, s, Vt = svds(A, k=n_comps, tol=tol, v0=v0)

order = np.argsort(s)[::-1]
s = s[order]
U = U[:, order]
Vt = Vt[order, :]

U, Vt = sklearn.utils.extmath.svd_flip(
    U,
    Vt,
    u_based_decision=False,
)
```

Then:

```python
scores = U * s
components = Vt
loadings = Vt.T
```

## Explained variance and total variance

For:

```math
R = S + uv^\top,
```

calculate column means and squared norms from the actual represented values:

```math
R_{ij} =
\begin{cases}
S_{ij} + u_i v_j & (i,j) \in \operatorname{supp}(S),\\
u_i v_j & \text{otherwise}.
\end{cases}
```

Do not calculate large sparse and low-rank component norms and then subtract
them. For sparse-support columns, accumulate the implicit baseline using
stable moments of `u` and replace values on stored support. For columns with
more than half their entries stored, evaluate the full column directly; this
avoids severe subtraction for dense and nearly dense support while retaining
`O(nnz + n_obs + n_vars)` complexity.

The centered squared norm is accumulated directly about the stored operator
mean:

```math
\|R_c\|_F^2 = \sum_{ij}(R_{ij}-\bar r_j)^2.
```

Statistics describe the post-cast `S`, `u`, `v`, and mean used by the operator
passed to ARPACK. Before invoking ARPACK, treat the centered matrix as
numerically zero when:

```python
frobenius_centered_squared <= (
    np.finfo(operator_dtype).eps**2
    * n_obs
    * n_vars
    * frobenius_uncentered_squared
)
```

Raise `ValueError` in that case because PCA directions are undefined.

Then:

```python
total_variance = frobenius_centered_squared / (n_obs - 1)
explained_variance = singular_values**2 / (n_obs - 1)
explained_variance_ratio = explained_variance / total_variance
```

## Testing strategy

### 1. Dense oracle tests

For small sparse count matrices, compute dense residual matrices directly and compare operator products.

Test all supported combinations:

```text
poisson pearson
poisson deviance
binomial pearson
binomial deviance
scaled_nb pearson
scaled_nb deviance
```

Compare:

```python
A @ z
A.T @ y
A @ Z
A.T @ Y
```

against dense matrix multiplication.

Include tests with:

```text
clip=None
clip=small value with clip_mode="upper"
clip=small value with clip_mode="symmetric"
center=True
center=False
```

### 2. SVD oracle tests

For small matrices:

```python
np.linalg.svd(R_centered, full_matrices=False)
```

Compare singular values and right singular subspace against package output.

Do not compare raw signs directly. Compare after sign flipping or compare projection matrices.

### 3. Scanpy comparison tests

For Poisson Pearson residuals, compare against Scanpy experimental Pearson residual normalization with infinite overdispersion.

Test residual matrix values on small data.

Then test PCA by running Scanpy PCA on the explicit residual matrix and comparing singular values/subspaces to the implicit package result.

### 4. Deviance oracle tests using scipy.stats

Use scipy distribution log-likelihoods as test oracles for deviance, not for production code.

General formula:

```python
deviance = 2 * (loglik_saturated - loglik_null)
residual = np.sign(x - mu) * np.sqrt(np.maximum(deviance, 0))
```

Poisson:

```python
loglik_null = scipy.stats.poisson.logpmf(x, mu)
loglik_sat = scipy.stats.poisson.logpmf(x, x)
```

Handle `x=0` carefully; saturated Poisson mean is zero.

Binomial:

```python
loglik_null = scipy.stats.binom.logpmf(x, n_i, p_j)
p_hat = x / n_i
loglik_sat = scipy.stats.binom.logpmf(x, n_i, p_hat)
```

Handle boundary cases `p_hat=0` and `p_hat=1`.

Negative binomial with variance `mu + alpha * mu**2`:

```python
r = 1 / alpha
prob = r / (r + mu)
loglik_null = scipy.stats.nbinom.logpmf(x, r, prob)
```

For saturated NB:

```python
prob_sat = r / (r + x)
loglik_sat = scipy.stats.nbinom.logpmf(x, r, prob_sat)
```

For scaled NB:

```python
alpha_tilde = alpha_j / s_i
r = 1 / alpha_tilde
prob = r / (r + mu)
```

Boundary cases should be handled explicitly in tests to avoid treating SciPy boundary behavior as the definition.

### 5. AnnData tests

Test:

```text
layer=None
layer="counts"
use_raw=True with raw containing all current var_names
use_raw=True with raw missing at least one current var_name
mask_var=_empty with highly_variable present
mask_var=_empty without highly_variable present
mask_var=None
mask_var="highly_variable"
use_highly_variable=True
use_highly_variable=False
key_added=None
key_added="resid_pca"
copy=True
copy=False
```

For `use_raw=True`, verify that:

```python
X_used == adata.raw[:, adata.var_names].X
```

and that loadings are written to `adata.varm[...]` aligned to current `adata.var_names`.

### 6. Clipping tests

Construct matrices where symmetric zero-count clipping definitely occurs and
where both residual tails cross the threshold.

Test:

```text
upper clipping matches the dense exact oracle
upper clipping leaves the negative tail unchanged
upper clipping does not expand sparse support
symmetric clipping matches the dense exact oracle
clip_max_nnz_ratio=None allows exact support growth
clip_max_nnz_ratio=1.0 rejects any support growth
finite guard includes below the threshold and raises at the threshold
```

### 7. Variance tests

For small matrices, compare analytic total variance to dense calculation:

```python
np.sum((R - R.mean(axis=0)) ** 2) / (n_obs - 1)
```

Also compare explained variance ratio to dense SVD.

### 8. Townes reference test

`tests/townes_reference/test_townes_reference.py` stores singular values
generated from `null_residuals()` in the Townes `scrna2019` repository and
compares them with `residual_pca_matrix`.

Normal pytest runs must not require R, network access, or a local reference
checkout. `tests/townes_reference/generate_reference.R` is a manual
regeneration script that requires the path to the upstream `functions.R` as
its sole argument. The adjacent README records the upstream repository,
commit, source file, and regeneration command.

## V1 implementation order

1. `residual_pca_matrix` with Poisson Pearson, no clipping.
2. `ResidualLinearOperator`, centering, ARPACK SVD, sign flipping.
3. AnnData wrapper `residual_pca`.
4. Analytic total variance and explained variance ratio.
5. Exact upper and symmetric clipping with a support-growth guard.
6. Poisson deviance.
7. Binomial Pearson/deviance.
8. Scaled-NB Pearson/deviance.
9. Scanpy comparison tests.
10. Package docs and examples.
