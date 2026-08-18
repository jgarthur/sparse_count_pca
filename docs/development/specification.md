# sparse-count-pca package specification

> **Audience:** maintainers, reviewers, and coding agents. This document is
> normative. User-facing explanations are in the guides and concept pages.

## Scope

The package computes PCA and correspondence analysis of implicitly transformed
sparse count data without materializing dense transformed matrices.

The package is AnnData-first. The one-step public API operates on `AnnData` and
writes Scanpy-compatible PCA outputs. A public two-step transform API supports
inspection and repeated PCA, and a lower-level sparse matrix API supports
non-AnnData use.

This version supports:

* Poisson Pearson residuals
* Poisson deviance residuals
* Binomial Pearson residuals
* Binomial deviance residuals
* size-factor-scaled negative-binomial Pearson residuals
* size-factor-scaled negative-binomial deviance residuals
* size-factor-normalized log1p PCA
* count-scale shifted-CLR PCA
* composition-scale shifted-CLR PCA
* Dirichlet-log PCA
* Dirichlet-CLR PCA
* classical correspondence analysis with principal coordinates
* experimental scaled-NB Pearson-residual correspondence-like ordination

The package does **not** expose the usual standard negative-binomial residuals because their zero-count terms do not factorize into sparse-plus-rank-one form.

PCA is the primary analysis contract. Correspondence analysis remains as a
closely related spectral decomposition with its own result type. That result
stores principal coordinates only; standard coordinates are derived by
dividing principal coordinates by singular values when needed.

### Correspondence and experimental residual ordination

For grand total `N`, row totals `n_i`, column proportions `p_j`, and expected
counts `mu_ij = n_i p_j`, classical correspondence analysis decomposes the
Poisson Pearson residual matrix with an additional total scaling:

```math
Z_{ij}^{CA}
= \frac{1}{\sqrt N}\frac{X_{ij}-\mu_{ij}}{\sqrt{\mu_{ij}}}.
```

This identity is implemented through the shared stable Pearson-residual
representation. The correspondence path uses `center=False`, recomputes
margins after `mask_var`, and derives principal coordinates with the observed
row and column masses.

The size-factor-scaled negative-binomial model listed above uses per-gene
overdispersion with cell-depth scaling; its full residual definition appears
under [Supported residuals](#supported-residuals). In correspondence analysis,
`model="scaled_nb"` is an explicitly experimental extension. It replaces the
Poisson variance by the package's size-factor-scaled negative-binomial
variance and decomposes

```math
Z_{ij}^{NB}
= \frac{1}{\sqrt N}
  \frac{X_{ij}-\mu_{ij}}
       {\sqrt{\mu_{ij}(1 + \alpha_j \bar n p_j)}}.
```

Values of `alpha_j < 1e-8` use the Poisson limit, matching residual PCA. The
mode requires `alpha`, emits a `UserWarning`, and records
`experimental=True`. Its `total_inertia` is the squared Frobenius norm of
`Z_NB`, not Pearson chi-square divided by `N`. Observed count masses still
scale the reported coordinates, but the full chi-square and barycentric
interpretations of classical CA are not claimed.

## Public API

```python
import sparse_count_pca as scp

scp.residual_pca(adata, ...)
scp.residual_pca_matrix(X, ...)
scp.log1p_norm_pca(adata, ...)
scp.log1p_norm_pca_matrix(X, ...)
scp.shifted_clr_pca(adata, count_shift=...)
scp.shifted_clr_pca_matrix(X, count_shift=...)
scp.proportion_shifted_clr_pca(adata, composition_shift=...)
scp.proportion_shifted_clr_pca_matrix(X, composition_shift=...)
scp.dirichlet_log_pca(adata, ...)
scp.dirichlet_log_pca_matrix(X, ...)
scp.dirichlet_clr_pca(adata, ...)
scp.dirichlet_clr_pca_matrix(X, ...)
scp.correspondence_analysis(adata, ...)
scp.correspondence_analysis_matrix(X, ...)
scp.transform(adata_or_X, scp.Residual(...))
scp.transform(adata_or_X, scp.Log1pNormalized(...))
scp.transform(adata_or_X, scp.ShiftedCLR(...))
scp.transform(adata_or_X, scp.ProportionShiftedCLR(...))
scp.transform(adata_or_X, scp.DirichletLog(...))
scp.transform(adata_or_X, scp.DirichletCLR(...))
scp.TransformedMatrix
scp.PCAResult
scp.CorrespondenceAnalysisResult
```

No public `pp` namespace for v1. Correspondence analysis is intentionally not
a `Transform`: applying its mask before computing margins is part of defining
the selected contingency table.

## Two-step transform API

```python
transformed = scp.transform(
    adata_or_X,
    method,
    layer=None,
    check_values=True,
    dtype="float64",
)
```

`method` is one of the immutable `Transform` specifications listed above.
String-valued per-variable parameters such as `alpha="dispersion"` require an
AnnData input and are resolved when the transform is built. A fitted transform
is tied to that input matrix; applying fitted state to another dataset is not a
supported contract.

Fitting allocates transformed sparse values and takes one owned snapshot of CSR
support even when canonical input validation can borrow the original count
matrix. This prevents later input mutation from changing the fitted transform.

`TransformedMatrix` subclasses `scipy.sparse.linalg.LinearOperator` and
represents the full uncentered transformed matrix. It is an in-process object,
is not serialized by `write_h5ad`, and must be rebuilt after restarting Python.
Its public operations are matrix multiplication, `materialize`, and `pca`.

```python
transformed.materialize(
    obs=None,
    var=None,
    out=None,
    block_size=1024,
)
```

`obs` and `var` accept positional integers, slices, one-dimensional boolean
masks, ordered integer arrays, and names when the source is AnnData. Scalar
selection retains two dimensions. `out` must have the exact selected shape, a
writable floating dtype, and may be an ndarray or memory-mapped array.
Materialization fills it in observation blocks.

The current backend stores its sparse correction as CSR. Selecting variables
across all observations emits `scipy.sparse.SparseEfficiencyWarning` because
that access may require scanning rows. Observation blocks are the preferred
access direction. Backed inputs are still materialized during canonicalization;
future out-of-core support may replace the backend without changing the public
operator and materialization contracts.

```python
transformed.pca(
    n_comps=50,
    mask_var=_empty,
    use_highly_variable=None,
    solver="arpack",
    random_state=0,
    tol=0.0,
    return_operator=False,
)
```

For AnnData-derived transforms, mask resolution matches the one-step AnnData
API. Matrix-derived transforms accept a boolean `mask_var` or `None`. The mask
selects columns only after the full transform has been defined. One transform
may therefore be reused for several PCA masks or component counts.
If `TransformedMatrix.pca(return_operator=True)` retains the mutable centered
operator, its arrays are copied to isolate them from the still-live transform.
The named one-step APIs instead transfer their private representation to a
returned operator without that additional copy.

## Log-transform semantics

Shift parameters are keyword-only, required, finite, and positive. The API
does not use a bare `pseudocount` parameter because its scale would be
ambiguous.

For the same reason, every log-family result records `shift_domain` in its
`params` metadata, naming the scale on which the shift is fixed:

| Transform | `shift_domain` |
| --- | --- |
| `log1p_norm` | `"normalized_count"` |
| `shifted_clr` | `"count"` |
| `proportion_shifted_clr` | `"composition"` |
| `dirichlet_log`, `dirichlet_clr` | `"dirichlet_prior_counts"` |

A stored result therefore identifies its own shift scale without a reader
inferring it from the parameter name.

### Size-factor-normalized log1p

`log1p_norm` analyzes

```math
Z_{ij}
= \log\!\left(1+\frac{X_{ij}}{s_i}\right)
= \log(X_{ij}+s_i) - \log s_i,
```

where `s_i` is the per-observation size factor. Its shift is fixed at one on
the normalized-count scale, so no shift parameter is exposed; the effective
raw-count shift is `s_i` itself, recorded as
`effective_count_shift="size_factor"`.

Size factors come from exactly one of two recipes:

* `target_sum=t` sets `s_i = n_i / t` from the row total `n_i`. The default
  `target_sum=None` resolves `t` to the median row total, so the size factors
  have median one.
* `size_factors` supplies finite positive per-observation divisors, or an
  `adata.obs` key on the AnnData interfaces, used without rescaling.

An explicit `target_sum` and `size_factors` are mutually exclusive. Result
metadata records the requested `target_sum`, `resolved_target_sum` (`None`
for supplied factors), `size_factor_source` (`"median_target_sum"`,
`"target_sum"`, or `"supplied"`), and `size_factor_median`. The
representation has rank zero: zeros map to zero, and only PCA centering adds
a low-rank term.

### Count-scale shifted CLR

For `count_shift=a`, `shifted_clr` analyzes

```math
Z_{ij}
= \log(X_{ij}+a)
- \frac{1}{G}\sum_k\log(X_{ik}+a).
```

Current PFlog is this transform with `a = 1 / (4 * alpha)`, evaluated sparsely
as row-centered `log1p(4 * alpha * X)`. Although that formula is finite on a
zero-total row, the package-wide input policy rejects empty cells before any
transform is fitted.

### Composition-scale shifted CLR

For `composition_shift=tau`, `proportion_shifted_clr` analyzes

```math
Z_i = \operatorname{clr}(X_i / s_i + \tau\mathbf 1)
    = \operatorname{clr}(X_i + s_i\tau\mathbf 1),
```

where `s_i` is the row total. Its effective raw-count shift varies by cell. It
is invariant to deterministic row rescaling and rejects zero-total rows.

### Dirichlet transforms

For total concentration `A` and prior composition `p`, the prior counts are
`a_j = A p_j`. `dirichlet_log` analyzes

```math
\log\frac{X_{ij}+a_j}{s_i+A},
```

and `dirichlet_clr` analyzes `clr(X_i + a)`. Scalar count-scale shifted CLR is
the uniform-prior special case `A = G * count_shift`.

All normalization quantities and CLR row means use the full selected input
matrix. `mask_var` is applied only afterward to select PCA columns.

## Package layout

```text
src/
  sparse_count_pca/
    __init__.py
    _anndata.py
    _counts.py
    _transform.py
    _pca.py
    _residual_pca.py
    _operator.py
    _representation.py
    _residuals.py
    _log_transforms.py
    _log_pca.py
    _dirichlet_pca.py
    _correspondence.py
    _clip.py
    _svd.py
    py.typed
tests/
  __init__.py
  conftest.py
  _oracles.py
  test_dirichlet.py
  test_log_pca.py
  test_residual_anndata.py
  test_residual_pca.py
  test_residual_scanpy.py
  test_transform.py
  townes_reference/
    README.md
    generate_reference.R
    test_townes_reference.py
  shifted_clr_reference/
    README.md
    oracle.py
  proportion_shifted_clr_reference/
    README.md
    oracle.py
  corral_reference/
    README.md
    generate_reference.R
    test_corral_reference.py
  real_data_reference/
    README.md
    generate_fixture.py
    generate_sctransform_reference.R
    test_fixture.py
    test_gene_masks.py
    test_log_transform_dense_oracles.py
    test_residual_dense_oracle.py
    test_correspondence_reference.py
    test_sctransform_reference.py
```

All package modules other than `__init__.py` are private. `tests/_oracles.py`
holds shared dense oracle helpers for residual, logarithmic, Dirichlet, and
correspondence transforms, plus subspace comparison utilities, so test-only
code is not installed in the runtime package. Simulated and real-data tests use
the same formula implementations.

`__init__.py` is the authoritative list of public transforms, analysis
functions, and result types. The package-layout overview belongs to the
[architecture guide](architecture.md); it is summarized here only to define
the public/private module boundary.

The `src/` layout prevents repository-root imports from succeeding unless the
package has been installed into the active environment.

### Distribution contents

The wheel contains only `src/sparse_count_pca`.

The source distribution contains package sources, ordinary tests, committed
real-data fixtures, transform reference formulas and provenance, executable
examples, `README.md`, `CONTRIBUTING.md`, `mkdocs.yml`, the `docs/` tree, and
`pyproject.toml`. It excludes editor configuration, lock files, and local
development scripts. The Townes reference test runs offline; its adjacent R
script is only needed to regenerate the pinned reference values.

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
    mask_var=_empty,
    use_highly_variable=None,
    key_added=None,
    model="poisson",
    residual="pearson",
    alpha=None,
    clip="seurat",
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
    clip="seurat",
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
class PCAResult:
    scores: np.ndarray
    components: np.ndarray
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

`alpha` must be finite and nonnegative, with one exception: a variable that
contributes no residual cannot affect any output, so its value is replaced with
zero rather than validated. That covers variables with no counts for residual
PCA, and analyzed columns with zero mass for correspondence analysis. Array
shape is always validated, and a variable that does contribute still raises on
a non-finite or negative value.

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

Accept:

```python
clip="seurat"   # sqrt(n_obs / 30), the default
clip="scanpy"   # sqrt(n_obs)
clip=float      # finite positive clipping threshold
clip=None       # no clipping
```

Validation:

```python
if isinstance(clip, str):
    if clip not in {"seurat", "scanpy"}:
        raise ValueError(
            f"Unknown clip name {clip!r}; clip must be one of: "
            "scanpy, seurat, a finite positive float, or None"
        )
elif clip is not None and (not np.isfinite(clip) or clip <= 0):
    raise ValueError("clip must be finite and positive or None")
```

Names are matched exactly and are case-sensitive.

#### Named-threshold resolution

A named threshold is `sqrt(n_obs / divisor)`, with divisor `30` for `"seurat"`
and `1` for `"scanpy"`. `n_obs` is the number of rows of the count matrix the
transform is fitted on, taken as passed; the package never filters
observations. A named threshold that resolves to a nonpositive value —
possible only with no rows — raises `ValueError` rather than clipping at
zero.

Resolution affects only the threshold. It does not change `clip_mode` or
`clip_max_nnz_ratio`, and it is not specific to a model or residual family:
`clip="seurat"` means the same threshold for Poisson deviance residuals as for
scaled-NB Pearson residuals.

A `Residual` specification stores `clip` as passed and resolves it at each
fit, so one reused specification can produce different thresholds on different
datasets. The `TransformedMatrix`, the `PCAResult`, and any retained operator
record both values: `params["clip"]` holds the request as passed and
`params["clip_threshold"]` holds the resolved float, or `None` when clipping is
disabled.

### `clip_mode`

Allowed values:

```python
"symmetric"  # clip both tails to [-clip, clip]
"upper"      # clip only the upper tail to clip
```

The default is `"symmetric"`. `clip_mode` is validated even when `clip=None`,
and has the same meaning however the threshold was specified. See the Clipping
section for mathematical semantics.

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
supported model. It is reachable under the defaults, which clip symmetrically;
the guard depends on `clip_mode`, not on how the threshold was specified.

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
`float64` are accepted. Float64 is recommended for numerical precision.
Explicit `float32` is a lower-memory approximate mode: it changes the stored
representation and the operator passed to ARPACK, so it lowers calculation
precision rather than merely downcasting returned arrays.

- Validation and computation of `n_i`, `p_j`, `mean`, total variance, and
  Frobenius norms are performed in `float64`.
- Residual support values are evaluated in bounded `float64` blocks and written
  directly into `S.data` in the requested dtype; fitting must not construct a
  full-`nnz` float64 intermediate for each term.
- `S.data`, `u`, `v`, `mean` are cast to the requested `dtype` for storage
  and operator use.
- `_matvec`, `_rmatvec`, `_matmat`, `_rmatmat` outputs use the operator
  dtype.
- `singular_values`, `explained_variance`, `explained_variance_ratio`,
  `total_variance` are returned as `float64`.

`dtype` controls representation and operator output, not accumulation.

Users who want float64 computation with more compact persisted scores or
components may downcast those arrays after PCA. Such a post-computation cast
does not change the completed decomposition or float64 variance statistics; it
only reduces the precision of later operations on the cast arrays.

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
if not 1 <= n_comps < min(n_obs, n_nonempty_vars):
    raise ValueError(
        "n_comps must satisfy 1 <= n_comps < min(n_obs, n_nonempty_vars) "
        "when solver='arpack'"
    )
```

`n_nonempty_vars` counts the selected columns that are not identically zero, so
an empty gene cannot support a component. It is deliberately distinct from
`n_vars_used`, which everywhere else means the selected output width and
therefore still counts retained empty genes. The selected column count,
`int(mask.sum())` after `mask_var` resolution, is recorded as `pca_n_vars`. Do
not silently clamp.

This is a dimension rule about empty variables, not a rank estimate. A column
counts as nonempty when it has stored sparse support or a nonzero low-rank
factor; exact cancellation between those terms is not detected, so the count is
an upper bound on the rank.

The message reports `n_comps`, `n_obs`, and `n_nonempty_vars` as values, and
when `n_nonempty_vars` is smaller than the selected column count it also says
how many selected variables are identically zero, so the bound is not mistaken
for an off-by-one.

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
with the exact `SparseLowRankLinearOperator` passed to ARPACK, post-clipping
and centered. Matrix APIs for other PCA transforms follow the same rule. It is
intended for tests and custom downstream SVDs. The AnnData API does not expose
`return_operator` and never writes the operator into `adata`.

### Centering

PCA is performed on column-centered residuals. `center=True` is fixed in the
public API. The internal operator carries a `center` flag for testing dense
operator products against an uncentered baseline, but users cannot set it.

## Count matrix selection

Input matrix selection is:

```python
X = adata.layers[layer] if layer is not None else adata.X
var = adata.var
var_names = adata.var_names
```

## Input validation

The selected count matrix is canonicalized and validated before any residual
math runs.

Canonicalization uses copy-on-write behavior:

```python
if isinstance(X, (anndata.abc.CSRDataset, anndata.abc.CSCDataset)):
    X = X.to_memory()
    X = X if X.format == "csr" else X.tocsr()
elif scipy.sparse.issparse(X):
    if X.format != "csr":
        X = X.tocsr()
    elif not X.has_canonical_format or (X.data == 0).any():
        warnings.warn("CSR input was copied for canonicalization: ...")
        X = X.copy()
else:
    warnings.warn(
        "Dense input was converted to CSR. This may require substantial memory.",
        UserWarning,
    )
    X = scipy.sparse.csr_matrix(X)
```

Canonical zero-free CSR input is borrowed and never mutated. Caller-owned CSR
is copied, with a `UserWarning`, only when duplicate summing, index sorting, or
explicit-zero elimination is required. CSC, COO, and other sparse formats are
accepted and necessarily allocate during CSR conversion. AnnData-backed sparse
datasets are loaded into memory
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

Empty cells are rejected for every transform, during count canonicalization:

```python
if (n == 0).any():
    raise ValueError(
        "Cells with zero total counts are not supported; filter empty rows "
        "out of the count matrix first"
    )
```

Empty genes are never rejected. Every transform retains them, with a value that
follows from its own definition; see
[empty genes and cells](#empty-genes-and-cells).

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

## Empty genes and cells

### Empty genes

A gene with zero total count is **retained by every transform**; none is
dropped, so the analyzed matrix always keeps the caller's variable universe and
components stay aligned to input columns.

Its value is well defined in each family:

| Family | Value at an empty gene | Effect on other genes |
| --- | --- | --- |
| Residual (Poisson, binomial, scaled-NB) | `0`, the limit of the residual as the fitted mean goes to zero | none |
| Correspondence analysis | `0`, the limit of the standardized deviation as column mass goes to zero | none |
| Size-factor-normalized log1p | `0`, since `log1p` of zero is zero | none |
| Count-scale shifted CLR | `log(a) - m_i` | scales the log-ratio centering |
| Composition-scale shifted CLR | analogous | scales the log-ratio centering |
| Dirichlet log | `log(a_j) - log(s_i + A)` for prior counts `a_j = A p_j` | consumes prior mass |
| Dirichlet CLR | `log(a_j)` minus the row mean of the log posterior counts | consumes prior mass |

For the residual, correspondence, and normalized-log1p families the column is
**identically zero**. Such a column carries no variance and no inertia, so it contributes
nothing to the decomposition and its coefficient comes back as zero to solver
precision. An empty gene therefore cannot change any other gene's result, and
the outcome matches the same input with the column removed. Two consequences:

* `n_comps` is validated against the number of columns that are not
  identically zero, so empty genes cannot buy components that carry no
  variance;
* per-gene parameters of an empty gene cannot reach any output, so a
  scaled-NB `alpha` value there is replaced with zero rather than validated. Array
  shape is still checked, and a retained gene's `alpha` is still required to be
  finite and nonnegative.

For the log-ratio and Dirichlet families the value is nonzero and does affect
retained genes, because those transforms normalize each observation against the
full declared variable universe: empty genes weaken the shifted-CLR centering
and consume Dirichlet prior mass. This follows from the transform definitions
rather than being a defect. `n_empty_vars` is reported so the effect is
visible; filter empty genes before calling to avoid it.

The one undefined *output* is the correspondence-analysis column principal
coordinate, which divides a singular-vector entry by the square root of the
column mass. That coordinate is reported as `np.nan`, and the column mass is
reported as zero.

`varm` therefore keeps a single convention: `np.nan` means the gene was
excluded by `mask_var`, and a finite value, including zero, means it was
decomposed. Correspondence analysis adds `np.nan` for a zero-mass column's
undefined coordinate.

### Empty cells

Empty cells are rejected for every transform, during count canonicalization,
before any transform-specific code runs.

The value at an empty cell is undefined for residual, composition-scale shifted
CLR, and correspondence analysis. For count-scale shifted CLR it is the zero
vector, and for the Dirichlet families it is the prior; in both cases every
empty cell maps to the same point, so the coordinate is an artifact of the
transform rather than a property of the observation. No family produces an
informative embedding for one.

Empty cells are rejected rather than marked, because the marker would not be
inert. Scores in `obsm` feed directly into downstream neighbor and embedding
steps: a neighbor graph built from scores containing a `NaN` row can acquire
edges to that row without raising, leaving the exclusion invisible downstream.
Removing an empty cell also discards no information, since its counts are all
zero. Filter zero-total rows out of the same matrix or layer before calling.

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

Use the following terms consistently in documentation and metadata:

- `normalization_n_vars`: the number of columns in the selected count source
  before `mask_var`;
- `pca_n_vars`: the number of columns decomposed, after `mask_var`;
- `n_empty_vars`: the number of empty genes in the normalization universe.

Correspondence analysis scopes `n_empty_vars` to its analyzed table instead,
because its mask defines that table and its margins are fitted after masking.
It therefore counts the zero-mass columns of the analyzed table, which are
exactly the columns that receive `NaN` principal coordinates.

The `alpha` exception deliberately uses a different scope, because it answers a
different question. `n_empty_vars` describes the output, so it follows the
analyzed table. The `alpha` rule asks whether a value can reach any output, and
emptiness is a property of the input: an empty variable's overdispersion can
never matter, selected or not. Scoping it to the input keeps the rule identical
across residual PCA and correspondence analysis, so the same `alpha` array
behaves the same way in both.

`normalization_n_vars` and `pca_n_vars` are equal only when PCA decomposes
every normalization gene. Because no column is ever dropped for being empty,
`pca_n_vars` is exactly the selected column count.

`n_empty_vars` is a property of the normalization universe, which is fitted
before `mask_var` is applied, so it counts every empty gene regardless of the
mask. It is what makes the shifted-CLR and Dirichlet retention effect
measurable: the log-ratio centering scale is
`(normalization_n_vars - n_empty_vars) / normalization_n_vars`.

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

    return _ResolvedMask(
        values=mask,
        mask_var=resolved_scanpy_mask_selector,
        use_highly_variable=resolved_scanpy_hvg_flag,
        details={
            "kind": resolved_kind,
            "key": resolved_key,
            "n_vars_used": int(mask.sum()),
        },
    )
```

Mask resolution, Scanpy-compatible parameter values, and compact metadata are
computed together and carried as one `_ResolvedMask`. Result writers therefore
cannot serialize a different interpretation from the mask used for PCA.

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
        "clip_threshold": resolved_clip,
        "clip_mode": clip_mode,
        "clip_max_nnz_ratio": clip_max_nnz_ratio,
        "zero_center": True,
        "layer": layer,
        "mask_var": resolved_mask.mask_var,
        "use_highly_variable": resolved_mask.use_highly_variable,
        "mask_var_details": resolved_mask.details,
        "solver": solver,
        "n_comps": n_comps,
        "pca_n_vars": n_vars_used,
        "n_empty_vars": n_empty_vars,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": resolved_dtype,
        "package_version": __version__,
    },
}
```

Correspondence analysis uses its own default keys:

```python
obsm_key = "X_ca"
varm_key = "CA"
uns_key = "ca"
```

It writes row principal coordinates to `adata.obsm[obsm_key]`, column
principal coordinates to `adata.varm[varm_key]`, and inertia, masses, singular
values, and parameters to `adata.uns[uns_key]`. As with the PCA APIs, passing
`key_added="foo"` uses `"foo"` for all three keys.

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

That is the only reason a PCA `varm` row is `np.nan`: empty genes are
decomposed like any other and carry their true coefficient of zero. See
[empty genes and cells](#empty-genes-and-cells).

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

Residual transforms and both shifted CLR transforms produce rank-one
representations. Size-factor-normalized log1p produces a rank-zero
representation. Dirichlet log and Dirichlet CLR produce representations of
rank at most two.
The generic form supports transforms with multiple implicit components. Row
scaling, column scaling, and column selection preserve the sparse-plus-low-rank
form.

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

At a structural zero, `X_ij = 0`, so `R_ij = -mu_ij / sqrt(V_ij)`.
Substituting each variance below separates this value into `u_i v_j`.

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

Equality therefore does not clip. Exclude stored count locations through each
row's sorted CSR support. Residual evaluation may generate support-aligned row
indices only in bounded blocks; it must not require one full-`nnz` row vector.

The defaults are:

```python
clip="seurat"
clip_mode="symmetric"
clip_max_nnz_ratio=2.0
```

The default clips symmetrically at `sqrt(n_obs / 30)` for every residual
family, so the support-growth guard above is reachable under default
arguments. `clip=None` recovers unclipped residuals.

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
if not 1 <= n_comps < min(n_obs, n_nonempty_vars):
    raise ValueError(
        "n_comps must satisfy 1 <= n_comps < min(n_obs, n_nonempty_vars) "
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
```

## Explained variance and total variance

For a rank-`k` representation

```math
R = S + UV^\top,
```

let `v_j` be column `j` of `V.T`, let
`b_j = Uv_j` be that column's low-rank baseline, and let `P_j` contain the row
indices stored in column `j` of `S`. The represented values are

```math
R_{ij} =
\begin{cases}
b_{ij} + S_{ij} & i \in P_j,\\
b_{ij} & i \notin P_j.
\end{cases}
```

Both required Frobenius norms measure squared deviations of the columns of `R`
from a per-column scalar center `c_j`: `c = 0` for the uncentered norm, and
`c` equal to the column means of `R` for the centered norm. Compute the column
means first.

With `1` the length-`N` vector of ones, let

```math
\bar u = \frac{1}{N}U^\top \mathbf 1
```

hold the column means of `U`. Averaging column `j` of `R` averages the
baseline `b_j = Uv_j` and adds the stored corrections, giving the represented
mean

```math
\bar r_j
=
\frac{1}{N}\mathbf 1^\top U v_j
+
\frac{1}{N}\sum_{i \in P_j} S_{ij}
=
\bar u^\top v_j
+
\frac{1}{N}\sum_{i \in P_j} S_{ij}.
```

For the squared deviations, center the low-rank factor and form its Gram
matrix:

```math
U_c = U - \mathbf 1\bar u^\top,
\qquad
\Gamma_U = U_c^\top U_c.
```

For any scalar center `c_j`, the baseline decomposes as

```math
b_j - c_j\mathbf 1
=
U_c v_j
+
(\bar u^\top v_j - c_j)\mathbf 1,
```

and `1^T U_c = 0` eliminates the cross term, so the baseline squared deviation
over all rows is

```math
\sum_i (b_{ij} - c_j)^2
=
v_j^\top \Gamma_U v_j
+
N(\bar u^\top v_j - c_j)^2.
```

Replace that baseline on stored support to obtain the represented column norm:

```math
q_j(c_j)
=
\sum_i (b_{ij} - c_j)^2
-
\sum_{i \in P_j}(b_{ij} - c_j)^2
+
\sum_{i \in P_j}(b_{ij} + S_{ij} - c_j)^2.
```

Summing over columns gives both norms:

```math
\|R - \mathbf 1 c^\top\|_F^2 = \sum_j q_j(c_j),
```

using `c = 0` for `frobenius_squared_uncentered()` and the stored operator mean
`c = bar_r` for `frobenius_squared_centered()`.

Do not calculate large sparse and low-rank component norms and then subtract
them. For columns with more than half their entries stored, calculate means and
squared deviations from the full represented column directly; this avoids
severe subtraction for dense and nearly dense support. For the remaining
columns, use the support-replacement identities above and traverse sparse
support in `O(nnz)` for fixed representation rank.

Accumulate sparse support sums with vectorized `float64` reductions over
bounded blocks of complete CSR rows. Ordinary floating-point summation applies:
block size and CSR order may affect low-order bits when mixed-sign corrections
cancel severely. Bitwise invariance across block sizes is not required.

Statistics describe the post-cast `S`, `U`, `V`, and mean used by the operator
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

## Verification contracts

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
clip="seurat" and clip="scanpy" with both clip modes
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

### 6. Empty gene and cell tests

No transform may change its variable universe, so a fitted transform must
accept masks sized to its original input.

For the residual and correspondence-analysis families, appending an all-zero
column must leave every retained coefficient, score, and singular value
unchanged to solver precision, and give the empty column a zero coefficient.
Cover every model, residual, and clipping mode, since upper-tail clipping
depends on the sign of the structural-zero residual. `n_comps` at the nonempty
column count must raise. A non-finite `alpha` on an empty gene must be ignored
while one on a retained gene still raises, and a wrong-length `alpha` must
still raise, for both residual PCA and correspondence analysis. A zero-mass
correspondence-analysis column must receive `NaN` principal coordinates and
zero mass.

For the shifted-CLR and Dirichlet families, empty genes must match the same
dense oracle as any other gene, and `n_empty_vars` must report their total.

Empty cells must be rejected by every entry point, including a row emptied by a
correspondence-analysis variable mask.

### 7. Clipping tests

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
each named threshold equals its numeric equivalent in both clipping modes
named thresholds resolve against the fitted observation count, including for
  a specification reused across datasets
a named threshold can trigger the support-growth guard
the clipped default applies to Pearson and deviance residuals
metadata records the request and the resolved threshold for names, numbers,
  and None
matrix, AnnData, one-step, and two-step entry points resolve names alike
unknown clip names are rejected
```

### 8. Variance tests

For small matrices, compare analytic total variance to dense calculation:

```python
np.sum((R - R.mean(axis=0)) ** 2) / (n_obs - 1)
```

Also compare explained variance ratio to dense SVD.

### 9. Townes reference test

`tests/townes_reference/test_townes_reference.py` stores singular values
generated from `null_residuals()` in the Townes `scrna2019` repository and
compares them with `residual_pca_matrix`.

Normal pytest runs must not require R, network access, or a local reference
checkout. `tests/townes_reference/generate_reference.R` is a manual
regeneration script that requires the path to the upstream `functions.R` as
its sole argument. The adjacent README records the upstream repository,
commit, source file, and regeneration command.
