# Size-factor-normalized log1p

Normalized log1p divides each observation's counts by a positive size factor,
takes the natural logarithm of one plus that value, and runs column-centered
PCA. Zeros stay zero, so the normalized matrix remains sparse; PCA centering
is handled implicitly.

This guide's [complete example](#complete-example) is
[`examples/log1p_normalization.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/log1p_normalization.py).

## AnnData workflow

Pass raw counts from `.X` or a layer:

```python
import sparse_count_pca as scp

scp.log1p_norm_pca(adata, layer="counts", n_comps=20)
```

By default, each observation is normalized to the median count total across
observations. If observation `i` has total `n_i` and the target is `t`, its
size factor is `n_i / t`, and the transformed count is `log1p(x_ij / (n_i / t))`.

The call writes scores to `adata.obsm["X_pca"]`, components to
`adata.varm["PCs"]`, and variance statistics and parameters to
`adata.uns["pca"]`. It returns `None` and leaves counts unchanged. Use
`copy=True` to return a modified copy, or `key_added` to keep several analyses;
see [AnnData workflows](anndata-workflows.md).

## Choose a target total

Set `target_sum` to normalize each observation to an explicit positive total:

```python
scp.log1p_norm_pca(adata, layer="counts", n_comps=20, target_sum=1e4)
```

This uses the same normalization formula as Scanpy's `pp.normalize_total`
followed by `pp.log1p`. The package keeps transformed values implicit and
writes PCA results rather than replacing `.X` with normalized values. See
[Scanpy compatibility](../reference/compatibility.md#coming-from-scanpy) for
the scope of the comparison and differences in supported options.

## Supply size factors

Without supplied factors, the package divides each observation's total count
by the median total to obtain its size factor. For totals of 500, 1000, and
2000, the default factors are 0.5, 1, and 2. An explicit `target_sum` replaces
the median in that calculation.

If you already have size factors from another method, pass them instead as a
vector or an `adata.obs` column. They are used as supplied, without adjusting
their median. For example, if your factors are in `adata.obs["size_factor"]`:

```python
scp.log1p_norm_pca(
    adata,
    layer="counts",
    n_comps=20,
    size_factors="size_factor",
)
```

## Matrix workflow and inspecting values

Outside AnnData, use the matrix function. The median default, explicit
`target_sum`, and supplied `size_factors` behave the same way:

```python
result = scp.log1p_norm_pca_matrix(counts, n_comps=20, target_sum=1e4)
result.scores
result.components
```

To inspect normalized values or reuse normalization with different PCA masks,
fit `Log1pNormalized` first:

```python
normalized = scp.transform(adata, scp.Log1pNormalized(target_sum=1e4), layer="counts")
values = normalized.materialize(obs=slice(0, 3))
result = normalized.pca(n_comps=20)
```

`materialize` returns a dense array for the requested slice. See
[transform once, reuse](transform-reuse.md) for bounded materialization and
the two-step result contract.

## Normalization universe and input checks

Cell totals and the median target are computed from every gene in the count
matrix before `mask_var` selects PCA variables. The default mask uses
`adata.var["highly_variable"]` if present; `mask_var=None` uses all genes.
Slice the count matrix first if excluded genes should not affect totals. See
[normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

Remove zero-total observations before fitting, even when supplying your own
size factors.

## Complete example

--8<-- "examples/log1p_normalization.md"
