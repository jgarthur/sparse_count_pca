# Transform once, reuse several times

The one-step APIs are the simplest route to PCA, but the two-step API via `transform` allows you
to inspect transformed values, materialize bounded slices, or reuse one fitted
normalization with several PCA masks.

This guide's [complete example](#complete-example) is
[`examples/transform_reuse.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/transform_reuse.py).

## Fit a transform

```python
import sparse_count_pca as scp

transformed = scp.transform(
    adata,
    scp.Residual(model="poisson", residual="pearson"),
    layer="counts",
)
```

Other primary transform specifications are `Log1pNormalized`, `ShiftedCLR`,
and `ProportionShiftedCLR`:

```python
log_normalized = scp.transform(
    adata,
    scp.Log1pNormalized(size_factors="size_factors"),
    layer="counts",
)
```

String-valued per-observation parameters, such as the `adata.obs` size-factor
key above, are resolved when the transform is fitted. Correspondence analysis
remains a separate one-step API because its variable selection changes the
fitted table margins.

## Inspect selected values

`TransformedMatrix` is an uncentered `scipy.sparse.linalg.LinearOperator`.
Materialization always returns a two-dimensional array, including for a single
row or variable:

```python
cell = transformed.materialize(obs=10)
gene = transformed.materialize(var=25)
subset = transformed.materialize(obs=[10, 3], var=[25, 2, 8])
```

For AnnData input, observation and variable names are also accepted. Slices,
boolean masks, and ordered integer arrays are supported.

The current backend is observation-oriented CSR. Observation blocks are the
efficient access direction. Selecting variables across every observation may
scan the CSR rows and emits `SparseEfficiencyWarning`.

## Bound materialization memory

Provide `out` and `block_size` to stream row blocks into an existing NumPy
array or memory map:

```python
import numpy as np

out = np.memmap(
    "transformed.dat",
    mode="w+",
    shape=transformed.shape,
    dtype="float32",
)
transformed.materialize(out=out, block_size=2048)
```

This bounds temporary output memory, but it does not make the current fitting
backend out of core: backed sparse inputs are loaded into an in-memory CSR
matrix during count canonicalization.

## Reuse normalization with different PCA masks

```python
first = transformed.pca(n_comps=20, mask_var="highly_variable")
second = transformed.pca(n_comps=20, mask_var=my_alternative_mask)
```

Normalization state was fitted once on the full count matrix. Each `pca` call
selects variables and then computes its own column mean. This ordering is
described in [normalization, masking, and centering](../concepts/normalization-masking-and-centering.md).

Unlike the one-step APIs, `pca` **returns** a `PCAResult` and writes nothing
back, even when the transform was fitted from an `AnnData` object. That is what
makes several results comparable side by side. To store one, assign it
yourself:

```python
adata.obsm["X_pca"] = first.scores
adata.varm["PCs"] = first.components.T
```

With a `mask_var`, `components` covers only the selected variables, so a
full-width `varm` array needs the unselected rows filled with `np.nan`.

## Lifetime and ownership

The fitted object owns the sparse support needed to isolate it from later
mutation of the caller's matrix. It lives only in the current Python process:
fitting a transform adds nothing to the AnnData object, so `write_h5ad` does
not store it, and it has to be refitted after reloading the file or restarting
Python. Only PCA results written to `.obsm`, `.varm`, and `.uns` persist.

Passing `return_operator=True` to `pca` retains the centered operator ARPACK
used. It is isolated from the `TransformedMatrix` it came from: mutating either
one cannot affect the other.

## Complete example

--8<-- "examples/transform_reuse.md"
