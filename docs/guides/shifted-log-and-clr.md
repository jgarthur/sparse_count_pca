# Shifted log and shifted CLR

The log-family APIs make the shift domain explicit. A count-scale shift and a
composition-scale shift imply different transforms, especially when cell
totals vary.

## Fixed-count shifted log

`ShiftedLog` uses

```text
log1p(x_ij / count_shift).
```

Its column-centered PCA is identical to PCA of `log(x_ij + count_shift)`, since
the two matrices differ only by the constant `log(count_shift)`.

```python
result = scp.shifted_log_pca_matrix(
    counts,
    n_comps=20,
    count_shift=1.0,
)
```

This is a fixed raw-count transform. It does not divide rows by library size
before taking logs and should not be described as Scanpy's usual
`normalize_total` plus `log1p` workflow.

## Fixed-count shifted CLR

`ShiftedCLR` subtracts the within-observation mean after applying the same
raw-count shift to every gene:

```text
log(x_ij + count_shift)
    - mean_j(log(x_ij + count_shift)).
```

```python
scp.shifted_clr_pca(
    adata,
    layer="counts",
    n_comps=20,
    count_shift=1.0,
)
```

Current PFlog is this transform with
`count_shift = 1 / (4 * alpha)`:

```python
alpha = 0.25
result = scp.shifted_clr_pca_matrix(
    counts,
    n_comps=20,
    count_shift=1 / (4 * alpha),
)
```

The dedicated [`cleartools/scclr`](https://github.com/cleartools/scclr)
implementation provides the same PFlog parameterization through Rust-backed
Python tooling. See [compatibility](../reference/compatibility.md) for the
scope of this relationship.

## Fixed-composition shifted CLR

`ProportionShiftedCLR` is a distinct historical transform:

```text
clr(x_ij / cell_total_i + composition_shift).
```

```python
result = scp.proportion_shifted_clr_pca_matrix(
    counts,
    n_comps=20,
    composition_shift=1.0,
)
```

Its effective raw-count shift is
`cell_total_i * composition_shift`, so it changes with observation depth. Do
not exchange `count_shift` and `composition_shift` as if they were the same
parameter.

## Normalization universe and masking

CLR row means use all genes in the chosen count matrix before `mask_var`
selects PCA variables. Consequently, removing a gene from the input and masking
that gene out of PCA are not equivalent. Even an all-zero gene changes the CLR
row-mean denominator if it remains in the normalization universe.

See [normalization, masking, and centering](../concepts/normalization-masking-and-centering.md)
for the package-wide ordering rule.

## Related Dirichlet transforms

`DirichletLog` and `DirichletCLR` replace a single fixed shift with an explicit
Dirichlet prior whose per-gene pseudocounts are
`concentration * prior_proportions`. Use them when that prior has a meaningful
interpretation; see [choosing a transform](../choosing-a-transform.md) and the
[transform API](../reference/api/transforms.md).
