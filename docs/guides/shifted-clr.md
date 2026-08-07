# Shifted CLR

Centered log-ratio (CLR) coordinates subtract each observation's mean log
abundance, representing within-observation log ratios. The shift applied before
taking logs can be fixed on the count scale or on the composition scale, and
those imply different transforms.

This guide's [complete example](#complete-example) is
[`examples/shifted_clr.py`](https://github.com/jgarthur/sparse_count_pca/blob/main/examples/shifted_clr.py).

Each function below has a `_matrix` counterpart that takes a dense or sparse
matrix and returns a result object instead of working with AnnData.

## Count-scale shifted CLR

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

The PFlog normalization proposed in
[Booeshaghi et al., preprint version 4 (June 22,
2026)](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4) is this
transform with `count_shift = 1 / (4 * alpha)`:

```python
alpha = 0.25
result = scp.shifted_clr_pca_matrix(
    counts,
    n_comps=20,
    count_shift=1 / (4 * alpha),
)
```

That `alpha` is a dataset-wide overdispersion under a negative-binomial
size-factor model, not the per-gene
[`scaled_nb` `alpha`](../reference/scaled-nb-model.md) of the residual
transforms in this package. Because the shift is its reciprocal, a more
overdispersed dataset gets a smaller `count_shift`.

The dedicated [`cleartools/scclr`](https://github.com/cleartools/scclr) package
provides the same PFlog parameterization through Rust-backed Python tooling,
built on the [`runorm`](https://github.com/cleartools/runorm) normalization
crate. `runorm` selects the shift through a proportional-fitting target rather
than a pseudocount argument; that target choice decides whether it matches
this transform or `ProportionShiftedCLR`. See
[compatibility](../reference/compatibility.md#shifted-clr-pflog-and-cleartools)
for the scope of this relationship.

## Composition-scale shifted CLR

`ProportionShiftedCLR` is a distinct historical transform:

```text
clr(x_ij / cell_total_i + composition_shift).
```

```python
scp.proportion_shifted_clr_pca(
    adata,
    layer="counts",
    n_comps=20,
    composition_shift=0.1,
)
```

Its effective raw-count shift is
`cell_total_i * composition_shift`, so it changes with observation depth. Do
not exchange `count_shift` and `composition_shift` as if they were the same
parameter.

## Normalization universe and masking

CLR row means use all genes in the chosen count matrix before `mask_var`
selects PCA variables. Consequently, removing a gene from the input and masking
that gene out of PCA are not equivalent.

See [normalization, masking, and centering](../concepts/normalization-masking-and-centering.md)
for the package-wide ordering rule.

## Complete example

--8<-- "examples/shifted_clr.md"
