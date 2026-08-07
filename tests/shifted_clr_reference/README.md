# Count-scale shifted CLR reference

The dense oracle in `oracle.py` follows the current PFlog count-scale formula:

```text
center_rows(log1p(4 * alpha * X))
```

This is equivalent to `clr(X + 1 / (4 * alpha))`, so the package's `count_shift`
corresponds to `1 / (4 * alpha)`. That `alpha` is a dataset-wide overdispersion
under a common negative-binomial size-factor model, not the per-gene,
depth-scaled `alpha` the package's residual transforms accept.

The formula and sparse implementation are documented by the upstream
[`pachterlab/BHGP_2022`](https://github.com/pachterlab/BHGP_2022/tree/ddb3602120f7bc422cf68bd0e13cded6c1a2b0dc)
repository at commit `ddb3602120f7bc422cf68bd0e13cded6c1a2b0dc` (June 24,
2026) and by [`cleartools/scclr`](https://github.com/cleartools/scclr), which is
built on the [`runorm`](https://github.com/cleartools/runorm) normalization
crate. In `runorm` this is PFlog under an alpha proportional-fitting target,
which holds the row scale constant across cells; a depth target gives the
composition-scale shift instead.

The oracle is an independent dense implementation written for this test suite;
no upstream source code is copied.
