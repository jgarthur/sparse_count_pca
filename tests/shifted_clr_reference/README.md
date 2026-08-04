# Count-scale shifted CLR reference

The dense oracle in `oracle.py` follows the current PFlog count-scale formula:

```text
center_rows(log1p(4 * alpha * X))
```

This is equivalent to `clr(X + 1 / (4 * alpha))`. The formula and sparse
implementation are documented by the upstream
[`pachterlab/BHGP_2022`](https://github.com/pachterlab/BHGP_2022/tree/ddb3602120f7bc422cf68bd0e13cded6c1a2b0dc)
repository at commit `ddb3602120f7bc422cf68bd0e13cded6c1a2b0dc` (June 24,
2026) and by [`cleartools/scclr`](https://github.com/cleartools/scclr).

The oracle is an independent dense implementation written for this test suite;
no upstream source code is copied.
