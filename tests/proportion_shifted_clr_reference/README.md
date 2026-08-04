# Composition-scale shifted CLR reference

The dense oracle in `oracle.py` implements
`clr(X / row_sum(X) + composition_shift)`.

This is the formula used by `do_pf()` and `norm_clr()` in the authors'
[`pachterlab/BHGP_2022`](https://github.com/pachterlab/BHGP_2022/tree/bde2ad7d34ef05241b4d907e8c6ba251f5974c34)
repository at commit
`bde2ad7d34ef05241b4d907e8c6ba251f5974c34`, matching the June 10, 2026
manuscript revision. The oracle is an independent dense implementation written
for this test suite; no upstream source code is copied.
