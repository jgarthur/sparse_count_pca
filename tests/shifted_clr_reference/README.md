# Shifted CLR reference implementation

`oracle.py` contains the formula implementation copied from `do_pf()` and
`norm_clr()` in the authors' BHGP_2022 reference repository:

- Repository: <https://github.com/pachterlab/BHGP_2022>
- Commit: `bde2ad7d34ef05241b4d907e8c6ba251f5974c34`
- Source file: `scripts/norm_sparse.py`

That pinned revision computes `u = x / sum(x)` and then returns
`log(u + c) - mean(log(u + c))`, matching the equation in the June 10, 2026
paper. The next repository revision changed the default proportional-fitting
target to mean cell depth, so the commit is intentionally pinned rather than
tracking the current branch.

The copied code remains under the upstream BSD 2-Clause license in `LICENSE`.
