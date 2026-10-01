# TODO

- Before finalizing the release, verify the changelog claim about peak temporary
  memory across supported input formats and dtypes. Distinguish bounded
  arithmetic scratch from per-cell/per-gene vectors, owned CSR structure,
  canonicalization copies, unusually long rows, feature selection, and retained
  operators; qualify the claim if those allocations require it.
