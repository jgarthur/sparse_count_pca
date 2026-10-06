# TODO

- Qualify the changelog memory claim: validation and transform arithmetic use
  bounded scratch, with complete-row limits allowing a longer row to exceed the
  target. Per-cell/per-gene vectors, dense conversion, canonicalization copies,
  feature selection, transformed storage during one-step PCA, and retained
  operators can grow with the dataset. Describe those costs separately from
  bounded arithmetic buffers.
