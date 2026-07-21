# Real-data oracle suite history

This note records the real-data verification work completed for pull request
2. It was moved out of the wishlist once implemented.

The repository now commits deterministic observed-depth and equal-depth
PBMC3k-derived CSR fixtures for offline tests. Their generator verifies the
official 10x source archive and uses identity-based SHA-256 ranking for cell,
gene, and molecule selection. A compact manifest pins the source and artifact
checksums plus shape, sparse-entry, and count totals; the tests check the
failure modes that matter for committed binaries without duplicating the full
fixture contents in JSON.

Shared independent dense formulas compare every matrix entry for all public
residual, logarithmic, Dirichlet, and correspondence transforms. Small
simulated tests and full real-data tests use the same formula implementations,
while retaining separate cases for their distinct API, SVD, support-growth,
and real-count responsibilities. The suite also covers noncontiguous gene
masks and runtime-appended all-zero genes.

The equal-depth fixture additionally has a pinned SCTransform v2 reference.
It uses the `glmGamPoi_offset` backend and tests the shared scaled-negative-
binomial Pearson formula, clipping-before-centering order, and all centered
residual entries.

Remaining external implementation coverage stays in
[`wishlist.md`](../wishlist.md). Future completed wishlist items should be
removed there and recorded in a normative or historical document rather than
being left as already-finished wishes.
