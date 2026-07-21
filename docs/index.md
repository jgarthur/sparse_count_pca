# Documentation

`sparse-count-pca` is organized around named count transforms, exact implicit
matrix representations, and PCA. Start with the path that matches your goal.

## Review the project

1. [Architecture review guide](architecture.md) — guided source tour, diagrams,
   invariants, and test strategy.
2. [Package specification](specification.md) — normative API, mathematical,
   validation, numerical, and AnnData contracts.
3. [Log-transform design](design/log-transforms.md) — why fixed-count,
   fixed-composition, and Dirichlet shifts are separate APIs.
4. [Wishlist](wishlist.md) — plausible additions, especially block-wise PCA
   reconstruction and inverse transforms.

## Historical context

- [Correctness and numerical robustness brief](history/robustness-work-brief.md)
  records the hardening work that preceded the generic PCA architecture. It is
  archival rather than normative.
- [Real-data oracle suite history](history/real-data-oracle-suite.md) records
  the completed PBMC3k fixture and shared dense-oracle work.

The former standalone PFlog correction note was removed after its conclusions
were incorporated into the log-transform design, implementation, tests, and
specification.
