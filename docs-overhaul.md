# Documentation overhaul work brief

> **Status:** Living planning document for the documentation overhaul.
> Keep this file updated while the work is in progress. Before the repository
> is published, either move it to the private workbench or remove it after its
> durable decisions have been incorporated into public documentation.

## Goal

Create a release-ready documentation foundation that gives users a short path
to a successful first analysis while preserving the package's detailed
scientific, numerical, and implementation contracts for readers who need them.

The documentation should serve three audiences without interleaving their
needs:

1. **Users** need installation, task-oriented guides, transform selection,
   expected outputs, caveats, examples, and API reference.
2. **Scientific readers and reviewers** need the mathematical representation,
   model derivations, exactness claims, compatibility analysis, and explicit
   limitations.
3. **Contributors and maintainers** need architecture, normative behavior,
   invariants, validation contracts, source orientation, and test strategy.

## Agreed decisions

- The README will be an orientation page, not the complete documentation.
- The current README length is not itself the problem. Its specification-level
  detail will be moved behind clear links and its information hierarchy will be
  rewritten.
- The documentation homepage will be user-facing. Architecture, specification,
  design, and history will remain accessible but will not be the primary entry
  points.
- We will start with a locally buildable MkDocs site. Read the Docs integration
  will be a later step after the local content and build are stable.
- There are no benchmark results, preprint URL, figure assets, or citation
  metadata yet. The documentation will not invent placeholders or unsupported
  performance claims for them.
- Notebooks, benchmark infrastructure, paper reproduction, custom site design,
  versioned documentation, and a public roadmap are deferred.
- No public root-level `AGENTS.md` will be added for the initial release.
  Durable rules belong in contributor, architecture, specification, and testing
  documentation; tool-specific routing stays private.
- Public API documentation will be generated selectively rather than rendering
  every internal module.
- Existing Google-style Python docstrings will be retained and completed for
  the public API. Converting styles is lower priority than documenting exported
  behavior accurately and consistently.
- Examples will be small, executable, fast, and checked in CI.

## Author-owned copy

The project author will write the opening section of the README, including the
project's framing, scientific motivation, and central promise. This is the most
important place to preserve the author's voice and judgment.

The documentation work can proceed around that section. Codex can:

- provide a structural prompt or a few alternative outlines;
- edit the author's draft for clarity and concision;
- check that technical claims match the implementation and tests;
- connect the opening naturally to installation and quick start;
- preserve the author's wording where it reflects scientific positioning.

Other passages that may benefit from direct author input are:

- why this package should exist alongside related methods;
- which workflow should be presented as the recommended default;
- the strength and boundaries of exactness and compatibility claims;
- limitations that should be prominent before the first release.

Draft copy should not block structural work. Pages can use clearly marked
editorial notes while author-owned text is pending, but no such notes should
remain in the release-ready documentation.

### Initial author draft

The following draft was supplied on 2026-07-21. Preserve its scientific intent
while checking quantitative and implementation claims before adapting it for
the README.

> Single-cell transcriptomics data is delivered to the user as a large, very
> sparse matrix of integer counts. That matrix is often made dense by
> normalization operations, which may result in ~10-20x [check this] higher
> memory costs for storing the matrix entries. Fortunately, the primary
> downstream operation on the normalized matrix is PCA; common R and Python
> operations
>
> This package covers: "standard" log1p-normalization, shifted centered
> log-ratio normalization, Pearson/deviance residuals on Poisson and scaled NB
> models (see note below), correspondence analysis, and a few other
> experimental methods like Bayesian (Dirichlet prior) pseudocounts and a
> scaled-NB variation on correspondence analysis.
>
> The implementation is pure Python, with extensive testing against prior
> dense-matrix implementations of normalization schemes. We utilize scipy's
> LinearOperator API, which allows one to avoid materializing the full dense
> matrix and instead only implement matrix multiplication. This scheme is
> compatible with SVD (exact function?) but also any function that only relies
> on matrix multiplication.
>
> Prior works, both of which use rust for implicit normalization and PCA,
> includes:
>
> - the cleartools packages for shifted CLR (aka PFlog), which have Python and R
>   frontends;
> - the rust-only scan-rs package developed for 10x Genomics' Ranger pipelines;
>   see AdaptiveMat and normalization code.

Claim review notes:

- Replace the unmeasured `10-20x` statement with a density- and dtype-dependent
  explanation until a reproducible benchmark exists. A sparse-to-dense
  conversion can increase entry storage by an order of magnitude or more, but
  no single multiplier is package-wide.
- Name the current numerical path precisely: SciPy's
  `scipy.sparse.linalg.svds` with `solver="arpack"` accepts the package's
  `LinearOperator` and computes a truncated SVD from matrix products.
- Do not imply that every function accepting a `LinearOperator` automatically
  works with this package. State that the representation can in principle be
  used by compatible matrix-free algorithms, while PCA is the supported public
  workflow today.
- Present `cleartools/scclr` and `scan-rs` as related work. Keep detailed
  relationships and pinned source links in the compatibility page rather than
  in the opening paragraphs.

## Scope for this overhaul

### README

Restructure `README.md` around:

1. author-written opening and central promise;
2. installation;
3. minimal AnnData quick start;
4. a compact transform-selection table;
5. a short explanation of sparse-plus-low-rank computation;
6. the three interface paths;
7. concise compatibility and precision caveats;
8. links to user documentation, scientific details, and development material;
9. development commands.

Move detailed masking semantics, AnnData keys, backed-array behavior, clipping
contracts, dtype semantics, validation rules, external equality-oracle details,
and full transform explanations into the appropriate guides or references.

### User documentation

Create the following initial pages when the content justifies them:

```text
docs/
    index.md
    getting-started.md
    choosing-a-transform.md

    guides/
        residual-pca.md
        shifted-log-and-clr.md
        transform-reuse.md
        correspondence-analysis.md
        anndata-workflows.md

    concepts/
        sparse-plus-low-rank.md
        normalization-masking-and-centering.md
        clipping-and-precision.md

    reference/
        compatibility.md
        api/
```

Pages should be created only when they contain useful material. Do not create
empty destination pages merely to match this layout.

### Development and reviewer documentation

Reorganize the existing technical material under:

```text
docs/development/
    architecture.md
    specification.md
    testing.md
    design/

docs/history/
```

The specification will be labeled normative and addressed to maintainers,
reviewers, and coding agents. The architecture guide will explain how the
implementation is divided. The testing guide will describe the evidence
required to preserve documented behavior.

The existing wishlist will not appear in primary site navigation. Its durable
decisions should move to appropriate public documents or issues; transient
planning should move to the private workbench or be removed.

### Public API documentation

Add curated generated pages for:

- AnnData entry points;
- matrix entry points;
- transform specifications and `transform`;
- `TransformedMatrix`;
- PCA and correspondence-analysis result objects.

Every exported function, class, and result object should document, as
applicable:

- purpose;
- parameter forms and defaults;
- matrix orientation and shapes;
- dtype behavior;
- return values or AnnData side effects;
- major exceptions;
- important semantic notes;
- a small example;
- links to relevant guides and concepts.

Docstrings should link to long explanations rather than duplicate them.

### Examples

Add small generated-data scripts for canonical workflows:

```text
examples/
    residual_pca_anndata.py
    shifted_clr_anndata.py
    transform_reuse.py
    correspondence_analysis.py
```

Each Python file must have a module docstring. Any tests added to execute these
examples must give every test function a behavior-focused docstring.

## Deliberately deferred work

- Read the Docs project setup and `.readthedocs.yaml` configuration;
- measured performance pages and README benchmark figures;
- benchmark scripts, environments, and machine-readable results;
- preprint and citation links;
- notebooks and notebook rendering;
- paper reproduction workflows;
- custom theme work;
- documentation versioning;
- a public roadmap;
- a large example gallery;
- a public, tool-specific agent instruction file.

Deferred items should be added only when the underlying content or workflow
exists. Their absence must not leave broken links or placeholder sections.

## Proposed implementation sequence

### 1. Local documentation foundation

- Add pinned documentation dependencies.
- Add `mkdocs.yml` with Material and local search.
- Configure selective API rendering with `mkdocstrings`.
- Add a strict local documentation build command.
- Add that build to CI after it succeeds locally.

### 2. Information architecture and relocation

- Rewrite `docs/index.md` as the user homepage.
- Move architecture, specification, and design files to their intended
  development paths.
- Repair repository-relative and site-relative links.
- Remove the wishlist and archival documents from primary navigation.

### 3. README and core user journey

- Reserve the opening section for author-written copy.
- Add installation and the minimal working example.
- Add the transform-selection table and interface overview.
- Move detailed material into guides, concepts, and references.
- Keep only concise, carefully scoped compatibility statements in the README.

### 4. Guides and concepts

- Rework existing README and specification material into task-oriented guides.
- Write the sparse-plus-low-rank explanation for scientific users.
- Explain normalization, masking, centering, clipping, and precision without
  reproducing the normative specification.
- Centralize compatibility and non-equivalence claims.

### 5. Public API and examples

- Complete public docstrings using the repository's existing style.
- Add curated generated API pages.
- Add and test the canonical example scripts.
- Ensure examples and guides use the same recommended patterns.

### 6. Release-readiness review

- Build documentation strictly with no warnings.
- Run links and code examples through the available offline checks.
- Run the full test and lint suites.
- Check that no planning placeholders remain.
- Decide whether this work brief moves to the private workbench or is removed.
- Begin Read the Docs integration as a separate final step.

## Definition of done for the local phase

- A new user can install the package and reach a successful first call from the
  README or documentation homepage without reading developer material.
- A user can choose among residual, shifted-log/CLR, Dirichlet, and
  correspondence-analysis workflows without reading the specification.
- Detailed masking, clipping, precision, AnnData, and compatibility behavior
  has one clear home and is linked rather than repeatedly restated.
- The public API reference is curated and generated from useful docstrings.
- Canonical examples execute in CI.
- Architecture, specification, and testing documentation are clearly separated
  by purpose.
- The MkDocs site builds locally in strict mode and the CI build passes.
- The site contains no unsupported benchmark, citation, preprint, or hosting
  claims.

## Progress

- [x] Create a clean `docs-overhaul` worktree from `real-data-oracles`.
- [x] Inventory the existing README, documentation, source exports, and CI.
- [x] Agree on the initial documentation scope and deferred work.
- [x] Reserve the README opening for author-written copy.
- [x] Establish the local MkDocs build.
- [x] Reorganize the documentation hierarchy.
- [x] Draft the user homepage and getting-started path.
- [x] Restructure the README around an edited version of the author draft.
- [x] Add guides, concepts, and compatibility reference.
- [x] Complete public API docstrings and generated reference pages.
- [x] Add executable examples and CI coverage.
- [x] Pass strict documentation build, lint, and the full test suite.
- [ ] Complete the author's editorial review of the README opening.
- [ ] Perform visual browser QA when a browser session is available.
- [ ] Move this work brief to the private workbench or remove it before release.
- [ ] Configure and connect Read the Docs.

## Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-07-21 | Start with a local MkDocs site; add Read the Docs later. | Stabilize content and the strict build before adding hosting integration. |
| 2026-07-21 | Omit benchmarks, preprint, and citation sections for now. | Their underlying results and metadata do not yet exist. |
| 2026-07-21 | Keep the README opening author-owned. | Project framing and scientific positioning should carry the author's voice. |
| 2026-07-21 | Do not add a public `AGENTS.md` initially. | Durable engineering knowledge belongs in public contributor documentation; tool workflow remains private. |
| 2026-07-21 | Retain and complete Google-style public docstrings. | The repository already uses this style; completeness matters more than format conversion. |
| 2026-07-21 | Describe shifted log as a fixed-count transform, not standard library-size-normalized log1p. | The implemented formula is `log1p(X / count_shift)` and does not estimate or apply row size factors. |
| 2026-07-21 | Describe memory benefit as density- and dtype-dependent. | No reproducible project benchmark exists yet, so a universal `10-20x` claim would be unsupported. |
| 2026-07-21 | Name SciPy `svds` with ARPACK as the supported numerical path. | This is the exact function and solver used with the package's `LinearOperator`. |
| 2026-07-21 | Integrate merged PR #2 before continuing editorial work. | Its final commits document the real-data oracle suite and update provenance contracts already covered by the overhaul; deferring would leave stale links and an obsolete branch base. |

## Verification record

Verified on 2026-07-21:

```text
mkdocs build --strict: passed
ruff check .: passed
pytest after main integration: 369 passed, 1 skipped
canonical example scripts: 4 passed
```

The Material theme emits an informational upstream notice about future MkDocs
2.0 compatibility. It does not fail strict mode. Visual browser QA was
attempted, but no browser session was available in the environment.

The branch was fast-forwarded to merged PR #2 at `ccb3958` before the final
documentation reconciliation. Its PBMC3k fixture, shared-oracle,
source-distribution, and remaining-parity details were folded into the
development, testing, and compatibility pages.

## Open editorial questions

- What exact problem statement and promise should open the README?
- Which transform should be presented as the default starting point for a user
  who has sparse single-cell counts but no method preference?
- Should the first quick start assume an existing `AnnData` object with a
  `"counts"` layer, or construct a tiny complete object?
- Which limitations deserve visibility in the README rather than only in the
  compatibility and concept pages?
