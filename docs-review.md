# Documentation review — findings and implementation brief

> Reviewed 2026-07-22 against branch `docs-overhaul` at `d462562`.
> Audience for this file: the agent implementing the changes. Companion to
> `docs-overhaul.md` (the plan); this file is the critique of the result.
> Reading position taken throughout: **a competent Scanpy user who has never
> heard of this package.**

## Summary

The documentation is accurate, carefully hedged, and unusually disciplined
about not overclaiming. The problem is not quality; it is **findability and
missing provenance**. Three things are wrong at the structural level:

1. **No page answers "what transforms are offered, exactly what are they, and
   where do they come from."** That information exists, but it is scattered
   across five files at three different levels of formality, and the parts a
   scientist most wants — the actual formulas and the source papers — live
   almost entirely in developer-facing material or in `tests/*/README.md`.
2. **`Examples/` is dead weight.** It is orphaned (nothing links to it),
   prose-free, and duplicates code already in the guides. It is the main reason
   the `examples / guides / concepts` split feels hard to navigate.
3. **The Scanpy user's first three questions are unanswered:** what does this
   replace in my pipeline, what do I gain, and how do I keep going afterward.

Priorities below are ordered for implementation. **P0** items are the ones that
change whether a new user succeeds.

---

## P0-1. Add a canonical "Transforms" reference page

This is the single highest-value change and directly addresses "what transforms
are offered and exactly what are they, where do they come from."

### Where that information lives today

| What the user wants | Where it currently is | Problem |
| --- | --- | --- |
| List of transforms | `README.md:52-60` table | Names + one-liners only; omits Dirichlet; below the fold |
| When to pick which | `docs/choosing-a-transform.md` | Selection advice, but **no formulas at all** |
| Shifted-log / CLR formulas | `docs/guides/shifted-log-and-clr.md:12,36,73` | In ```` ```text ```` fences, not typeset math |
| Pearson / deviance formulas | `docs/development/specification.md:1164-1266` | **Developer-facing only**; user never sees them |
| `scaled_nb` mean & variance | `docs/guides/residual-pca.md:36-44` | The *only* real math in the user guides |
| Dirichlet formulas | `docs/development/specification.md:225-238` | Spec only; invisible to users |
| Method provenance / citations | `tests/*/README.md` | **Not in the docs site at all** |

A user comparing methods must currently open the README, `choosing-a-transform`,
two guides, and the normative specification, and still will not find a citation
for deviance residuals or correspondence analysis.

### The provenance gap is the sharpest finding

The repository knows exactly where every method comes from — it validates
against pinned external implementations — but **none of that reaches the
reader.** Every user-facing citation in the entire docs tree:

- `https://www.biorxiv.org/content/10.1101/2022.05.06.490859v4` (Booeshaghi et al., PFlog) — 4 places
- `https://doi.org/10.1093/bioinformatics/btt143` (sSeq) — 1 place, and only as a "long history" aside

Meanwhile the test fixtures cite, precisely and with commit pins:

- `tests/townes_reference/README.md` → Townes et al., `willtownes/scrna2019`, `null_residuals()` — the deviance/Pearson residual-PCA lineage
- `tests/corral_reference/README.md` → Bioconductor `corral` 1.20.0 — the correspondence-analysis lineage
- `tests/shifted_clr_reference/README.md`, `tests/proportion_shifted_clr_reference/README.md` → `pachterlab/BHGP_2022` with commit pins
- `tests/real_data_reference/README.md` → `sctransform` 0.4.3

**Nothing in `docs/` outside `development/` and `history/` mentions Townes,
corral, Greenacre, Lause/Kobak, Hafemeister, or Aitchison.** Grep confirms it.

### What to build

Create `docs/transforms.md`, placed in nav **immediately after Getting started
and before Choosing a transform**. It is a reference catalogue, not a tutorial:
`choosing-a-transform.md` stays as the decision aid and links into it.

Structure — one section per transform, each with the same six fields:

```
### <Transform name>            e.g. "Pearson / deviance residuals"

Formula        typeset math (MathJax $$), not a ```text fence
Call it with   AnnData fn | matrix fn | Transform spec object   (all three)
Parameters     the ones that define the transform, with defaults
Origin         the paper/implementation it comes from, with citation
Validated by   what this package tests it against, linked to tests/*/README.md
Caveats        one or two lines, linking to the concepts pages
```

Open the page with a compact index table so the whole surface is visible at a
glance — this is the table the README should link to and, in reduced form,
mirror:

| Transform | AnnData function | Matrix function | Spec object | Status |
| --- | --- | --- | --- | --- |
| Residual (Poisson / binomial / scaled-NB × Pearson / deviance) | `residual_pca` | `residual_pca_matrix` | `Residual` | stable |
| Fixed-count shifted log | `shifted_log_pca` | `shifted_log_pca_matrix` | `ShiftedLog` | stable |
| Fixed-count shifted CLR (PFlog) | `shifted_clr_pca` | `shifted_clr_pca_matrix` | `ShiftedCLR` | stable |
| Proportion-shifted CLR | `proportion_shifted_clr_pca` | `proportion_shifted_clr_pca_matrix` | `ProportionShiftedCLR` | stable |
| Dirichlet log | `dirichlet_log_pca` | `dirichlet_log_pca_matrix` | `DirichletLog` | experimental |
| Dirichlet CLR | `dirichlet_clr_pca` | `dirichlet_clr_pca_matrix` | `DirichletCLR` | experimental |
| Correspondence analysis | `correspondence_analysis` | `correspondence_analysis_matrix` | — (one-step only) | stable (`scaled_nb` mode experimental) |

Notes for the implementer:

- **This table does not exist anywhere today.** There is no single place that
  maps a transform to the three ways of calling it. Building it will also
  surface the `shifted_log_pca` gap noted in P1-4.
- Lift the exact formulas from `docs/development/specification.md:181-238`
  (log family, Dirichlet) and `:1147-1266` (Pearson, deviance, and the
  per-model `u_i v_j` table). Convert to `$$` so they actually typeset — see
  P2-1, the spec's own math currently does not render.
- Lift provenance from the `tests/*/README.md` files listed above. Cite the
  papers properly, not just the repos: Townes et al. for deviance/null
  residuals, Lause/Berens/Kobak for Poisson Pearson residuals in scRNA-seq,
  Hafemeister & Satija for SCTransform, Greenacre (or corral / Hsu & Culhane)
  for CA, Aitchison for CLR. **Verify each citation before adding it — do not
  invent DOIs.** If a citation cannot be verified, name the implementation and
  link the pinned test fixture instead; that is still far better than silence.
- Cross-link every entry to `reference/compatibility.md`, which already holds
  the careful non-equivalence statements and should not be duplicated.

---

## P0-2. Collapse `Examples/` into the guides

This is the concrete fix for "hard to see where things are."

### Evidence that Examples is not pulling its weight

- **Orphaned.** Nothing links to it. `grep -rn "examples/" docs/*.md docs/guides
  docs/concepts docs/reference README.md` returns zero hits. It is reachable
  only from the sidebar. Neither `README.md:182-191` nor `docs/index.md:70-80`
  lists it in their documentation indexes.
- **No prose.** All four pages are bare code + shape printouts, with no
  explanation of what is being demonstrated and no link back to the relevant
  guide. `docs/examples/residual_pca_anndata.md` is 42 lines, of which the
  informational content is `scores: (5, 2)`.
- **Duplicative.** Its code is a near-copy of the code already in
  `guides/residual-pca.md:9-19`, `guides/shifted-log-and-clr.md:41-48`, and
  `guides/transform-reuse.md:9-17`. Four separate toy 5×4 CSR matrices appear
  across `getting-started.md` and the three example pages.
- **Generated and gitignored** (`.gitignore` line: `docs/examples/`), so the nav
  points at build artifacts.

### What to do

Keep `examples/*.py` — they are executable, CI-tested, and genuinely valuable
as regression coverage. Change only where their *rendered output* lands:

- Delete the `Examples:` nav section from `mkdocs.yml:61-65`.
- Render each executed script **inline in the guide it belongs to**, via a
  snippet include, so the guide's code block is the CI-tested code:
  `residual_pca_anndata.py` → `guides/residual-pca.md`; `shifted_clr_anndata.py`
  → `guides/shifted-log-and-clr.md`; `transform_reuse.py` →
  `guides/transform-reuse.md`; `correspondence_analysis.py` →
  `guides/correspondence-analysis.md`.
- Adapt `docs/hooks/render_examples.py` to emit include-able fragments rather
  than standalone pages (it already writes only-if-changed; keep that).

This removes a whole nav section, removes the duplicate code paths, and
guarantees the code in the guides is executed in CI — which is stronger than
what either surface offers today.

### On `guides` vs `concepts` specifically

Keep this split; it is the right one (task-oriented vs. explanatory) and both
sets of pages are well written. The boundary does leak in one place worth
tightening: **masking semantics are explained four times** — `README.md:145-151`,
`choosing-a-transform.md:54-62`, `guides/anndata-workflows.md:35-38`,
`guides/residual-pca.md:95-98`, plus the canonical
`concepts/normalization-masking-and-centering.md`. The concept page should own
it; the others should state one sentence and link. This is the package's most
subtle behavior, so the repetition is defensible, but four restatements is more
than it needs and they will drift.

Resulting nav (8 sections → 6):

```
Home · Getting started · Transforms · Choosing a transform ·
Guides (5) · Concepts (3) · Reference · Development
```

---

## P0-3. Answer the Scanpy user's first three questions

### (a) What does this replace in my pipeline?

Scanpy 1.12 ships `sc.experimental.pp.normalize_pearson_residuals_pca` — a
direct functional competitor to `residual_pca` — plus
`normalize_pearson_residuals`, `recipe_pearson_residuals`, and
`highly_variable_genes(flavor="pearson_residuals")`. Scanpy is already a test
dependency (`pyproject.toml:22`, `scanpy>=1.10`).

**None of these are mentioned anywhere in the docs.** The only Scanpy
preprocessing function named at all is `normalize_total`, once, in a negation
(`guides/shifted-log-and-clr.md:29`).

For the target reader this is the *first* question: "Scanpy can already do
Pearson-residual PCA — why this?" Add a short section (in the compatibility
reference, linked prominently from the README and homepage) with a
from → to mapping:

| If you currently call | Use | Note |
| --- | --- | --- |
| `sc.experimental.pp.normalize_pearson_residuals_pca` | `scp.residual_pca` | never materializes the dense residual matrix; also offers deviance residuals and binomial/scaled-NB models |
| `sc.pp.normalize_total` + `sc.pp.log1p` + `sc.pp.pca` | **no equivalent** — see below | |

### (b) State the library-size-normalization limitation plainly

The docs currently say `ShiftedLog` is *not* the `normalize_total` + `log1p`
workflow (`README.md`, `choosing-a-transform.md:10`,
`guides/shifted-log-and-clr.md:26-29`) but **never say that the package has no
library-size-normalized log transform at all.** A reader can easily infer that
one exists elsewhere in the API. It does not — `docs-overhaul.md:414-416`
records this as an open API question.

This is the most likely false expectation a Scanpy user will form. Say it
directly, once, high up: the shifted-log transform is a raw-count transform;
size-factor / library-size-normalized log is not currently offered.

### (c) What do I gain, and what do I do next?

- **No quantitative benefit is stated anywhere.** Deliberately avoiding
  benchmark claims is correct (`docs-overhaul.md:36-38`), but the docs currently
  give the reader *nothing* — not even arithmetic. Add one worked storage
  example to `concepts/sparse-plus-low-rank.md`: for a stated shape and density,
  dense float64 bytes vs. the sparse-plus-low-rank footprint, labeled as
  arithmetic from the storage formula rather than a measured benchmark. That
  satisfies "why should I care" without a single unsupported claim.
  While there: `concepts/sparse-plus-low-rank.md:19` has a literal
  `___dtype___` placeholder in the storage formula.
- **No downstream continuation.** `getting-started.md` ends by printing array
  shapes. Close the loop with the two lines that make it a real workflow —
  `sc.pp.neighbors(adata)` / `sc.tl.umap(adata)` — since the whole point of
  writing Scanpy-compatible keys is that this works.
- **No HVG guidance.** `mask_var` defaults to `adata.var["highly_variable"]`
  (`guides/anndata-workflows.md:22`), but no page tells the user how to produce
  it, or which flavor pairs with which transform. For residual PCA the natural
  pairing is Scanpy's `flavor="pearson_residuals"`.

---

## P1. Substantive gaps

**P1-1. Dirichlet transforms are invisible to users.** `dirichlet_log_pca`,
`dirichlet_clr_pca`, `DirichletLog`, and `DirichletCLR` are exported in
`__all__` but appear in **no** user-facing page — not the README transform
table, not `choosing-a-transform.md`, not any guide. They exist only in
`reference/api/*.md` (as bare `:::` directives) and the normative spec.

Per the author (2026-07-22, in-session): these are experimental and should not
be front and center, **but they do need to be somewhere in the docs — they do
not need a guide.** Implement exactly that:

- add both rows to the P0-1 transforms table, marked *experimental*;
- give them one short section on that page with the formula from
  `specification.md:225-238` and the note that scalar fixed-count shifted CLR is
  the uniform-prior special case `A = G * count_shift` — that relationship is
  genuinely useful and currently buried in the spec;
- **do not** add a guide, a homepage "choose your path" entry, or a
  `choosing-a-transform.md` row.

On the README: the Dirichlet table row and the coverage-sentence mention were
**deliberately removed** in `78f1742`, consistent with the de-emphasis decision.
Do not restore either. There is a residual honesty problem, though: `README.md:17-20`
reads "It covers PCA of model residuals, ... and classical correspondence
analysis," which a reader will take as the complete inventory while two public
transforms and the experimental scaled-NB CA mode are omitted. The author's own
draft did mention both — "a few other experimental methods like Bayesian
(Dirichlet prior) pseudocounts and a scaled-NB variation on correspondence
analysis" (`docs-overhaul.md:93-95`). Resolve with a trailing clause rather than
a table row or a restored list: "…and classical correspondence analysis, plus
experimental prior-based and scaled-NB variants (see Transforms)." That keeps
them out of the spotlight without letting the README imply an exhaustive list.

Also amend `docs-overhaul.md:319-321`: its definition of done still requires
that "a user can choose among residual, shifted-log/CLR, **Dirichlet**, and
correspondence-analysis workflows without reading the specification," which
contradicts the 2026-07-22 de-emphasis decision at `:367`. Reword the DoD to
"can discover and identify" rather than "can choose among."

**P1-2. No troubleshooting page.** The package raises a number of specific,
user-reachable errors and warnings that are documented only in the normative
spec: zero-total gene rejected; `n_comps` must be strictly less than both
dimensions; numerically-zero centered variance; the `clip_max_nnz_ratio`
support-growth refusal; `SparseEfficiencyWarning` on variable-direction
materialization (`guides/transform-reuse.md:41`); the scaled-NB CA experimental
warning; mask-dtype rejection (`guides/anndata-workflows.md:33`). A user who
hits one of these has nowhere user-facing to land. Add a short
"Errors and warnings" reference page: symptom → cause → fix.

**P1-3. No stability/experimental policy in one place.** "Experimental" is
attached to Dirichlet (API pages), scaled-NB CA (`compatibility.md:91-95`,
`guides/correspondence-analysis.md:52-71`), and implicitly to `scaled_nb`
itself, whose name may change before 1.0 (`docs-overhaul.md:417-419`). Collect
this into one short status section — ideally the Status column of the P0-1
table plus a paragraph. Also state the pre-1.0 stability posture; nothing
currently tells a user how much the API may move.

**P1-4. `shifted_log_pca` (the AnnData form) is never shown.**
`guides/shifted-log-and-clr.md` demonstrates `shifted_log_pca_matrix` and
`proportion_shifted_clr_pca_matrix` but only `shifted_clr_pca` in AnnData form
(`:19-20, :41-48, :77-83`). The log family therefore reads as matrix-only while
residual and CLR read as AnnData-first. Make the primary example in each family
the AnnData call, consistently.

**P1-5. No guidance on `n_comps`.** Default is 50 (`_residual_pca.py`), used
throughout without comment. One or two sentences on choosing it, and on the
ARPACK constraint, would help.

---

## P2. Defects and hygiene

**P2-1. The specification's math does not render.** `specification.md` writes
display math in ```` ```math ```` fences, but `pymdownx.arithmatex` only
processes `$$`/`\(`. Verified in the built site: the spec's equations render as
**plain grey code blocks** (`<div class="highlight"><pre><code>Z_{ij}^{CA} =
\frac{1}{\sqrt N}...`) with zero `arithmatex` spans on the page, while
`guides/residual-pca.md` — which uses `$$` — renders correctly. The document
with the most mathematics in the repository has none of it typeset.

Worse, `docs/hooks/validate_math.py` only detects *unprocessed `$$`*, so it
passes cleanly and gives false confidence. Fix: convert `specification.md`'s
```` ```math ```` fences to `$$`, and extend the hook to also fail on a
surviving `language-math` code block. `docs-overhaul.md:345` records math
rendering as done; it is done only for the pages that happened to use `$$`.

**P2-2. Internal planning documents are published to the public site.**
`mkdocs.yml:53-56` uses `not_in_nav` for `history/**` and `wishlist.md`, which
suppresses the nav entry but **still builds and publishes the pages**. Both are
live in `site/`: `site/wishlist/index.html` and
`site/history/robustness-work-brief/`. The wishlist is explicitly "unfinished
plausible additions without treating them as committed API" — exactly the
material that should not be publicly reachable at a guessable URL before
release. Use `exclude_docs` (or move the files out of `docs/`) instead of
`not_in_nav`.

**P2-3. Hook source is published as site content.** `docs/hooks/` lives under
`docs/`, so MkDocs copies it into the build: `site/hooks/render_examples.py`,
`site/hooks/validate_math.py`, and both `__pycache__/*.pyc`. Move the hooks
outside `docs/` (e.g. `scripts/mkdocs_hooks/`) and update `mkdocs.yml:6-8`, or
add them to `exclude_docs`.

**P2-4. README and homepage duplicate ~70% of each other.** `README.md:1-92`
and `docs/index.md:1-68` carry the same opening framing, the same minimal
example, the same sparse-plus-low-rank explanation, and near-identical
documentation indexes. They will drift. Decide the division — README as a short
orientation that funnels to the site; `index.md` as the real homepage — and
keep the transform table in only one of them, with the other linking.

**P2-5. `README.md`'s transform table has the wrong column.** The
"Public specification" column (`README.md:54-60`) shows `Residual(model=...)`,
`ShiftedLog(count_shift=...)` etc. — these are the **two-step** API objects,
introduced 70 lines later at `:127`. A new reader meets the spec objects before
learning they exist. And the CA row's cell reads "separate one-step API," which
is not a specification at all, so the column is not even internally consistent.
Replace with the function the reader would actually call (`residual_pca(...)`),
or adopt the fuller three-column mapping from P0-1.

**P2-6. README quick start is not runnable.** `README.md:34-44` uses a bare
`adata` that is never constructed, while `getting-started.md:21-51` builds a
complete object — the divergence post-dates commit `d462562`, "Make
documentation examples directly runnable." Acceptable if intentional for a
teaser, but worth making a deliberate choice rather than an accident.

**P2-7. `docs-overhaul.md` is at the repository root.** Its own header says to
move it to the private workbench or remove it before publication
(`docs-overhaul.md:3-6`, `:349`). Still outstanding. This review file has the
same status.

**P2-8. Minor.** `concepts/normalization-masking-and-centering.md:68` —
"indepedent". Verify the Booeshaghi v4 date used in four places ("June 22,
2026") against the two other dates circulating in the test fixtures ("June 24,
2026" commit, "June 10, 2026 manuscript revision"); they may all be correct for
different artifacts, but one pass to confirm is worth it. `use_highly_variable`
is a deprecated parameter that appears in rendered API signatures but is
explained nowhere user-facing.

**P2-9. In-progress author comments.** HTML comments remain in
`concepts/sparse-plus-low-rank.md:10,13,25,44` and
`concepts/normalization-masking-and-centering.md:52,72` (the three files
modified in the working tree). Flagged for completeness; the author is already
handling these. `sparse-plus-low-rank.md:10` also has a broken sentence —
"Other normalizations, the popular count sum standardization followed by
`log1p`, map zero to zero" — needing a rewrite, not just comment removal.

---

## What is working well — preserve it

Do not let the volume of findings above suggest a rewrite. These are strengths
and should survive the changes:

- **Public docstrings are genuinely good.** `residual_pca` documents every
  parameter, its orientation, dtype semantics, AnnData side effects, and the
  fit-then-mask ordering. The curated `reference/api/` pages are the right
  approach.
- **`reference/compatibility.md` is the best page in the set.** Separating
  controlled equalities from broad similarities — with the five explicit
  SCTransform conditions — is exactly right and rare.
- **Intellectual honesty throughout.** "Exact" is scoped to the represented
  matrix at a given dtype; no benchmark, citation, or parity claim is invented;
  `scaled_nb` is repeatedly flagged as a package-specific label. Keep this
  discipline when adding the provenance and memory-arithmetic material.
- **`guides/anndata-workflows.md`** cleanly answers count source, mask, keys,
  and copy semantics — the practical questions, in one place.
- **The masking concept page** handles the package's subtlest behavior, and the
  residual-vs-CA ordering rationale (`docs-overhaul.md:369`) is well reasoned.

---

## Suggested implementation order

1. `docs/transforms.md` — the catalogue, with formulas and provenance (P0-1),
   including Dirichlet as experimental reference-only rows (P1-1).
2. Scanpy mapping, the library-size-normalization limitation, HVG note, and
   downstream continuation (P0-3).
3. Collapse `Examples/` into the guides; update `mkdocs.yml` nav and the render
   hook (P0-2).
4. README restructure: transform table pointing at `transforms.md`, fix the
   column, de-duplicate against `index.md` (P2-4, P2-5).
5. Math rendering fix plus hook hardening (P2-1) — do this before or alongside
   step 1, since the new page depends on math actually typesetting.
6. Troubleshooting page and stability/experimental section (P1-2, P1-3).
7. Build hygiene: `exclude_docs`, move hooks out of `docs/` (P2-2, P2-3).
8. Remaining P1/P2 cleanups.

Re-run `mkdocs build --strict`, `ruff check .`, and `pytest` after each of
steps 1–5; the example-rendering change in step 3 touches CI-executed code.
