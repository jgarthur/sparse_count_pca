# Contributing

Contributions should preserve the package's documented mathematical and API
contracts. Begin with the [architecture guide](docs/development/architecture.md)
for source orientation and the
[package specification](docs/development/specification.md) for normative
behavior. The [testing guide](docs/development/testing.md) explains the evidence
expected for changes to transforms, operators, PCA, or AnnData integration.

## Development setup

Install the package and test dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra test --extra docs
```

Run the standard checks:

```bash
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run mkdocs build --strict
```

`ruff format` is the source of truth for layout, so apply it rather than
hand-wrapping:

```bash
uv run ruff format .
```

## Documentation

Serve the site locally while editing:

```bash
uv run mkdocs serve
```

Executed example output is cached from one live-reload build to the next.
Markdown-only edits therefore do not rerun the guide notebooks. Changes to an
example, package source, documentation dependencies, or the rendering hook
invalidate the relevant cache entries. To execute every notebook regardless of
the cache, run:

```bash
DOCS_FORCE_EXAMPLES=1 uv run mkdocs build --strict
```

User guides should explain complete tasks. Concept pages should explain the
mathematics and scientific interpretation. Exact defaults, validation rules,
and edge-case behavior belong in the specification or generated API reference.
Link to detailed explanations instead of copying them between pages.

Every Python file must have a module docstring. Every test function must have a
docstring that describes the behavior it verifies. Public functions, classes,
and result objects use Google-style docstrings and should document shapes,
side effects, important exceptions, and semantic caveats.

## Critical invariants

- A fitted transform owns the sparse support needed to keep later caller
  mutation from changing its values.
- Residual, log, CLR, and Dirichlet normalization state is fitted before the
  PCA-only variable mask is applied.
- PCA column centering occurs after variable selection.
- Correspondence-analysis masks instead define a new contingency table and
  therefore new margins.
- Clipping is exact. Symmetric clipping may expand sparse support; upper
  clipping does not.
- `float64` is the default representation and calculation dtype. Explicit
  `float32` is an approximate lower-memory computation mode.
- Dense-oracle values, operator products, singular values, and subspaces must
  agree within tolerances appropriate to the calculation dtype.
- Public functions name complete, reproducible transforms. Do not expose a
  normalization-construction API of composable shift, divisor, and coordinate
  objects: it would multiply the combinations the package must define,
  validate, and test, and this is a normalization-plus-PCA/SVD tool rather
  than a general normalization framework.

Do not change a documented invariant without updating the specification,
tests, and affected user documentation in the same contribution.
