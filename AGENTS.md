## Python documentation requirements

- Every Python file must have a file-level module docstring.
- Every test function must have a docstring that describes the behavior it verifies.
- Write a comment only to state a constraint the code cannot show. Justifying a
  change, explaining why a test was trimmed, or naming where coverage moved to
  belongs in the commit message, not the source.

## Documentation prose

- Docs other than `CHANGELOG.md` describe current behavior. Before/after
  comparisons, measurements tied to a commit or date, and words like "now",
  "existing", or "in this change" belong in the commit message.
