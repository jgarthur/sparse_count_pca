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

## Changelog prose

- Lead with the user-visible outcome and name the affected functionality.
  Explain the implementation only when it helps the reader understand the impact.
- Introduce new options in the context of the workflow and benefit they support.
  Keep necessary caveats brief; detailed contracts belong in guides or the spec.
- Numerical claims need identifiable evidence. Distinguish synthetic regression
  tests, benchmarks, and real-data measurements; do not imply a measured result
  applies to datasets or scales that were not tested.
- Separate fixes from performance improvements and numerical side effects.
  Summation-order differences belong under other changes, not fixes.
