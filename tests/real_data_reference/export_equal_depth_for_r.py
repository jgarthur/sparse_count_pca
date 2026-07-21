"""Export the equal-depth CSR fixture as a genes-by-cells Matrix Market file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

REFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = REFERENCE_DIR / "pbmc3k_equal_depth_counts.npz"


def main() -> None:
    """Validate and export the pinned fixture in sctransform's orientation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output Matrix Market path")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    counts = sparse.load_npz(args.input).tocsr()
    row_totals = np.asarray(counts.sum(axis=1, dtype=np.int64)).ravel()
    if counts.shape != (256, 1024):
        raise ValueError(f"Unexpected fixture shape: {counts.shape}")
    if not np.array_equal(row_totals, np.full(256, 1000, dtype=np.int64)):
        raise ValueError("The exported fixture must have exactly 1,000 counts per cell")
    if np.asarray(counts.sum(axis=0)).ravel().min() <= 0:
        raise ValueError("The exported fixture contains an empty gene")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mmwrite(args.output, counts.T, field="integer", symmetry="general")


if __name__ == "__main__":
    main()
