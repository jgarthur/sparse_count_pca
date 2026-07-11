from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

FloatArray: TypeAlias = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class SparseLowRankMatrix:
    """Represent a matrix as ``sparse + left @ right.T``."""

    sparse: sparse.csr_matrix
    left: FloatArray
    right: FloatArray

    def __init__(
        self,
        sparse_part: sparse.spmatrix | sparse.sparray,
        left: ArrayLike,
        right: ArrayLike,
    ) -> None:
        sparse_csr = sparse_part.tocsr(copy=True)
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.ndim == 1:
            left_array = left_array[:, None]
        if right_array.ndim == 1:
            right_array = right_array[:, None]
        if left_array.ndim != 2 or right_array.ndim != 2:
            raise ValueError("left and right factors must be one- or two-dimensional")
        if sparse_csr.shape != (left_array.shape[0], right_array.shape[0]):
            raise ValueError("sparse, left, and right factors have incompatible shapes")
        if left_array.shape[1] != right_array.shape[1]:
            raise ValueError("left and right factors must have the same rank")
        object.__setattr__(self, "sparse", sparse_csr)
        object.__setattr__(self, "left", left_array)
        object.__setattr__(self, "right", right_array)

    @property
    def shape(self) -> tuple[int, int]:
        return self.sparse.shape

    @property
    def rank(self) -> int:
        return self.left.shape[1]

    def select_columns(self, mask: ArrayLike) -> SparseLowRankMatrix:
        columns = np.asarray(mask)
        return SparseLowRankMatrix(
            self.sparse[:, columns].tocsr(), self.left, self.right[columns]
        )

    def scale_rows(self, weights: ArrayLike) -> SparseLowRankMatrix:
        weights_array = np.asarray(weights)
        if weights_array.shape != (self.shape[0],):
            raise ValueError("row weights must have shape (n_obs,)")
        return SparseLowRankMatrix(
            self.sparse.multiply(weights_array[:, None]).tocsr(),
            self.left * weights_array[:, None],
            self.right,
        )

    def scale_columns(self, weights: ArrayLike) -> SparseLowRankMatrix:
        weights_array = np.asarray(weights)
        if weights_array.shape != (self.shape[1],):
            raise ValueError("column weights must have shape (n_vars,)")
        return SparseLowRankMatrix(
            self.sparse.multiply(weights_array[None, :]).tocsr(),
            self.left,
            self.right * weights_array[:, None],
        )

    def scaled(self, value: float) -> SparseLowRankMatrix:
        return SparseLowRankMatrix(
            self.sparse * value, self.left * value, self.right
        )
