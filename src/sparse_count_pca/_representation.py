"""Sparse-plus-low-rank matrix representation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from ._sparse import _csr_with_data

FloatArray: TypeAlias = NDArray[np.floating[Any]]
SupportOwnership: TypeAlias = Literal["owned", "shared", "transient", "handoff"]


def _borrows_input_support(ownership: SupportOwnership) -> bool:
    """Return whether builders may borrow the count matrix's CSR support.

    ``"owned"`` is the public ``transform`` default and ``"shared"`` is
    ``transform(share_structure=True)``. A one-step analysis is ``"transient"``
    when it does not return its operator, so nothing built from the borrowed
    support outlives the call, and ``"handoff"`` when it does, so its
    representation owns support that the returned operator takes over.
    """
    return ownership in ("shared", "transient")


def _copies_returned_operator(ownership: SupportOwnership) -> bool:
    """Return whether an operator returned by PCA must copy its arrays.

    A public ``TransformedMatrix`` stays live after ``pca``, so its returned
    operator must not share its arrays. A ``"handoff"`` representation belongs
    to one one-step call and is passed on without a copy. A ``"transient"``
    representation never returns its operator.
    """
    return ownership in ("owned", "shared")


def _one_step_ownership(return_operator: bool) -> SupportOwnership:
    """Choose the ownership of a one-step analysis's private representation."""
    return "handoff" if return_operator else "transient"


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
        # Builders hand off newly allocated data and treat the representation as
        # immutable, so an existing CSR can be borrowed without another full
        # data/index/indptr copy. Other sparse formats still allocate on conversion.
        sparse_csr = sparse_part.tocsr(copy=False)
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

    def scaled(
        self, value: float, *, share_structure: bool = False
    ) -> SparseLowRankMatrix:
        """Multiply by a scalar, optionally borrowing the unchanged CSR support."""
        return SparseLowRankMatrix(
            _csr_with_data(
                self.sparse,
                self.sparse.data * value,
                share_structure=share_structure,
            ),
            self.left * value,
            self.right,
        )
