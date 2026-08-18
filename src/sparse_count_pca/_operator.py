"""Linear operator and bounded statistics for implicit transformed matrices."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ._representation import SparseLowRankMatrix
from ._sparse import _support_row_blocks

FloatArray: TypeAlias = NDArray[np.floating[Any]]

_STATS_MEAN_BLOCK_NNZ = 1_000_000
# The norm sweep materializes factor values and deviations for each stored
# entry in the current row block, so its smaller target bounds scratch memory.
_STATS_NORM_BLOCK_NNZ = 100_000


def _normalize_operator_dtype(dtype: DTypeLike) -> np.dtype[np.floating[Any]]:
    """Normalize and validate the supported representation dtypes."""
    try:
        result = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("dtype must be float32 or float64") from error
    if result not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("dtype must be float32 or float64")
    return result


def _squared_norm_is_numerically_zero(
    value: float,
    *,
    scale: float,
    shape: tuple[int, int],
    dtype: DTypeLike,
) -> bool:
    """Compare a squared norm with the package's roundoff-scale boundary."""
    eps = np.finfo(_normalize_operator_dtype(dtype)).eps
    tolerance = eps * eps * math.prod(shape) * scale
    return value <= tolerance


class SparseLowRankLinearOperator(LinearOperator):
    """Represent a sparse-plus-low-rank matrix with optional column centering.

    The represented matrix is ``S + U V.T - 1 mean.T`` when centered and
    ``S + U V.T`` otherwise.

    Args:
        representation: Sparse-plus-low-rank representation.
        center: Whether to subtract column means in matrix products.
        dtype: Storage and output floating-point dtype.
        copy: Whether to isolate stored arrays from the representation.

    Attributes:
        S: CSR sparse part stored in the operator dtype.
        left: Left factor stored in the operator dtype.
        right: Right factor stored in the operator dtype.
        mean: Column means in the operator dtype when centered.
        center: Whether matrix products apply column centering.
    """

    S: sparse.csr_matrix
    left: FloatArray
    right: FloatArray
    mean: FloatArray | None
    center: bool

    def __init__(
        self,
        representation: SparseLowRankMatrix,
        *,
        center: bool = True,
        dtype: DTypeLike = "float64",
        copy: bool = True,
    ) -> None:
        """Initialize the operator and precompute float64 summary statistics.

        Raises:
            ValueError: If the dtype is unsupported.
        """
        operator_dtype = _normalize_operator_dtype(dtype)

        self.S = representation.sparse.astype(operator_dtype, copy=copy).tocsr(
            copy=False
        )
        self.left = representation.left.astype(operator_dtype, copy=copy)
        self.right = representation.right.astype(operator_dtype, copy=copy)
        self.center = bool(center)
        super().__init__(dtype=operator_dtype, shape=self.S.shape)

        self._left_float64 = self.left.astype(np.float64, copy=False)
        self._right_float64 = self.right.astype(np.float64, copy=False)
        self._left_sum_float64 = np.array(
            [math.fsum(column) for column in self._left_float64.T]
        )
        self._left_mean_float64 = self._left_sum_float64 / self.shape[0]
        left_deviations = self._left_float64 - self._left_mean_float64
        self._left_centered_gram_float64 = left_deviations.T @ left_deviations

        column_counts = np.bincount(self.S.indices, minlength=self.shape[1])
        dense_columns = column_counts > self.shape[0] // 2

        if self.center:
            self._mean_float64 = self._column_means(dense_columns)
            self.mean = self._mean_float64.astype(operator_dtype)
        else:
            self.mean = None

        zero = np.zeros(self.shape[1], dtype=operator_dtype)
        if self.center:
            assert self.mean is not None
            uncentered, centered = self._squared_norms_about(
                (zero, self.mean),
                column_counts,
                dense_columns,
            )
        else:
            (uncentered,) = self._squared_norms_about(
                (zero,),
                column_counts,
                dense_columns,
            )
            centered = uncentered
        self._frobenius_squared_uncentered_float64 = uncentered
        self._frobenius_squared_centered_float64 = centered

    def _csc_row_block(
        self,
        row_start: int,
        row_stop: int,
        value_start: int,
        value_stop: int,
    ) -> sparse.csc_matrix:
        """Group one borrowed CSR row block by column.

        A matrix that fits in one block takes the direct whole-matrix shortcut,
        so dense-column handling may allocate one full CSC copy.
        """
        if row_start == 0 and row_stop == self.shape[0]:
            return self.S.tocsc()
        indptr = self.S.indptr[row_start : row_stop + 1] - value_start
        block = sparse.csr_matrix(
            (
                self.S.data[value_start:value_stop],
                self.S.indices[value_start:value_stop],
                indptr,
            ),
            shape=(row_stop - row_start, self.shape[1]),
            copy=False,
        )
        return block.tocsc()

    def _mean_block_sums(
        self,
        row_start: int,
        row_stop: int,
        value_start: int,
        value_stop: int,
        dense_columns: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return one row block's per-column mean numerators."""
        columns = self.S.indices[value_start:value_stop]
        partial = np.bincount(
            columns,
            weights=self.S.data[value_start:value_stop].astype(
                np.float64,
                copy=False,
            ),
            minlength=self.shape[1],
        )
        dense_column_indices = np.flatnonzero(dense_columns)
        if dense_column_indices.size:
            block = self._csc_row_block(
                row_start,
                row_stop,
                value_start,
                value_stop,
            )
            left = self.left[row_start:row_stop]
            for column in dense_column_indices:
                begin, end = block.indptr[column : column + 2]
                rows = block.indices[begin:end]
                values = left @ self.right[column]
                values[rows] += block.data[begin:end]
                partial[column] = math.fsum(values.astype(np.float64, copy=False))
        return partial

    def _dense_block_squared_sums(
        self,
        row_start: int,
        row_stop: int,
        value_start: int,
        value_stop: int,
        centers: Sequence[FloatArray],
        dense_column_indices: NDArray[np.intp],
    ) -> NDArray[np.float64]:
        """Return direct squared sums for dense columns in one row block."""
        block = self._csc_row_block(
            row_start,
            row_stop,
            value_start,
            value_stop,
        )
        partial = np.empty(
            (len(centers), dense_column_indices.size),
            dtype=np.float64,
        )
        left = self.left[row_start:row_stop]
        for output_column, column in enumerate(dense_column_indices):
            begin, end = block.indptr[column : column + 2]
            rows = block.indices[begin:end]
            values = left @ self.right[column]
            values[rows] += block.data[begin:end]
            for center_index, center in enumerate(centers):
                deviations = (values - center[column]).astype(
                    np.float64,
                    copy=False,
                )
                partial[center_index, output_column] = math.fsum(
                    deviations * deviations
                )
        return partial

    def _column_means(
        self,
        dense_columns: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Calculate column means in bounded row-major passes."""
        n_obs, n_vars = self.shape
        total = np.zeros(n_vars, dtype=np.float64)

        for (
            row_start,
            row_stop,
            value_start,
            value_stop,
        ) in _support_row_blocks(self.S, _STATS_MEAN_BLOCK_NNZ):
            partial = self._mean_block_sums(
                row_start,
                row_stop,
                value_start,
                value_stop,
                dense_columns,
            )
            total += partial

        baseline_sums = self._right_float64 @ self._left_sum_float64
        means = (baseline_sums + total) / n_obs
        means[dense_columns] = total[dense_columns] / n_obs
        return means

    def _norm_block_squared_sums(
        self,
        row_start: int,
        row_stop: int,
        value_start: int,
        value_stop: int,
        centers: Sequence[FloatArray],
        dense_columns: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return one row block's removed and added support squared sums."""
        n_vars = self.shape[1]
        shape = (len(centers), n_vars)
        removed_partial = np.zeros(shape, dtype=np.float64)
        added_partial = np.zeros(shape, dtype=np.float64)
        row_counts = np.diff(self.S.indptr[row_start : row_stop + 1])
        rows = np.repeat(
            np.arange(row_start, row_stop, dtype=np.intp),
            row_counts,
        )
        columns = self.S.indices[value_start:value_stop]
        stored = self.S.data[value_start:value_stop]

        baseline_float64 = np.einsum(
            "ij,ij->i",
            self._left_float64[rows],
            self._right_float64[columns],
        )
        if self.dtype == np.dtype(np.float64):
            baseline = baseline_float64
        else:
            baseline = np.einsum(
                "ij,ij->i",
                self.left[rows],
                self.right[columns],
            )
        actual = np.asarray(baseline + stored, dtype=self.dtype)

        sparse_support = ~dense_columns[columns]
        sparse_columns = columns[sparse_support]
        sparse_baseline = baseline_float64[sparse_support]
        sparse_actual = actual[sparse_support]

        for index, center in enumerate(centers):
            baseline_deviations = sparse_baseline - center[sparse_columns].astype(
                np.float64,
                copy=False,
            )
            actual_deviations = (sparse_actual - center[sparse_columns]).astype(
                np.float64,
                copy=False,
            )
            removed_partial[index] = np.bincount(
                sparse_columns,
                weights=baseline_deviations * baseline_deviations,
                minlength=n_vars,
            )
            added_partial[index] = np.bincount(
                sparse_columns,
                weights=actual_deviations * actual_deviations,
                minlength=n_vars,
            )

        dense_column_indices = np.flatnonzero(dense_columns)
        if dense_column_indices.size:
            added_partial[:, dense_column_indices] = self._dense_block_squared_sums(
                row_start,
                row_stop,
                value_start,
                value_stop,
                centers,
                dense_column_indices,
            )
        return removed_partial, added_partial

    def _squared_norms_about(
        self,
        centers: Sequence[FloatArray],
        column_counts: NDArray[np.integer[Any]],
        dense_columns: NDArray[np.bool_],
    ) -> list[float]:
        """Return stable squared norms about each center in one CSR pass.

        Implements the support-replacement identities from the specification
        section "Explained variance and total variance"
        (``docs/development/specification.md``).
        """
        n_obs, n_vars = self.shape
        cast = [np.asarray(center, dtype=self.dtype) for center in centers]
        shape = (len(cast), n_vars)
        removed_total = np.zeros(shape, dtype=np.float64)
        added_total = np.zeros(shape, dtype=np.float64)

        for block_bounds in _support_row_blocks(self.S, _STATS_NORM_BLOCK_NNZ):
            removed_partial, added_partial = self._norm_block_squared_sums(
                *block_bounds,
                cast,
                dense_columns,
            )
            removed_total += removed_partial
            added_total += added_partial

        column_norms = np.empty(shape, dtype=np.float64)
        for column in range(n_vars):
            if dense_columns[column]:
                column_norms[:, column] = added_total[:, column]
                continue

            v = self._right_float64[column]
            mean_projection = float(v @ self._left_mean_float64)
            baseline_quadratic = v @ self._left_centered_gram_float64 @ v
            for index, center in enumerate(cast):
                offset = mean_projection - float(center[column])
                baseline_total = float(baseline_quadratic + n_obs * offset * offset)
                column_squared = math.fsum(
                    (
                        baseline_total,
                        -removed_total[index, column],
                        added_total[index, column],
                    )
                )
                rounding_bound = (
                    np.finfo(np.float64).eps
                    * max(baseline_total, 1.0)
                    * max(int(column_counts[column]), 1)
                )
                if column_squared < -rounding_bound:
                    raise ArithmeticError(
                        "stable squared-norm calculation became negative"
                    )
                column_norms[index, column] = max(column_squared, 0.0)
        return [math.fsum(norms) for norms in column_norms]

    def _matvec(self, z: ArrayLike) -> FloatArray:
        """Multiply the represented matrix by a vector.

        Args:
            z: Vector with length ``n_vars``.

        Returns:
            A vector with length ``n_obs`` in the operator dtype.
        """
        z = np.asarray(z, dtype=self.dtype)
        result = self.S @ z
        if self.right.shape[1]:
            result += self.left @ (self.right.T @ z)
        if self.center:
            assert self.mean is not None
            result -= np.dot(self.mean, z)
        return np.asarray(result, dtype=self.dtype)

    def _rmatvec(self, y: ArrayLike) -> FloatArray:
        """Multiply the transpose of the represented matrix by a vector.

        Args:
            y: Vector with length ``n_obs``.

        Returns:
            A vector with length ``n_vars`` in the operator dtype.
        """
        y = np.asarray(y, dtype=self.dtype)
        result = self.S.T @ y
        if self.right.shape[1]:
            result += self.right @ (self.left.T @ y)
        if self.center:
            assert self.mean is not None
            result -= self.mean * np.sum(y, dtype=self.dtype)
        return np.asarray(result, dtype=self.dtype)

    def _matmat(self, Z: ArrayLike) -> FloatArray:
        """Multiply the represented matrix by a dense matrix.

        Args:
            Z: Matrix with ``n_vars`` rows.

        Returns:
            A matrix with ``n_obs`` rows in the operator dtype.
        """
        Z = np.asarray(Z, dtype=self.dtype)
        result = self.S @ Z
        if self.right.shape[1]:
            result += self.left @ (self.right.T @ Z)
        if self.center:
            assert self.mean is not None
            result -= (self.mean @ Z)[None, :]
        return np.asarray(result, dtype=self.dtype)

    def _rmatmat(self, Y: ArrayLike) -> FloatArray:
        """Multiply the transpose by a dense matrix.

        Args:
            Y: Matrix with ``n_obs`` rows.

        Returns:
            A matrix with ``n_vars`` rows in the operator dtype.
        """
        Y = np.asarray(Y, dtype=self.dtype)
        result = self.S.T @ Y
        if self.right.shape[1]:
            result += self.right @ (self.left.T @ Y)
        if self.center:
            assert self.mean is not None
            result -= self.mean[:, None] * np.sum(Y, axis=0, dtype=self.dtype)[None, :]
        return np.asarray(result, dtype=self.dtype)

    def frobenius_squared_uncentered(self) -> float:
        """Return the squared Frobenius norm before column centering."""
        return self._frobenius_squared_uncentered_float64

    def frobenius_squared_centered(self) -> float:
        """Return the squared Frobenius norm after column centering."""
        return self._frobenius_squared_centered_float64

    def centered_variance_is_numerically_zero(self) -> bool:
        """Return whether centered variation is indistinguishable from rounding."""
        return _squared_norm_is_numerically_zero(
            self.frobenius_squared_centered(),
            scale=self.frobenius_squared_uncentered(),
            shape=self.shape,
            dtype=self.dtype,
        )
