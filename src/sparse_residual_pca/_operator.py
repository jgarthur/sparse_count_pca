from __future__ import annotations

import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ._representation import SparseLowRankMatrix

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _normalize_operator_dtype(dtype: DTypeLike) -> np.dtype[np.floating[Any]]:
    """Normalize and validate the supported representation dtypes."""
    try:
        result = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("dtype must be float32 or float64") from error
    if result not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("dtype must be float32 or float64")
    return result


class SparseLowRankLinearOperator(LinearOperator):
    """Represent a sparse-plus-low-rank matrix with optional column centering.

    The represented matrix is ``S + U V.T - 1 mean.T`` when centered and
    ``S + U V.T`` otherwise.

    Args:
        representation: Sparse-plus-low-rank representation.
        center: Whether to subtract column means in matrix products.
        dtype: Storage and output floating-point dtype.

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
    ) -> None:
        """Initialize the operator and precompute float64 summary statistics.

        Raises:
            ValueError: If the dtype is unsupported.
        """
        operator_dtype = _normalize_operator_dtype(dtype)

        self.S = representation.sparse.astype(operator_dtype, copy=True).tocsr()
        self.left = np.asarray(representation.left, dtype=operator_dtype)
        self.right = np.asarray(representation.right, dtype=operator_dtype)
        self.center = bool(center)
        super().__init__(dtype=operator_dtype, shape=self.S.shape)

        self._S_csc = self.S.tocsc()
        self._left_float64 = self.left.astype(np.float64)
        self._right_float64 = self.right.astype(np.float64)
        self._left_mean_float64 = np.array(
            [math.fsum(column) / self.shape[0] for column in self._left_float64.T]
        )
        left_deviations = self._left_float64 - self._left_mean_float64
        self._left_centered_gram_float64 = left_deviations.T @ left_deviations
        self._left_sum_float64 = np.array(
            [math.fsum(column) for column in self._left_float64.T]
        )
        self._mean_float64 = self._stable_column_means()

        if self.center:
            self.mean = self._mean_float64.astype(operator_dtype)
        else:
            self.mean = None

        zero = np.zeros(self.shape[1], dtype=operator_dtype)
        self._frobenius_squared_uncentered_float64 = self._squared_norm_about(zero)
        if self.center:
            assert self.mean is not None
            self._frobenius_squared_centered_float64 = self._squared_norm_about(
                self.mean
            )
        else:
            self._frobenius_squared_centered_float64 = (
                self._frobenius_squared_uncentered_float64
            )

        del self._S_csc

    def _stored_column_values(
        self, column: int, rows: NDArray[np.int32] | NDArray[np.int64]
    ) -> FloatArray:
        """Evaluate stored-support values using the operator dtype."""
        start, stop = self._S_csc.indptr[column : column + 2]
        baseline = self.left[rows] @ self.right[column]
        return np.asarray(baseline + self._S_csc.data[start:stop], dtype=self.dtype)

    def _stable_column_means(self) -> NDArray[np.float64]:
        """Calculate means without cancelling dense support corrections."""
        n_obs, n_vars = self.shape
        means = np.empty(n_vars, dtype=np.float64)
        for column in range(n_vars):
            start, stop = self._S_csc.indptr[column : column + 2]
            rows = self._S_csc.indices[start:stop]
            if rows.size > n_obs // 2:
                values = self.left @ self.right[column]
                values = values.copy()
                values[rows] += self._S_csc.data[start:stop]
                means[column] = math.fsum(values.astype(np.float64)) / n_obs
            else:
                sparse_sum = math.fsum(
                    self._S_csc.data[start:stop].astype(np.float64)
                )
                baseline_sum = float(
                    self._left_sum_float64 @ self._right_float64[column]
                )
                means[column] = math.fsum((baseline_sum, sparse_sum)) / n_obs
        return means

    def _squared_norm_about(self, center: FloatArray) -> float:
        """Return ``sum((S + U V.T - center)**2)`` without cancellation."""
        n_obs, n_vars = self.shape
        center = np.asarray(center, dtype=self.dtype)
        column_norms: list[float] = []
        for column in range(n_vars):
            start, stop = self._S_csc.indptr[column : column + 2]
            rows = self._S_csc.indices[start:stop]
            if rows.size > n_obs // 2:
                values = self.left @ self.right[column]
                values = values.copy()
                values[rows] += self._S_csc.data[start:stop]
                deviations = (values - center[column]).astype(np.float64)
                column_norms.append(math.fsum(deviations * deviations))
                continue

            v = self._right_float64[column]
            offset = float(v @ self._left_mean_float64) - float(center[column])
            baseline_total = float(
                v @ self._left_centered_gram_float64 @ v
                + n_obs * offset * offset
            )
            baseline_support = self._left_float64[rows] @ v - float(center[column])
            actual_support = (
                self._stored_column_values(column, rows) - center[column]
            ).astype(np.float64)
            column_squared = math.fsum(
                (
                    baseline_total,
                    -math.fsum(baseline_support * baseline_support),
                    math.fsum(actual_support * actual_support),
                )
            )
            rounding_bound = (
                np.finfo(np.float64).eps
                * max(baseline_total, 1.0)
                * max(rows.size, 1)
            )
            if column_squared < -rounding_bound:
                raise ArithmeticError("stable squared-norm calculation became negative")
            column_norms.append(max(column_squared, 0.0))
        return math.fsum(column_norms)

    def _matvec(self, z: ArrayLike) -> FloatArray:
        """Multiply the represented matrix by a vector.

        Args:
            z: Vector with length ``n_vars``.

        Returns:
            A vector with length ``n_obs`` in the operator dtype.
        """
        z = np.asarray(z, dtype=self.dtype)
        result = self.S @ z + self.left @ (self.right.T @ z)
        if self.center:
            assert self.mean is not None
            result = result - np.dot(self.mean, z)
        return np.asarray(result, dtype=self.dtype)

    def _rmatvec(self, y: ArrayLike) -> FloatArray:
        """Multiply the transpose of the represented matrix by a vector.

        Args:
            y: Vector with length ``n_obs``.

        Returns:
            A vector with length ``n_vars`` in the operator dtype.
        """
        y = np.asarray(y, dtype=self.dtype)
        result = self.S.T @ y + self.right @ (self.left.T @ y)
        if self.center:
            assert self.mean is not None
            result = result - self.mean * np.sum(y, dtype=self.dtype)
        return np.asarray(result, dtype=self.dtype)

    def _matmat(self, Z: ArrayLike) -> FloatArray:
        """Multiply the represented matrix by a dense matrix.

        Args:
            Z: Matrix with ``n_vars`` rows.

        Returns:
            A matrix with ``n_obs`` rows in the operator dtype.
        """
        Z = np.asarray(Z, dtype=self.dtype)
        result = self.S @ Z + self.left @ (self.right.T @ Z)
        if self.center:
            assert self.mean is not None
            result = result - (self.mean @ Z)[None, :]
        return np.asarray(result, dtype=self.dtype)

    def _rmatmat(self, Y: ArrayLike) -> FloatArray:
        """Multiply the transpose by a dense matrix.

        Args:
            Y: Matrix with ``n_obs`` rows.

        Returns:
            A matrix with ``n_vars`` rows in the operator dtype.
        """
        Y = np.asarray(Y, dtype=self.dtype)
        result = self.S.T @ Y + self.right @ (self.left.T @ Y)
        if self.center:
            assert self.mean is not None
            result = (
                result
                - self.mean[:, None] * np.sum(Y, axis=0, dtype=self.dtype)[None, :]
            )
        return np.asarray(result, dtype=self.dtype)

    def mean_axis0(self) -> FloatArray:
        """Return uncentered residual column means in the operator dtype."""
        if self.center:
            assert self.mean is not None
            return self.mean.copy()
        return self._mean_float64.astype(self.dtype)

    @property
    def u(self) -> FloatArray:
        """Return the left factor for rank-one compatibility."""
        if self.left.shape[1] != 1:
            raise AttributeError("u is only defined for rank-one representations")
        return self.left[:, 0]

    @property
    def v(self) -> FloatArray:
        """Return the right factor for rank-one compatibility."""
        if self.right.shape[1] != 1:
            raise AttributeError("v is only defined for rank-one representations")
        return self.right[:, 0]

    def frobenius_squared_uncentered(self) -> float:
        """Return the squared Frobenius norm before column centering."""
        return self._frobenius_squared_uncentered_float64

    def frobenius_squared_centered(self) -> float:
        """Return the squared Frobenius norm after column centering."""
        return self._frobenius_squared_centered_float64

    def centered_variance_is_numerically_zero(self) -> bool:
        """Return whether centered variation is indistinguishable from rounding."""
        centered = self.frobenius_squared_centered()
        uncentered = self.frobenius_squared_uncentered()
        eps = np.finfo(self.dtype).eps
        tolerance = eps * eps * np.prod(self.shape) * uncentered
        return centered <= tolerance


class ResidualLinearOperator(SparseLowRankLinearOperator):
    """Backward-compatible rank-one residual operator."""

    def __init__(
        self,
        S: sparse.spmatrix | sparse.sparray,
        u: ArrayLike,
        v: ArrayLike,
        *,
        center: bool = True,
        dtype: DTypeLike = "float64",
    ) -> None:
        super().__init__(
            SparseLowRankMatrix(S, u, v), center=center, dtype=dtype
        )
