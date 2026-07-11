from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.utils.extmath import svd_flip

Solver: TypeAlias = Literal["arpack"]
FloatArray: TypeAlias = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class TruncatedSVDResult:
    """Analysis-neutral truncated singular value decomposition."""

    left_vectors: FloatArray
    singular_values: NDArray[np.float64]
    right_vectors: FloatArray


def compute_truncated_svd(
    A: LinearOperator,
    n_comps: int,
    *,
    solver: Solver,
    random_state: int | None,
    tol: float,
) -> TruncatedSVDResult:
    """Compute a deterministic, sign-normalized truncated SVD.

    Args:
        A: Matrix or linear operator to decompose.
        n_comps: Number of singular triplets to return.
        solver: SVD solver name. Version 1 supports only ``"arpack"``.
        random_state: Seed for the ARPACK starting vector.
        tol: Convergence tolerance passed to SciPy.

    Returns:
        A tuple ``(U, singular_values, Vt)`` sorted by descending singular
        value and sign-flipped using scikit-learn's PCA convention.

    Raises:
        ValueError: If ``n_comps`` is invalid for ARPACK.
        NotImplementedError: If a solver other than ARPACK is requested.
    """
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    if not 1 <= n_comps < min(A.shape):
        raise ValueError(
            "n_comps must satisfy 1 <= n_comps < min(n_obs, n_vars_used) "
            "when solver='arpack'"
        )

    rng = np.random.default_rng(random_state)
    v0 = rng.standard_normal(min(A.shape))
    U, singular_values, Vt = svds(
        A,
        k=n_comps,
        solver="arpack",
        tol=tol,
        v0=v0,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    U = U[:, order]
    Vt = Vt[order, :]
    U, Vt = svd_flip(U, Vt, u_based_decision=False)
    return TruncatedSVDResult(
        left_vectors=U,
        singular_values=singular_values.astype(np.float64, copy=False),
        right_vectors=Vt,
    )


def compute_svd(
    A: LinearOperator,
    n_comps: int,
    *,
    solver: Solver,
    random_state: int | None,
    tol: float,
) -> tuple[FloatArray, NDArray[np.float64], FloatArray]:
    """Backward-compatible tuple interface for truncated SVD."""
    result = compute_truncated_svd(
        A,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    return result.left_vectors, result.singular_values, result.right_vectors
