"""Shared PCA execution for sparse-plus-low-rank representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._operator import SparseLowRankLinearOperator, _normalize_operator_dtype
from ._representation import SparseLowRankMatrix
from ._svd import Solver, compute_truncated_svd
from ._version import __version__

FloatArray: TypeAlias = NDArray[np.floating[Any]]


@dataclass
class PCAResult:
    """Outputs from PCA of an implicit sparse-plus-low-rank transform.

    Attributes:
        scores: Observation coordinates with shape ``(n_obs, n_comps)``, equal
            to the centered transform projected onto ``components``, that is the
            left singular vectors scaled by ``singular_values``.
        components: Unit-norm right singular vectors with shape
            ``(n_comps, n_vars_used)``, one component per row, over the
            variables selected for PCA.
        singular_values: Singular values of the centered transform, in
            descending order.
        explained_variance: Sample variance carried by each returned component,
            equal to its squared singular value divided by ``n_obs - 1``.
        explained_variance_ratio: ``explained_variance / total_variance``, the
            fraction of total centered variance carried by each returned
            component. Entries sum to one only if every component is returned.
        mean: Column mean of the selected uncentered transform, the vector
            subtracted before decomposition.
        total_variance: Total sample variance of the centered transform across
            all selected variables, not only the returned components. It is the
            denominator of ``explained_variance_ratio``.
        params: Transform, solver, dtype, masking, and reproducibility metadata.
        operator: Centered operator passed to ARPACK when
            ``return_operator=True``; otherwise ``None``.
    """

    scores: FloatArray
    components: FloatArray
    singular_values: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    mean: FloatArray
    total_variance: float
    params: dict[str, Any]
    operator: SparseLowRankLinearOperator | None = None


def _count_nonempty_columns(representation: SparseLowRankMatrix) -> int:
    """Count columns that are structurally, not numerically, nonzero.

    A column counts as nonempty when it has stored sparse support or a nonzero
    low-rank factor. Exact cancellation between the two terms is not detected,
    so this is an upper bound on the rank rather than a rank estimate. It is
    used only to stop empty variables from admitting extra components.
    """
    nonzero = np.zeros(representation.shape[1], dtype=bool)
    sparse_part = representation.sparse
    if sparse_part.nnz:
        nonzero[sparse_part.indices] = True
    right = representation.right
    if right.shape[1]:
        nonzero |= np.any(right != 0.0, axis=1)
    return int(nonzero.sum())


def compute_pca_from_representation(
    representation: SparseLowRankMatrix,
    n_comps: int,
    *,
    transform_label: str,
    params: dict[str, Any],
    check_values: bool,
    dtype: DTypeLike,
    solver: Solver,
    random_state: int | None,
    tol: float,
    return_operator: bool,
    copy_operator: bool,
) -> PCAResult:
    """Center and decompose an already validated implicit transform."""
    if solver != "arpack":
        raise NotImplementedError("Only solver='arpack' is supported in v1")
    operator_dtype = _normalize_operator_dtype(dtype)
    n_obs, n_selected = representation.shape
    # A column that is identically zero carries no variance and cannot support
    # a component. This is a dimension rule about empty variables, not a rank
    # estimate: it does not detect columns that are only numerically zero.
    n_nonempty = _count_nonempty_columns(representation)
    if not 1 <= n_comps < min(n_obs, n_nonempty):
        n_zero = n_selected - n_nonempty
        detail = (
            ""
            if not n_zero
            else (
                f"; n_nonempty_vars excludes {n_zero} of the {n_selected} "
                "selected variables that are identically zero and carry no "
                "variance"
            )
        )
        raise ValueError(
            "n_comps must satisfy 1 <= n_comps < "
            f"min(n_obs={n_obs}, n_nonempty_vars={n_nonempty}) when "
            f"solver='arpack'; got n_comps={n_comps}{detail}"
        )

    operator = SparseLowRankLinearOperator(
        representation,
        center=True,
        dtype=operator_dtype,
        copy=copy_operator,
    )
    centered_squared = operator.frobenius_squared_centered()
    if operator.centered_variance_is_numerically_zero():
        raise ValueError(
            f"{transform_label} matrix has numerically zero centered variance, "
            "so PCA directions are undefined. Every observation has the same "
            "transformed values across the selected variables."
        )

    decomposition = compute_truncated_svd(
        operator,
        n_comps,
        solver=solver,
        random_state=random_state,
        tol=tol,
    )
    singular_values = decomposition.singular_values
    scores = np.asarray(
        decomposition.left_vectors * singular_values, dtype=operator_dtype
    )
    components = np.asarray(decomposition.right_vectors, dtype=operator_dtype)
    total_variance = centered_squared / (n_obs - 1)
    explained_variance = singular_values**2 / (n_obs - 1)
    result_params = {
        **params,
        "zero_center": True,
        "n_comps": n_comps,
        "pca_n_vars": representation.shape[1],
        "solver": solver,
        "random_state": random_state,
        "tol": tol,
        "check_values": check_values,
        "dtype": str(operator_dtype),
        "package_version": __version__,
    }
    assert operator.mean is not None
    return PCAResult(
        scores=scores,
        components=components,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance / total_variance,
        mean=operator.mean.copy(),
        total_variance=float(total_variance),
        params=result_params,
        operator=operator if return_operator else None,
    )
