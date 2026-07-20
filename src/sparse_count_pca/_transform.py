"""Public two-step count transformation and materialization API."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from anndata import AnnData
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ._anndata import _empty, _get_count_matrix, _resolve_mask_var
from ._clip import ClipMode, validate_clip
from ._counts import BoolArray, CountMatrix, _canonicalize_counts, _sum_counts
from ._log_transforms import (
    PriorProportions,
    build_dirichlet_clr_representation,
    build_dirichlet_log_representation,
    build_proportion_shifted_clr_representation,
    build_shifted_clr_representation,
    build_shifted_log_representation,
    validate_dirichlet_prior,
    validate_positive_scalar,
)
from ._operator import _normalize_operator_dtype
from ._pca import PCAResult, compute_pca_from_representation
from ._representation import SparseLowRankMatrix
from ._residuals import (
    AlphaLike,
    Model,
    ResidualType,
    _validate_model,
    build_residual_representation,
)
from ._svd import Solver


class Transform(ABC):
    """Configuration for a count transformation fitted to one matrix."""

    @abstractmethod
    def _build(
        self,
        counts: sparse.csr_matrix,
        *,
        var: Any | None,
        columns: BoolArray | None,
    ) -> tuple[SparseLowRankMatrix, str, dict[str, Any]]:
        """Build a representation and its reproducibility metadata.

        ``var`` is available only for AnnData-derived transforms. ``columns``
        is either ``None`` for all variables or the PCA-only variable mask;
        builders must fit normalization state before applying that mask.
        """


def _resolve_vector_parameter(
    value: ArrayLike | str | None,
    *,
    var: Any | None,
    name: str,
) -> ArrayLike | None:
    """Resolve an optional per-variable parameter from AnnData metadata."""
    if not isinstance(value, str):
        return value
    if var is None:
        raise TypeError(f"{name}={value!r} requires an AnnData input")
    if value not in var:
        raise KeyError(f"{value!r} not found in adata.var")
    return np.asarray(var[value], dtype=np.float64)


def _serialize_parameter(value: Any, *, none: Any = None) -> Any:
    """Convert transform configuration values to stable result metadata."""
    if value is None:
        return none
    if isinstance(value, (str, int, float)):
        return value
    values = np.asarray(value)
    return values.item() if values.ndim == 0 else values.copy()


@dataclass(frozen=True)
class Residual(Transform):
    """Pearson or deviance residual transformation specification."""

    model: Model = "poisson"
    residual: ResidualType = "pearson"
    alpha: AlphaLike | str = None
    clip: float | None = None
    clip_mode: ClipMode = "symmetric"
    clip_max_nnz_ratio: float | None = 2.0

    def _build(
        self,
        counts: sparse.csr_matrix,
        *,
        var: Any | None,
        columns: BoolArray | None,
    ) -> tuple[SparseLowRankMatrix, str, dict[str, Any]]:
        validate_clip(self.clip, self.clip_mode, self.clip_max_nnz_ratio)
        n_vars = counts.shape[1]
        alpha_input = _resolve_vector_parameter(self.alpha, var=var, name="alpha")
        alpha_full = _validate_model(self.model, self.residual, alpha_input, n_vars)
        n = _sum_counts(counts, axis=1)
        column_totals = _sum_counts(counts, axis=0)
        if not (np.isfinite(n).all() and np.isfinite(column_totals).all()):
            raise ValueError("Count margins overflow float64")
        if (n == 0).any():
            raise ValueError("Cells with zero total counts are not supported")
        with np.errstate(over="ignore"):
            total = float(np.sum(n, dtype=np.float64))
        if not np.isfinite(total):
            raise ValueError("Grand total overflows float64")
        p_full = column_totals / total

        all_columns = columns is None or columns.all()
        p = p_full if all_columns else p_full[columns]
        if (p == 0).any():
            raise ValueError("Selected genes with zero total counts are not supported")
        if self.model == "binomial" and (p >= 1).any():
            raise ValueError("Binomial residuals require 0 < p_j < 1")
        alpha_used = (
            alpha_full if alpha_full is None or all_columns else alpha_full[columns]
        )
        counts_used = counts if all_columns else counts[:, columns].tocsr()
        representation = build_residual_representation(
            counts_used,
            n,
            p,
            model=self.model,
            residual=self.residual,
            alpha=alpha_used,
            clip=self.clip,
            clip_mode=self.clip_mode,
            clip_max_nnz_ratio=self.clip_max_nnz_ratio,
        )
        return (
            representation,
            "Residual",
            {
                "model": self.model,
                "residual": self.residual,
                "alpha": _serialize_parameter(self.alpha),
                "clip": self.clip,
                "clip_mode": self.clip_mode,
                "clip_max_nnz_ratio": self.clip_max_nnz_ratio,
                "normalization_n_vars": n_vars,
                "use_highly_variable": False,
                "mask_var": None,
                "layer": None,
            },
        )


@dataclass(frozen=True)
class ShiftedLog(Transform):
    """Fixed-count shifted-log transformation specification."""

    count_shift: float

    def _build(self, counts, *, var, columns):
        count_shift = validate_positive_scalar(self.count_shift, name="count_shift")
        representation = build_shifted_log_representation(
            counts, count_shift=count_shift
        )
        if columns is not None:
            representation = representation.select_columns(columns)
        return (
            representation,
            "Shifted log",
            {
                "transform": "shifted_log",
                "shift_domain": "count",
                "count_shift": count_shift,
                "normalization_n_vars": counts.shape[1],
            },
        )


@dataclass(frozen=True)
class ShiftedCLR(Transform):
    """Fixed-count shifted-CLR transformation specification."""

    count_shift: float

    def _build(self, counts, *, var, columns):
        count_shift = validate_positive_scalar(self.count_shift, name="count_shift")
        representation = build_shifted_clr_representation(
            counts, count_shift=count_shift
        )
        if columns is not None:
            representation = representation.select_columns(columns)
        return (
            representation,
            "Shifted CLR",
            {
                "transform": "shifted_clr",
                "shift_domain": "count",
                "count_shift": count_shift,
                "normalization_n_vars": counts.shape[1],
            },
        )


@dataclass(frozen=True)
class ProportionShiftedCLR(Transform):
    """Fixed-composition shifted-CLR transformation specification."""

    composition_shift: float

    def _build(self, counts, *, var, columns):
        composition_shift = validate_positive_scalar(
            self.composition_shift, name="composition_shift"
        )
        representation = build_proportion_shifted_clr_representation(
            counts, composition_shift=composition_shift
        )
        if columns is not None:
            representation = representation.select_columns(columns)
        return (
            representation,
            "Proportion-shifted CLR",
            {
                "transform": "proportion_shifted_clr",
                "shift_domain": "composition",
                "composition_shift": composition_shift,
                "effective_count_shift": "cell_total * composition_shift",
                "normalization_n_vars": counts.shape[1],
            },
        )


@dataclass(frozen=True)
class DirichletLog(Transform):
    """Dirichlet posterior-mean log transformation specification."""

    concentration: float = 1.0
    prior_proportions: PriorProportions | str = None

    def _build(self, counts, *, var, columns):
        prior_input = _resolve_vector_parameter(
            self.prior_proportions, var=var, name="prior_proportions"
        )
        concentration, proportions = validate_dirichlet_prior(
            self.concentration, prior_input, counts.shape[1]
        )
        representation = build_dirichlet_log_representation(
            counts,
            concentration=concentration,
            prior_proportions=proportions,
        )
        if columns is not None:
            representation = representation.select_columns(columns)
        return (
            representation,
            "Dirichlet log",
            {
                "transform": "dirichlet_log",
                "shift_domain": "dirichlet_prior_counts",
                "concentration": concentration,
                "prior_proportions": _serialize_parameter(
                    self.prior_proportions, none="uniform"
                ),
                "normalization_n_vars": counts.shape[1],
            },
        )


@dataclass(frozen=True)
class DirichletCLR(Transform):
    """Dirichlet posterior-mean CLR transformation specification."""

    concentration: float = 1.0
    prior_proportions: PriorProportions | str = None

    def _build(self, counts, *, var, columns):
        prior_input = _resolve_vector_parameter(
            self.prior_proportions, var=var, name="prior_proportions"
        )
        concentration, proportions = validate_dirichlet_prior(
            self.concentration, prior_input, counts.shape[1]
        )
        representation = build_dirichlet_clr_representation(
            counts,
            concentration=concentration,
            prior_proportions=proportions,
        )
        if columns is not None:
            representation = representation.select_columns(columns)
        return (
            representation,
            "Dirichlet CLR",
            {
                "transform": "dirichlet_clr",
                "shift_domain": "dirichlet_prior_counts",
                "concentration": concentration,
                "prior_proportions": _serialize_parameter(
                    self.prior_proportions, none="uniform"
                ),
                "normalization_n_vars": counts.shape[1],
            },
        )


def _normalize_selector(
    selector: Any,
    size: int,
    *,
    names: Any | None,
    axis_name: str,
) -> NDArray[np.intp]:
    """Normalize positional, boolean, slice, or AnnData-name selectors."""
    if selector is None:
        return np.arange(size, dtype=np.intp)
    if isinstance(selector, str):
        selector = [selector]
    values = np.asarray(selector) if not isinstance(selector, slice) else None
    if values is not None and values.size == 0:
        return np.empty(0, dtype=np.intp)
    if values is not None and values.dtype.kind in {"O", "U", "S"}:
        if names is None:
            raise TypeError(f"Named {axis_name} selection requires an AnnData input")
        if not names.is_unique:
            raise ValueError(
                f"Named {axis_name} selection requires unique AnnData names"
            )
        indices = names.get_indexer(values.tolist())
        if (indices < 0).any():
            missing = values[indices < 0].tolist()
            raise KeyError(f"Unknown {axis_name} names: {missing}")
        return indices.astype(np.intp, copy=False)
    try:
        indices = np.arange(size, dtype=np.intp)[selector]
    except (IndexError, TypeError, ValueError) as error:
        raise IndexError(f"Invalid {axis_name} selector") from error
    return np.atleast_1d(indices).astype(np.intp, copy=False)


class TransformedMatrix(LinearOperator):
    """Uncentered implicit transformed matrix with inspection and PCA methods."""

    def __init__(
        self,
        representation: SparseLowRankMatrix,
        *,
        transform_label: str,
        params: dict[str, Any],
        check_values: bool,
        dtype: DTypeLike,
        obs_names: Any | None = None,
        var_names: Any | None = None,
        var: Any | None = None,
        isolate_returned_operator: bool = True,
    ) -> None:
        operator_dtype = _normalize_operator_dtype(dtype)
        if (
            representation.sparse.dtype != operator_dtype
            or representation.left.dtype != operator_dtype
            or representation.right.dtype != operator_dtype
        ):
            representation = SparseLowRankMatrix(
                representation.sparse.astype(operator_dtype, copy=False),
                representation.left.astype(operator_dtype, copy=False),
                representation.right.astype(operator_dtype, copy=False),
            )
        self._representation = representation
        self._sparse = representation.sparse
        self._left = representation.left
        self._right = representation.right
        self.transform_label = transform_label
        self.params = dict(params)
        self.check_values = check_values
        self.obs_names = obs_names.copy() if obs_names is not None else None
        self.var_names = var_names.copy() if var_names is not None else None
        self._var = var.copy() if var is not None else None
        # Public two-step transforms remain live after ``pca`` and therefore
        # isolate a mutable returned operator. One-step wrappers mark their
        # temporary transform transferable so the operator can take ownership.
        self._isolate_returned_operator = isolate_returned_operator
        super().__init__(dtype=operator_dtype, shape=representation.shape)

    def _matvec(self, vector: NDArray[Any]) -> NDArray[Any]:
        vector = np.asarray(vector, dtype=self.dtype)
        values = self._sparse @ vector
        if self._right.shape[1]:
            values += self._left @ (self._right.T @ vector)
        return np.asarray(values, dtype=self.dtype)

    def _rmatvec(self, vector: NDArray[Any]) -> NDArray[Any]:
        vector = np.asarray(vector, dtype=self.dtype)
        values = self._sparse.T @ vector
        if self._right.shape[1]:
            values += self._right @ (self._left.T @ vector)
        return np.asarray(values, dtype=self.dtype)

    def _matmat(self, matrix: NDArray[Any]) -> NDArray[Any]:
        matrix = np.asarray(matrix, dtype=self.dtype)
        values = self._sparse @ matrix
        if self._right.shape[1]:
            values += self._left @ (self._right.T @ matrix)
        return np.asarray(values, dtype=self.dtype)

    def _rmatmat(self, matrix: NDArray[Any]) -> NDArray[Any]:
        matrix = np.asarray(matrix, dtype=self.dtype)
        values = self._sparse.T @ matrix
        if self._right.shape[1]:
            values += self._right @ (self._left.T @ matrix)
        return np.asarray(values, dtype=self.dtype)

    def materialize(
        self,
        *,
        obs: Any = None,
        var: Any = None,
        out: NDArray[Any] | None = None,
        block_size: int = 1024,
    ) -> NDArray[Any]:
        """Materialize arbitrary observations and variables into a dense array.

        Integer selections retain a two-dimensional result. Positional arrays,
        boolean masks, slices, and names from an AnnData input are accepted.
        Observation blocks are the efficient access direction of the current CSR
        backend. Selecting variables across all observations emits a warning.
        """
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        rows = _normalize_selector(
            obs, self.shape[0], names=self.obs_names, axis_name="observation"
        )
        columns = _normalize_selector(
            var, self.shape[1], names=self.var_names, axis_name="variable"
        )
        if var is not None and obs is None:
            warnings.warn(
                "Selecting variables across all observations may be slow with "
                "the current CSR transform backend; observation-block access is "
                "the efficient direction",
                sparse.SparseEfficiencyWarning,
                stacklevel=2,
            )
        shape = (rows.size, columns.size)
        if out is None:
            destination = np.empty(shape, dtype=self.dtype)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a NumPy array or memory-mapped array")
            destination = out
            if destination.shape != shape:
                raise ValueError(f"out must have shape {shape}")
            if not np.issubdtype(destination.dtype, np.floating):
                raise TypeError("out must have a floating-point dtype")
            if not destination.flags.writeable:
                raise ValueError("out must be writable")

        for start in range(0, rows.size, block_size):
            stop = min(start + block_size, rows.size)
            block_rows = slice(start, stop) if obs is None else rows[start:stop]
            sparse_block = self._sparse[block_rows]
            if var is not None:
                sparse_block = sparse_block[:, columns]
            sparse_values = sparse_block.toarray()
            if self._right.shape[1]:
                right = self._right if var is None else self._right[columns]
                sparse_values += self._left[block_rows] @ right.T
            np.copyto(destination[start:stop], sparse_values, casting="same_kind")
        return destination

    def pca(
        self,
        n_comps: int = 50,
        *,
        mask_var: Any = _empty,
        use_highly_variable: bool | None = None,
        solver: Solver = "arpack",
        random_state: int | None = 0,
        tol: float = 0.0,
        return_operator: bool = False,
    ) -> PCAResult:
        """Run PCA after selecting variables from the fitted transform."""
        params = dict(self.params)
        if self._var is None:
            if use_highly_variable is not None:
                raise TypeError(
                    "use_highly_variable requires a transform built from AnnData"
                )
            if mask_var is _empty or mask_var is None:
                mask = np.ones(self.shape[1], dtype=bool)
            elif isinstance(mask_var, str):
                raise TypeError(
                    "String mask_var requires a transform built from AnnData"
                )
            else:
                from ._counts import _validate_boolean_mask

                mask = _validate_boolean_mask(mask_var, self.shape[1], name="mask_var")
                if not mask.any():
                    raise ValueError("mask_var selected zero genes")
        else:
            resolved_mask = _resolve_mask_var(self._var, mask_var, use_highly_variable)
            mask = resolved_mask.values
            params.update(
                {
                    "mask_var": resolved_mask.mask_var,
                    "use_highly_variable": resolved_mask.use_highly_variable,
                    "mask_var_details": resolved_mask.details,
                }
            )
        representation = (
            self._representation
            if mask.all()
            else self._representation.select_columns(mask)
        )
        return compute_pca_from_representation(
            representation,
            n_comps,
            transform_label=self.transform_label,
            params=params,
            check_values=self.check_values,
            dtype=self.dtype,
            solver=solver,
            random_state=random_state,
            tol=tol,
            return_operator=return_operator,
            copy_operator=return_operator and self._isolate_returned_operator,
        )


def _transform(
    data: CountMatrix | AnnData,
    method: Transform,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    _columns: BoolArray | None = None,
    _isolate_returned_operator: bool = True,
) -> TransformedMatrix:
    """Build a public transform, optionally restricted for a PCA wrapper."""
    if not isinstance(method, Transform):
        raise TypeError("method must be a Transform instance")
    if isinstance(data, AnnData):
        X = _get_count_matrix(data, layer=layer, use_raw=use_raw)
        var = data.var
        obs_names = data.obs_names
        var_names = data.var_names
    else:
        if layer is not None or use_raw:
            raise TypeError("layer and use_raw are only valid for AnnData input")
        X = data
        var = None
        obs_names = None
        var_names = None
    counts = _canonicalize_counts(X, check_values=check_values)
    if _columns is not None:
        from ._counts import _validate_boolean_mask

        _columns = _validate_boolean_mask(_columns, counts.shape[1], name="columns")
        if not _columns.any():
            raise ValueError("columns selected zero genes")
        if _columns.all():
            _columns = None
    representation, label, params = method._build(counts, var=var, columns=_columns)
    if _columns is not None and var_names is not None:
        var_names = var_names[_columns]
        var = var.iloc[np.flatnonzero(_columns)]
    return TransformedMatrix(
        representation,
        transform_label=label,
        params=params,
        check_values=check_values,
        dtype=dtype,
        obs_names=obs_names,
        var_names=var_names,
        var=var,
        isolate_returned_operator=_isolate_returned_operator,
    )


def transform(
    data: CountMatrix | AnnData,
    method: Transform,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    check_values: bool = True,
    dtype: DTypeLike = "float64",
) -> TransformedMatrix:
    """Fit a count transform and return its implicit uncentered matrix.

    The result is an in-process object: it is not stored by ``write_h5ad`` and
    must be reconstructed after restarting Python. Correspondence analysis is
    intentionally not a transform method because its mask defines table margins.
    """
    return _transform(
        data,
        method,
        layer=layer,
        use_raw=use_raw,
        check_values=check_values,
        dtype=dtype,
    )
