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
from ._counts import (
    BoolArray,
    CountMatrix,
    _canonicalize_counts,
    _empty_columns,
    _sum_counts,
)
from ._log_transforms import (
    PriorProportions,
    build_dirichlet_clr_representation,
    build_dirichlet_log_representation,
    build_proportion_shifted_clr_representation,
    build_shifted_clr_representation,
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
    """Abstract configuration for a count transformation.

    Public transform specifications are immutable dataclasses passed to
    ``transform``. Subclassing is not currently a supported public
    extension mechanism because the representation-building contract is
    private.
    """

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
    """Specify a Pearson or deviance residual transformation.

    Every model uses the same fitted null mean ``mu_ij = n_i * p_j``, where
    ``n_i`` is observation ``i``'s count total and ``p_j`` is variable ``j``'s
    share of the grand total. Pearson residuals are
    ``(x_ij - mu_ij) / sqrt(V_ij)`` and deviance residuals are
    ``sign(x_ij - mu_ij) * sqrt(d(x_ij, mu_ij))``.

    Attributes:
        model: Null model supplying the variance ``V_ij``. ``"poisson"`` uses
            ``mu_ij``, ``"binomial"`` uses ``mu_ij * (1 - p_j)``, and
            ``"scaled_nb"`` uses ``mu_ij * (1 + alpha_j * mean_n * p_j)``, where
            ``mean_n`` is the mean observation count total.
        residual: ``"pearson"`` for standardized deviations from ``mu_ij``, or
            ``"deviance"`` for signed square-root deviance contributions.
        alpha: Per-variable overdispersion of the ``scaled_nb`` model, as a
            scalar broadcast to every variable, a length-``n_vars`` array, or an
            AnnData variable key. It is a dispersion, not a size: larger values
            mean more variance and ``alpha=0`` is the Poisson limit. An estimate
            reported as a size (``theta``, ``r``) must be inverted first. Must
            be nonnegative; values below ``1e-8`` use the Poisson limit. Used
            only by ``scaled_nb``, and rejected for the other models. Values for
            variables with no counts are replaced with zero instead of being
            validated, since they cannot reach any output.
        clip: Positive threshold applied to the uncentered residual values
            before PCA centering, or ``None`` for no clipping.
        clip_mode: ``"symmetric"`` clips residuals into ``[-clip, clip]``;
            ``"upper"`` clips only from above, into ``(-inf, clip]``, which
            leaves zero-count residuals untouched.
        clip_max_nnz_ratio: Upper bound on how far exact symmetric clipping may
            grow the stored sparse support, as a multiple of the input count
            matrix's number of stored nonzeros. Reaching it raises
            ``RuntimeError``; ``None`` removes the limit. Only exact symmetric
            clipping can add support, so the limit never binds when ``clip`` is
            ``None`` or ``clip_mode="upper"``.

    Examples:
        >>> method = Residual(model="poisson", residual="pearson")
        >>> transformed = transform(counts, method)
    """

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
        n = _sum_counts(counts, axis=1)
        column_totals = _sum_counts(counts, axis=0)
        if not (np.isfinite(n).all() and np.isfinite(column_totals).all()):
            raise ValueError("Count margins overflow float64")
        if (n == 0).any():
            raise ValueError(
                "Cells with zero total counts are not supported; filter empty "
                "rows out of the count matrix first"
            )
        # A gene with no counts has a fitted mean of zero, so its residual is
        # zero by continuity and the representation carries an exactly zero
        # column. Its overdispersion is therefore irrelevant.
        alpha_full = _validate_model(
            self.model,
            self.residual,
            alpha_input,
            n_vars,
            ignore=column_totals == 0,
        )
        with np.errstate(over="ignore"):
            total = float(np.sum(n, dtype=np.float64))
        if not np.isfinite(total):
            raise ValueError("Grand total overflows float64")
        p_full = column_totals / total

        all_columns = columns is None or columns.all()
        p = p_full if all_columns else p_full[columns]
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
class ShiftedCLR(Transform):
    """Specify count-scale shifted CLR coordinates.

    The transform is ``clr(x_i + count_shift)``, that is
    ``log(x_ij + count_shift) - mean_k log(x_ik + count_shift)`` over all
    ``n_vars`` variables.

    Attributes:
        count_shift: Positive constant added to every raw count before taking
            logs. It is an additive shift on the raw-count scale, not a
            multiplicative rescaling, and it is the same for every observation
            regardless of sequencing depth. Larger values shrink log-ratios
            toward zero. The PFlog formulation in Booeshaghi et al. preprint v4
            uses ``count_shift = 1 / (4 * alpha)``, for a dataset-wide ``alpha``
            under the standard NB2 model rather than the per-gene ``scaled_nb``
            ``alpha`` of ``Residual``.

    Examples:
        >>> transformed = transform(counts, ShiftedCLR(count_shift=1.0))
    """

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
    """Specify composition-scale shifted CLR coordinates.

    The transform is ``clr(x_i / n_i + composition_shift)``, where ``n_i`` is
    observation ``i``'s count total.

    Attributes:
        composition_shift: Positive constant added to every proportion after
            dividing each row by its count total. It is an additive shift on the
            composition scale, not a multiplicative rescaling. Because
            ``clr(x_i / n_i + composition_shift) = clr(x_i + n_i *
            composition_shift)``, the equivalent raw-count shift is
            ``n_i * composition_shift`` and therefore differs per observation.
            This is what distinguishes it from ``ShiftedCLR``.

    Examples:
        >>> transformed = transform(
        ...     counts, ProportionShiftedCLR(composition_shift=1.0)
        ... )
    """

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
            "Composition-scale shifted CLR",
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
    """Specify log posterior-mean compositions under a Dirichlet prior.

    With prior counts ``a_j = concentration * prior_proportions_j``, the
    transform is ``log((x_ij + a_j) / (n_i + concentration))``, where ``n_i`` is
    observation ``i``'s count total.

    Attributes:
        concentration: Positive total prior concentration, in units of counts:
            it is the number of prior pseudo-counts spread over the variables,
            so it is an additive shift of ``concentration *
            prior_proportions_j`` counts for variable ``j``. Larger values
            shrink each observation harder toward the prior composition.
        prior_proportions: Prior composition the pseudo-counts are distributed
            across, as a length-``n_vars`` array, an AnnData variable key, or
            ``None`` for a uniform prior. Values must be strictly positive and
            sum to one; these are proportions, not counts.

    Examples:
        >>> transformed = transform(counts, DirichletLog(concentration=1.0))
    """

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
    """Specify CLR posterior-mean compositions under a Dirichlet prior.

    With prior counts ``a_j = concentration * prior_proportions_j``, the
    transform is ``clr((x_i + a) / (n_i + concentration))``, which equals
    ``clr(x_i + a)`` because CLR removes the per-observation denominator. A
    uniform prior therefore reduces to ``ShiftedCLR`` with
    ``count_shift = concentration / n_vars``.

    Attributes:
        concentration: Positive total prior concentration, in units of counts:
            it is the number of prior pseudo-counts spread over the variables,
            so it is an additive shift of ``concentration *
            prior_proportions_j`` counts for variable ``j``. Larger values
            shrink each observation harder toward the prior composition.
        prior_proportions: Prior composition the pseudo-counts are distributed
            across, as a length-``n_vars`` array, an AnnData variable key, or
            ``None`` for a uniform prior. Values must be strictly positive and
            sum to one; these are proportions, not counts.

    Examples:
        >>> transformed = transform(counts, DirichletCLR(concentration=1.0))
    """

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
    """Uncentered implicit transformed matrix with inspection and PCA methods.

    Instances are returned by ``transform``. The fitted representation owns
    its sparse support and remains isolated from later mutation of caller-owned
    input. It exists only in the current Python process and is not serialized
    with AnnData.

    Attributes:
        shape: Transformed matrix shape ``(n_obs, n_vars)``.
        dtype: Floating-point representation and operator dtype.
        transform_label: Human-readable transform name.
        params: Fitted transform and reproducibility metadata. It records the
            transform's own parameters plus ``normalization_n_vars``, the number
            of variables the normalization was fitted on, ``n_empty_vars``, how
            many of those have no counts, and, for the shifted transforms,
            ``shift_domain``, naming the scale the shift was applied on.
        obs_names: Copied AnnData observation names, or ``None`` for matrix
            input.
        var_names: Copied AnnData variable names, or ``None`` for matrix input.
    """

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
        """Materialize selected transformed values into a dense array.

        Integer selections retain a two-dimensional result. Positional arrays,
        boolean masks, slices, and names from an AnnData input are accepted.
        Observation blocks are the efficient access direction of the current
        CSR backend. Selecting variables across all observations emits
        ``SparseEfficiencyWarning``.

        Args:
            obs: Observation selector, or ``None`` for every observation.
            var: Variable selector, or ``None`` for every variable.
            out: Writable floating-point NumPy array or memory map with the
                selected output shape. If omitted, allocate a new array.
            block_size: Positive number of selected observations densified per
                output block. It trades peak memory against the number of
                passes and does not change the returned values.

        Returns:
            Dense two-dimensional transformed values. When ``out`` is given,
            the returned object is ``out``.

        Raises:
            IndexError: If an observation or variable selector is invalid.
            KeyError: If a named AnnData selector is unknown.
            TypeError: If named selection is used for matrix input or ``out``
                has an unsupported type or dtype.
            ValueError: If ``block_size`` or the shape or writability of ``out``
                is invalid.

        Examples:
            >>> row = transformed.materialize(obs=10)
            >>> block = transformed.materialize(obs=[10, 3], var=[25, 2, 8])
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
        """Run centered PCA after selecting variables from the fitted transform.

        The fitted normalization state is reused. The variable mask is applied
        to that transform, then a new column mean and truncated SVD are computed
        for the selected columns.

        Args:
            n_comps: Number of principal components to return. Must satisfy
                ``1 <= n_comps < min(n_obs, n_nonempty_vars)``, where
                ``n_nonempty_vars`` counts selected variables that are not
                identically zero.
            mask_var: Boolean array or, for AnnData-derived transforms, an
                ``adata.var`` key. When omitted, use ``"highly_variable"`` if
                present; explicit ``None`` selects every variable.
            use_highly_variable: Deprecated Scanpy-compatible mask selector,
                available only for AnnData-derived transforms.
            solver: SVD solver. Only ``"arpack"`` is supported.
            random_state: Seed for the random starting vector handed to ARPACK.
                ``None`` draws an unseeded vector, so runs are no longer bit-for-bit
                reproducible.
            tol: Convergence tolerance passed to SciPy's ``svds``. ``0.0``
                requests machine precision; larger values stop sooner and less
                accurately.
            return_operator: Whether to retain an isolated centered operator in
                the result.

        Returns:
            PCA scores, components, variance statistics, metadata, and
            optionally the centered operator.

        Raises:
            TypeError: If AnnData-only mask features are used for matrix input.
            ValueError: If the mask, dimensions, or centered variance are
                invalid.

        Examples:
            >>> result = transformed.pca(n_comps=20, mask_var=my_gene_mask)
        """
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
    check_values: bool = True,
    dtype: DTypeLike = "float64",
    _columns: BoolArray | None = None,
    _isolate_returned_operator: bool = True,
) -> TransformedMatrix:
    """Build a public transform, optionally restricted for a PCA wrapper."""
    if not isinstance(method, Transform):
        raise TypeError("method must be a Transform instance")
    if isinstance(data, AnnData):
        X = _get_count_matrix(data, layer=layer)
        var = data.var
        obs_names = data.obs_names
        var_names = data.var_names
    else:
        if layer is not None:
            raise TypeError("layer is only valid for AnnData input")
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
    # Empty variables are never dropped: every transform gives them a defined
    # value, so the fitted matrix always keeps the caller's variable universe.
    params = {**params, "n_empty_vars": int(_empty_columns(counts).sum())}
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
    check_values: bool = True,
    dtype: DTypeLike = "float64",
) -> TransformedMatrix:
    """Fit a count transform and return its implicit uncentered matrix.

    The result is an in-process object: it is not stored by ``write_h5ad`` and
    must be reconstructed after restarting Python. Correspondence analysis is
    intentionally not a transform method because its mask defines table margins.

    Args:
        data: Dense, SciPy sparse, or backed sparse count matrix, or an AnnData
            object. Observations are rows and variables are columns.
        method: Residual, shifted-CLR, or Dirichlet transform specification.
        layer: AnnData count layer to use. By default, use ``adata.X``.
        check_values: When ``True``, reject floating-point input whose values
            are not within ``1e-8`` of integers. Set it to ``False`` to accept
            genuinely fractional input.
        dtype: Representation and operator dtype, either ``"float64"`` or
            ``"float32"``. Normalization is always fitted in float64 and cast
            afterwards.

    Returns:
        An uncentered implicit transformed matrix that supports matrix products,
        bounded materialization, and repeated PCA.

    Raises:
        TypeError: If ``method`` is not a transform specification or AnnData-only
            parameters are used with matrix input.
        ValueError: If counts, transform parameters, or dtype are invalid.
        KeyError: If a requested AnnData layer or variable parameter is absent.

    Examples:
        >>> transformed = transform(
        ...     adata,
        ...     Residual(model="poisson", residual="pearson"),
        ...     layer="counts",
        ... )
        >>> transformed.shape
        (adata.n_obs, adata.n_vars)
    """
    return _transform(
        data,
        method,
        layer=layer,
        check_values=check_values,
        dtype=dtype,
    )
