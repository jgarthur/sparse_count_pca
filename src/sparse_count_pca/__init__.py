"""Public API for sparse count-matrix PCA and correspondence analysis."""

from ._correspondence import (
    CorrespondenceAnalysisResult,
    correspondence_analysis,
    correspondence_analysis_matrix,
)
from ._dirichlet_pca import (
    dirichlet_clr_pca,
    dirichlet_clr_pca_matrix,
    dirichlet_log_pca,
    dirichlet_log_pca_matrix,
)
from ._log_pca import (
    proportion_shifted_clr_pca,
    proportion_shifted_clr_pca_matrix,
    shifted_clr_pca,
    shifted_clr_pca_matrix,
    shifted_log_pca,
    shifted_log_pca_matrix,
)
from ._pca import PCAResult
from ._residual_pca import residual_pca, residual_pca_matrix
from ._transform import (
    DirichletCLR,
    DirichletLog,
    ProportionShiftedCLR,
    Residual,
    ShiftedCLR,
    ShiftedLog,
    Transform,
    TransformedMatrix,
    transform,
)
from ._version import __version__

__all__ = [
    "CorrespondenceAnalysisResult",
    "DirichletCLR",
    "DirichletLog",
    "PCAResult",
    "ProportionShiftedCLR",
    "Residual",
    "ShiftedCLR",
    "ShiftedLog",
    "Transform",
    "TransformedMatrix",
    "__version__",
    "correspondence_analysis",
    "correspondence_analysis_matrix",
    "dirichlet_clr_pca",
    "dirichlet_clr_pca_matrix",
    "dirichlet_log_pca",
    "dirichlet_log_pca_matrix",
    "proportion_shifted_clr_pca",
    "proportion_shifted_clr_pca_matrix",
    "residual_pca",
    "residual_pca_matrix",
    "shifted_clr_pca",
    "shifted_clr_pca_matrix",
    "shifted_log_pca",
    "shifted_log_pca_matrix",
    "transform",
]
