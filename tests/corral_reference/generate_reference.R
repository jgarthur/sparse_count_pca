args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript generate_reference.R /path/to/corral")
}
corral_source <- args[[1]]

suppressPackageStartupMessages(library(Matrix))
source(file.path(corral_source, "R", "checkers.R"))
source(file.path(corral_source, "R", "utils.R"))
source(file.path(corral_source, "R", "corral.R"))

# Python uses cells x genes; corral conventionally uses genes x cells.
counts <- matrix(
  c(
    5, 1, 0, 2,
    1, 4, 2, 0,
    0, 2, 5, 1,
    3, 0, 1, 4,
    2, 3, 0, 2,
    1, 1, 3, 2
  ),
  nrow = 6,
  byrow = TRUE
)

set.seed(1729)
result <- corral(
  t(counts),
  method = "irl",
  ncomp = 2,
  rtype = "standardized",
  vst_mth = "none"
)

dput(
  list(
    singular_values = result$d,
    total_inertia = result$eigsum,
    row_standard_coordinates = result$SCv,
    row_principal_coordinates = result$PCv,
    column_standard_coordinates = result$SCu,
    column_principal_coordinates = result$PCu
  )
)
