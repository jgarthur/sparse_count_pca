# Generate the pinned SCTransform v2 equality oracle from equal-depth counts.

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(sctransform))
suppressPackageStartupMessages(library(glmGamPoi))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "usage: Rscript generate_sctransform_reference.R INPUT.mtx OUTPUT.npz",
    call. = FALSE
  )
}

input_path <- args[[1L]]
output_argument <- args[[2L]]
dir.create(dirname(output_argument), recursive = TRUE, showWarnings = FALSE)
output_path <- file.path(
  normalizePath(dirname(output_argument), mustWork = TRUE),
  basename(output_argument)
)

if (as.character(packageVersion("sctransform")) != "0.4.3") {
  stop("reference generation requires sctransform 0.4.3", call. = FALSE)
}
if (as.character(packageVersion("glmGamPoi")) != "1.22.0") {
  stop("reference generation requires glmGamPoi 1.22.0", call. = FALSE)
}

# SciPy exports the fixture transposed because sctransform expects genes x cells.
umi <- as(readMM(input_path), "CsparseMatrix")
if (!identical(dim(umi), c(1024L, 256L))) {
  stop("unexpected Matrix Market shape", call. = FALSE)
}
rownames(umi) <- sprintf("gene_%04d", seq_len(nrow(umi)))
colnames(umi) <- sprintf("cell_%04d", seq_len(ncol(umi)))
if (!all(colSums(umi) == 1000)) {
  stop("all cells must have exactly 1,000 counts", call. = FALSE)
}
if (any(rowSums(umi) == 0)) {
  stop("all genes must have positive margins", call. = FALSE)
}

# Use sctransform's preferred v2 offset backend. The equality test consumes the
# final regularized theta values, so it does not assume that another backend
# would estimate the same dispersions.
fit <- sctransform::vst(
  umi = umi,
  vst.flavor = "v2",
  do_regularize = TRUE,
  fix_intercept = TRUE,
  fix_slope = TRUE,
  min_variance = 0,
  res_clip_range = c(-Inf, Inf),
  min_cells = 5,
  n_genes = NULL,
  n_cells = ncol(umi),
  residual_type = "pearson",
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  return_corrected_umi = FALSE,
  verbosity = 1
)
fit_warnings <- warnings()
if (!is.null(fit_warnings)) {
  print(fit_warnings)
}
if (!identical(fit$arguments$method, "glmGamPoi_offset")) {
  stop("expected vst.flavor='v2' to select glmGamPoi_offset")
}

parameters <- fit$model_pars_fit[rownames(umi), , drop = FALSE]
theta <- parameters[, "theta"]
alpha <- ifelse(is.infinite(theta), 0, 1 / theta)
if (any(!is.finite(alpha)) || any(alpha < 0)) {
  stop("final theta values did not map to finite nonnegative alpha values")
}

log_umi <- rep(log10(1000), ncol(umi))
fitted_mean_matrix <- exp(
  outer(parameters[, "(Intercept)"], rep(1, ncol(umi))) +
    outer(parameters[, "log_umi"], log_umi)
)
fitted_mean <- rowMeans(fitted_mean_matrix)
empirical_mean <- rowMeans(umi)
if (max(abs(fitted_mean - empirical_mean)) > 1e-12) {
  stop("fixed SCT intercepts do not reproduce empirical gene means")
}

unclipped <- sctransform::get_residuals(
  vst_out = fit,
  umi = umi,
  residual_type = "pearson",
  res_clip_range = c(-Inf, Inf),
  min_variance = 0,
  verbosity = 0
)
clip <- sqrt(ncol(umi) / 30)
clipped <- sctransform::get_residuals(
  vst_out = fit,
  umi = umi,
  residual_type = "pearson",
  res_clip_range = c(-clip, clip),
  min_variance = 0,
  verbosity = 0
)

# Seurat's SCTransform wrapper clips residuals before ScaleData centers each
# gene. do.scale=FALSE leaves the centered values at their residual scale.
center_genes <- function(x) {
  sweep(x, MARGIN = 1L, STATS = rowMeans(x), FUN = "-")
}
centered_unclipped <- t(center_genes(unclipped))
centered_clipped <- t(center_genes(clipped))

if (!all(dim(centered_unclipped) == c(256L, 1024L))) {
  stop("unexpected centered reference shape")
}
if (!any(abs(unclipped) > clip)) {
  stop("the fixture does not exercise clipping")
}
if (max(abs(colMeans(centered_unclipped))) > 1e-12 ||
    max(abs(colMeans(centered_clipped))) > 1e-12) {
  stop("reference matrices are not gene-centered")
}

# NumPy's .npz format is a ZIP archive of .npy members. This small writer keeps
# the manual R generator self-contained and stores doubles without text-format
# precision loss.
write_npy_float64 <- function(path, value) {
  value <- as.array(value)
  dimensions <- dim(value)
  if (is.null(dimensions)) {
    shape <- sprintf("(%d,)", length(value))
    payload <- as.double(value)
  } else if (length(dimensions) == 1L) {
    shape <- sprintf("(%d,)", dimensions[[1L]])
    payload <- as.double(value)
  } else if (length(dimensions) == 2L) {
    shape <- sprintf("(%d, %d)", dimensions[[1L]], dimensions[[2L]])
    payload <- as.double(t(value))
  } else {
    stop("the reference NPY writer supports only vectors and matrices")
  }

  header_core <- sprintf(
    "{'descr': '<f8', 'fortran_order': False, 'shape': %s, }",
    shape
  )
  padding <- (64L - ((10L + nchar(header_core) + 1L) %% 64L)) %% 64L
  header <- paste0(header_core, strrep(" ", padding), "\n")
  header_raw <- charToRaw(header)
  if (length(header_raw) > 65535L) {
    stop("NPY header is too large for format version 1.0")
  }

  connection <- file(path, open = "wb")
  on.exit(close(connection), add = TRUE)
  writeBin(as.raw(c(0x93, utf8ToInt("NUMPY"))), connection)
  writeBin(as.raw(c(1L, 0L)), connection)
  writeBin(as.integer(length(header_raw)), connection, size = 2L, endian = "little")
  writeBin(header_raw, connection)
  writeBin(payload, connection, size = 8L, endian = "little")
}

temporary <- tempfile("sctransform-reference-")
dir.create(temporary)
on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
members <- list(
  theta = theta,
  alpha = alpha,
  fitted_mean = fitted_mean,
  clip = clip,
  clip_counts = c(sum(unclipped < -clip), sum(unclipped > clip)),
  centered_unclipped = centered_unclipped,
  centered_clipped = centered_clipped
)
member_names <- paste0(names(members), ".npy")
for (index in seq_along(members)) {
  member_path <- file.path(temporary, member_names[[index]])
  write_npy_float64(member_path, members[[index]])
  Sys.setFileTime(member_path, as.POSIXct("2000-01-01 00:00:00", tz = "UTC"))
}

if (file.exists(output_path)) {
  unlink(output_path)
}
old_working_directory <- setwd(temporary)
on.exit(setwd(old_working_directory), add = TRUE)
utils::zip(
  zipfile = output_path,
  files = member_names,
  flags = "-X -q"
)
if (!file.exists(output_path)) {
  stop("failed to create the NPZ reference artifact")
}

cat(sprintf("wrote %s\n", output_path))
cat(sprintf("theta finite: %d; theta infinite: %d\n", sum(is.finite(theta)), sum(is.infinite(theta))))
cat(sprintf("clip: %.17g\n", clip))
cat(sprintf(
  "unclipped entries below/above clip: %d/%d\n",
  sum(unclipped < -clip),
  sum(unclipped > clip)
))
