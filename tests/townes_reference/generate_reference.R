args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript generate_reference.R /path/to/functions.R")
}
functions_path <- args[[1]]

source(functions_path)

# functions.R uses genes x cells. The Python fixture is the transpose.
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
counts <- t(counts)

results <- list()
index <- 1
for (model in c("poisson", "binomial")) {
  for (residual in c("pearson", "deviance")) {
    residual_matrix <- null_residuals(
      counts,
      mod = model,
      type = residual
    )
    centered <- scale(t(residual_matrix), center = TRUE, scale = FALSE)
    singular_values <- svd(centered, nu = 0, nv = 0)$d[1:3]
    results[[index]] <- data.frame(
      model = model,
      residual = residual,
      component = seq_along(singular_values),
      singular_value = singular_values
    )
    index <- index + 1
  }
}

write.table(
  do.call(rbind, results),
  file = stdout(),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)
