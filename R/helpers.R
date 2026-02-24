# Helpers for notebooks.
# In a .qmd under notebooks/: source("../R/helpers.R") (no here needed).
# From R with project root as working dir: source("R/helpers.R").

#' Drop columns that are entirely NA or NULL
#'
#' Removes columns from a data frame where every value is NA, NaN, or NULL.
#' Useful after import to avoid carrying empty columns through the pipeline.
#'
#' @param df A data frame (or tibble).
#' @return A data frame with the same rows; columns that were all NA/NULL are removed.
#' @export
#'
#' @examples
#' d <- data.frame(a = 1:3, b = NA_character_, c = c(1, NA, 3))
#' drop_empty_columns(d)  # drops column b
drop_empty_columns <- function(df) {
  stopifnot(is.data.frame(df))
  dplyr::select(df, dplyr::where(function(x) !all(is.na(x))))
}


#' Format a decimal as a percent for inline printing (e.g. in Quarto)
#'
#' Converts a proportion (0 to 1) to a string like "75%" for use in
#' inline R in Quarto: `r pct(0.75)` prints "75%".
#'
#' @param x Numeric; proportion(s) in [0, 1] (e.g. 0.75 for 75%).
#' @param digits Number of decimal places (default 0). Use 1 for "75.3%".
#' @return Character vector of the same length as x; NA values become "NA".
#' @export
#'
#' @examples
#' pct(0.75)           # "75%"
#' pct(0.7532, 1)      # "75.3%"
#' pct(c(0.5, 0.25))  # "50%" "25%"
pct <- function(x, digits = 0) {
  if (length(x) > 1) return(vapply(x, pct, character(1L), digits = digits))
  if (is.na(x)) return("NA")
  paste0(format(round(100 * x, digits), nsmall = digits), "%")
}
