# Helpers for notebooks. From project root: source("R/helpers.R")
# Or in Quarto: source(here::here("R/helpers.R")) if using the here package.

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
  dplyr::select(df, where(function(x) !all(is.na(x))))
}
