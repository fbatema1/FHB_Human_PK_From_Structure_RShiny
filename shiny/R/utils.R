##############################################################################
# shiny/R/utils.R
# ===============
# Data parsing and table formatting helpers.
##############################################################################

# ── Parse raw API response into a tidy data.frame ─────────────────────────────
#
# result    : list of per-compound prediction objects from the API
# names_vec : character vector of compound names (same length as result)
# show_ci   : logical — include CI columns
#
parse_predictions <- function(result, names_vec, show_ci = TRUE) {

  rows <- lapply(seq_along(result), function(i) {
    r    <- result[[i]]
    name <- if (i <= length(names_vec)) names_vec[[i]] else sprintf("Compound_%d", i)

    if (!is.null(r$error) && nchar(r$error) > 0) {
      return(data.frame(
        name       = name,
        smiles     = r$smiles %||% NA_character_,
        status     = "error",
        error_msg  = r$error,
        CL_pred    = NA_real_,
        CL_lower   = NA_real_,
        CL_upper   = NA_real_,
        Vd_pred    = NA_real_,
        Vd_lower   = NA_real_,
        Vd_upper   = NA_real_,
        thalf_pred = NA_real_,
        lambdaz_pred = NA_real_,
        model_used = r$model_used %||% NA_character_,
        stringsAsFactors = FALSE
      ))
    }

    data.frame(
      name         = name,
      smiles       = r$smiles        %||% NA_character_,
      status       = "ok",
      error_msg    = NA_character_,
      CL_pred      = r$CL_pred       %||% NA_real_,
      CL_lower     = r$CL_lower      %||% NA_real_,
      CL_upper     = r$CL_upper      %||% NA_real_,
      Vd_pred      = r$Vd_pred       %||% NA_real_,
      Vd_lower     = r$Vd_lower      %||% NA_real_,
      Vd_upper     = r$Vd_upper      %||% NA_real_,
      thalf_pred   = r$thalf_pred    %||% NA_real_,
      lambdaz_pred = r$lambdaz_pred  %||% NA_real_,
      model_used   = r$model_used    %||% NA_character_,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, rows)
}

# ── Format for DT display ─────────────────────────────────────────────────────
format_results_table <- function(df, show_ci = TRUE) {

  ok  <- df$status == "ok"
  out <- data.frame(
    Name  = df$name,
    stringsAsFactors = FALSE
  )

  # Helper: format numeric with NA → "—"
  fmt <- function(x, digits) ifelse(is.na(x), "—", formatC(x, digits, format = "f"))

  if (show_ci) {
    out[["CL (mL/min/kg)"]]     <- fmt(df$CL_pred, 3)
    out[["CL 95% CI"]]          <- ci_string(df$CL_lower, df$CL_upper, 3)
    out[["Vd (L/kg)"]]          <- fmt(df$Vd_pred, 3)
    out[["Vd 95% CI"]]          <- ci_string(df$Vd_lower, df$Vd_upper, 3)
  } else {
    out[["CL (mL/min/kg)"]]     <- fmt(df$CL_pred, 3)
    out[["Vd (L/kg)"]]          <- fmt(df$Vd_pred, 3)
  }

  out[["t½ (h)"]]          <- fmt(df$thalf_pred,   2)
  out[["λz (1/h)"]]        <- fmt(df$lambdaz_pred, 4)
  out[["Model"]]                 <- df$model_used

  # Flag errors
  out[["Status"]] <- ifelse(
    ok,
    '<span class="badge bg-success">OK</span>',
    paste0('<span class="badge bg-danger" title="',
           htmltools::htmlEscape(ifelse(is.na(df$error_msg), "", df$error_msg)),
           '">Error</span>')
  )

  out
}

# ── "lower – upper" string helper ─────────────────────────────────────────────
ci_string <- function(lower, upper, digits) {
  ifelse(
    is.na(lower) | is.na(upper),
    "—",
    sprintf("[%s, %s]",
            formatC(lower, digits, format = "f"),
            formatC(upper, digits, format = "f"))
  )
}

# ── Null coalescing (also defined in api_client.R — safe to duplicate) ────────
`%||%` <- function(a, b) if (!is.null(a)) a else b
