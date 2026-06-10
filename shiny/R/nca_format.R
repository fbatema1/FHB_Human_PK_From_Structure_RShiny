##############################################################################
# shiny/R/nca_format.R
# ====================
# Phase 1 of the Wadhams NCA module: turn an arbitrary raw upload into a
# standardised NONMEM-style table that PKNCA can consume.
#
# Output columns (fixed order):
#   ID     integer subject identifier
#   TIME   numeric time in HOURS (standardised)
#   CONC   numeric concentration in ng/mL (standardised); NA on dose rows
#   DOSE   numeric dose in ng (standardised); NA on observation rows
#   EVID   0 = observation, 1 = dosing event
#   ROUTE  character route (dose rows only): "iv" / "po" / etc.
#   BLQ    0/1 flag — observation below LLOQ
#   ALQ    0/1 flag — observation above ULOQ
#
# The function never silently drops data: BLQ handling is explicit and the
# returned object carries a per-row flag vector for UI highlighting.
##############################################################################

# ── Heuristic column guesser (used to pre-fill the mapping dropdowns) ─────────
# Returns a named list: id/time/conc/dose/route -> best-guess column name or NA
guess_nca_columns <- function(col_names) {
  lower <- tolower(col_names)
  pick  <- function(patterns) {
    for (p in patterns) {
      hit <- which(grepl(p, lower))
      if (length(hit) > 0) return(col_names[hit[1]])
    }
    NA_character_
  }
  list(
    id    = pick(c("^id$", "subject", "usubjid", "animal", "patient")),
    time  = pick(c("^time$", "^t$", "tad", "nominal.*time", "time")),
    conc  = pick(c("^conc$", "^dv$", "concentration", "cp", "ng_ml", "amount")),
    dose  = pick(c("^dose$", "^amt$", "amount.*dose", "dose")),
    route = pick(c("route", "^cmt$", "admin"))
  )
}

# ── Main formatter ────────────────────────────────────────────────────────────
# raw          data.frame as uploaded
# mapping      list(id=, time=, conc=, dose=, route=) column-name strings (route
#              may be NA if a constant route is supplied instead)
# units        list(time=, conc=, dose=) unit codes (see nca_units.R)
# mw           molecular weight g/mol (NA if not needed)
# bw           body weight kg (NA if not needed)
# lloq, uloq   numeric limits in the SAME unit as the raw concentration column
#              (converted internally); NA disables the check
# blq_rule     "zero" | "half_lloq" | "exclude"
# const_route  character route applied to all dose rows when mapping$route is NA
format_nca_data <- function(raw, mapping, units,
                            mw = NA_real_, bw = NA_real_,
                            lloq = NA_real_, uloq = NA_real_,
                            blq_rule = "half_lloq",
                            const_route = "iv") {

  stopifnot(is.data.frame(raw))
  req_map <- c("id", "time", "conc")
  for (k in req_map) {
    if (is.null(mapping[[k]]) || is.na(mapping[[k]]) || !(mapping[[k]] %in% names(raw))) {
      stop(sprintf("Column mapping for '%s' is missing or not found in the upload.", k))
    }
  }

  # Pull raw vectors
  id_raw   <- raw[[mapping$id]]
  time_raw <- suppressWarnings(as.numeric(raw[[mapping$time]]))
  conc_raw <- suppressWarnings(as.numeric(raw[[mapping$conc]]))
  dose_raw <- if (!is.null(mapping$dose) && !is.na(mapping$dose) &&
                  mapping$dose %in% names(raw)) {
    suppressWarnings(as.numeric(raw[[mapping$dose]]))
  } else {
    rep(NA_real_, nrow(raw))
  }
  route_raw <- if (!is.null(mapping$route) && !is.na(mapping$route) &&
                   mapping$route %in% names(raw)) {
    as.character(raw[[mapping$route]])
  } else {
    rep(NA_character_, nrow(raw))
  }

  # ── Convert limits to ng/mL up front (limits are in raw conc unit) ──────────
  lloq_std <- if (!is.na(lloq)) convert_conc_to_ng_ml(lloq, units$conc, mw) else NA_real_
  uloq_std <- if (!is.na(uloq)) convert_conc_to_ng_ml(uloq, units$conc, mw) else NA_real_

  # ── Standardise units ───────────────────────────────────────────────────────
  time_h    <- convert_time_to_hours(time_raw, units$time)
  conc_std  <- convert_conc_to_ng_ml(conc_raw, units$conc, mw)
  dose_std  <- ifelse(is.na(dose_raw), NA_real_,
                      convert_dose_to_ng(dose_raw, units$dose, mw, bw))

  n <- nrow(raw)
  rows <- vector("list", 0)

  for (i in seq_len(n)) {
    has_dose <- !is.na(dose_std[i]) && dose_std[i] > 0
    has_conc <- !is.na(conc_std[i])

    # Dose event → EVID 1 (emitted first so it precedes same-time observations)
    if (has_dose) {
      r <- as.character(route_raw[i])
      if (is.na(r) || !nzchar(r)) r <- const_route
      rows[[length(rows) + 1]] <- data.frame(
        ID = id_raw[i], TIME = time_h[i], CONC = NA_real_, DOSE = dose_std[i],
        EVID = 1L, ROUTE = tolower(r), BLQ = 0L, ALQ = 0L,
        stringsAsFactors = FALSE
      )
    }

    # Observation → EVID 0 (with BLQ / ULOQ handling)
    if (has_conc) {
      cval <- conc_std[i]
      blq  <- 0L
      alq  <- 0L

      if (!is.na(lloq_std) && cval < lloq_std) {
        blq <- 1L
        cval <- switch(blq_rule,
          "zero"      = 0,
          "half_lloq" = lloq_std / 2,
          "exclude"   = NA_real_,
          lloq_std / 2
        )
      }
      if (!is.na(uloq_std) && !is.na(conc_std[i]) && conc_std[i] > uloq_std) {
        alq <- 1L
      }

      # exclude rule drops the row entirely
      if (!(blq == 1L && blq_rule == "exclude")) {
        rows[[length(rows) + 1]] <- data.frame(
          ID = id_raw[i], TIME = time_h[i], CONC = cval, DOSE = NA_real_,
          EVID = 0L, ROUTE = NA_character_, BLQ = blq, ALQ = alq,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(rows) == 0) {
    stop("No usable records produced — check your column mapping and units.")
  }

  out <- do.call(rbind, rows)

  # Coerce ID to a clean integer factor order, then sort by ID, TIME, (dose first)
  out$ID <- as.integer(factor(out$ID, levels = unique(out$ID)))
  out <- out[order(out$ID, out$TIME, -out$EVID), , drop = FALSE]
  rownames(out) <- NULL

  attr(out, "n_blq")   <- sum(out$BLQ == 1L)
  attr(out, "n_alq")   <- sum(out$ALQ == 1L)
  attr(out, "n_dose")  <- sum(out$EVID == 1L)
  attr(out, "n_obs")   <- sum(out$EVID == 0L)
  attr(out, "n_subj")  <- length(unique(out$ID))
  out
}
