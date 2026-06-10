##############################################################################
# shiny/R/nca_compute.R
# =====================
# Phase 2 of the Wadhams NCA module: feed the standardised table into PKNCA
# and return tidy non-compartmental results.
#
# Expects the data.frame produced by format_nca_data() (standardised to
# hours / ng·mL⁻¹ / ng). Splits it into observation and dosing records,
# builds PKNCAconc / PKNCAdose objects, and runs pk.nca().
#
# PKNCA is an optional dependency. nca_available() lets the UI degrade
# gracefully when it is not installed (e.g. a stripped shinyapps.io image).
##############################################################################

# ── Availability check (mirrors the HAS_* pattern in global.R) ────────────────
nca_available <- function() requireNamespace("PKNCA", quietly = TRUE)

# Parameters requested per interval. Kept to the widely-understood core set so
# the results table stays readable; extend here if users ask for more.
NCA_PARAMS <- c(
  "cmax",            # max concentration
  "tmax",            # time of cmax
  "tlast",           # time of last measurable conc
  "clast.obs",       # last observed conc
  "auclast",         # AUC to last measurable
  "aucinf.obs",      # AUC extrapolated to infinity (observed)
  "half.life",       # terminal half-life
  "cl.obs",          # clearance (dose / AUCinf)
  "vss.obs",         # volume of distribution at steady state
  "vz.obs",          # terminal volume of distribution
  "mrt.obs"          # mean residence time
)

# Friendly labels + the standardised unit each parameter comes out in,
# given the hours / ng·mL⁻¹ / ng basis (see nca_units.R::nca_basis_label).
NCA_PARAM_META <- list(
  cmax       = list(label = "Cmax",        unit = "ng/mL"),
  tmax       = list(label = "Tmax",        unit = "h"),
  tlast      = list(label = "Tlast",       unit = "h"),
  clast.obs  = list(label = "Clast",       unit = "ng/mL"),
  auclast    = list(label = "AUClast",     unit = "ng·h/mL"),
  aucinf.obs = list(label = "AUCinf",      unit = "ng·h/mL"),
  half.life  = list(label = "t½",          unit = "h"),
  cl.obs     = list(label = "CL",          unit = "mL/h"),
  vss.obs    = list(label = "Vss",         unit = "mL"),
  vz.obs     = list(label = "Vz",          unit = "mL"),
  mrt.obs    = list(label = "MRT",         unit = "h")
)

# ── Run NCA ───────────────────────────────────────────────────────────────────
# formatted   data.frame from format_nca_data()
# returns     list(results = tidy data.frame, raw = PKNCAresults, messages = chr)
run_nca <- function(formatted) {
  if (!nca_available()) {
    stop("The PKNCA package is not installed. Install it with install.packages('PKNCA').")
  }

  obs  <- formatted[formatted$EVID == 0L, c("ID", "TIME", "CONC"), drop = FALSE]
  dose <- formatted[formatted$EVID == 1L, c("ID", "TIME", "DOSE", "ROUTE"), drop = FALSE]

  if (nrow(obs) == 0)  stop("No observation records (EVID = 0) found.")
  if (nrow(dose) == 0) stop("No dosing records (EVID = 1) found — NCA needs a dose.")

  # Drop NA concentrations (e.g. excluded BLQ) so PKNCA doesn't choke
  obs <- obs[!is.na(obs$CONC), , drop = FALSE]

  msgs <- character(0)

  # Map our route strings to PKNCA's expectation ("intravascular"/"extravascular")
  route_lookup <- function(r) {
    r <- tolower(as.character(r))
    ifelse(r %in% c("iv", "ivb", "iv bolus", "intravenous", "intravascular"),
           "intravascular", "extravascular")
  }
  dose$pknca_route <- route_lookup(dose$ROUTE)

  # Add a time-0 concentration where the first sample is post-dose, otherwise
  # PKNCA refuses to integrate AUC from the dose time. IV bolus → log-linear
  # back-extrapolated C0; extravascular → C0 = 0.
  c0_added <- .add_t0_concentration(obs, dose)
  obs      <- c0_added$obs
  if (c0_added$n_iv > 0) {
    msgs <- c(msgs, sprintf(
      "Back-extrapolated C0 for %d IV subject(s) so AUC integrates from the dose time.",
      c0_added$n_iv))
  }

  conc_obj <- PKNCA::PKNCAconc(obs,  CONC ~ TIME | ID)

  # PKNCAdose: include route so IV vs extravascular parameters resolve correctly
  dose_obj <- PKNCA::PKNCAdose(
    dose, DOSE ~ TIME | ID,
    route = "pknca_route"
  )

  data_obj <- PKNCA::PKNCAdata(conc_obj, dose_obj)

  # Replace auto-intervals with our explicit parameter request across the full span
  intervals_tpl <- data_obj$intervals
  for (p in NCA_PARAMS) {
    if (p %in% names(intervals_tpl)) intervals_tpl[[p]] <- TRUE
  }
  data_obj$intervals <- intervals_tpl

  res <- suppressWarnings(PKNCA::pk.nca(data_obj))

  tidy <- .tidy_nca_results(res)
  list(results = tidy, raw = res, messages = msgs)
}

# ── Reshape PKNCAresults into a subject × parameter table ─────────────────────
.tidy_nca_results <- function(res) {
  long <- as.data.frame(res$result, stringsAsFactors = FALSE)

  # PKNCA long format has columns: ID, start, end, PPTESTCD, PPORRES (names vary
  # slightly by version) — normalise the parameter/value column names.
  pcol <- intersect(c("PPTESTCD", "ppt", "parameter"), names(long))[1]
  vcol <- intersect(c("PPORRES", "ppx", "value"),      names(long))[1]
  if (is.na(pcol) || is.na(vcol)) {
    # Fall back: assume last two columns are parameter, value
    pcol <- names(long)[ncol(long) - 1]
    vcol <- names(long)[ncol(long)]
  }

  long <- long[long[[pcol]] %in% NCA_PARAMS, , drop = FALSE]

  # Wide: one row per subject
  wide <- reshape(
    long[, c("ID", pcol, vcol)],
    idvar     = "ID",
    timevar   = pcol,
    direction = "wide"
  )
  names(wide) <- sub(paste0("^", vcol, "\\."), "", names(wide))

  # Order columns per NCA_PARAMS and apply friendly labels + units
  present <- intersect(NCA_PARAMS, names(wide))
  wide <- wide[, c("ID", present), drop = FALSE]

  pretty <- vapply(present, function(p) {
    m <- NCA_PARAM_META[[p]]
    sprintf("%s (%s)", m$label, m$unit)
  }, character(1))
  names(wide) <- c("Subject", pretty)

  # Round numeric columns for display
  num <- vapply(wide, is.numeric, logical(1))
  wide[num] <- lapply(wide[num], function(x) signif(x, 4))
  rownames(wide) <- NULL
  wide
}

# ── Insert a time-0 concentration per subject so AUC integrates from dose ──────
# obs   data.frame(ID, TIME, CONC)
# dose  data.frame(ID, TIME, DOSE, pknca_route)
# Returns list(obs = augmented obs, n_iv = # IV C0s back-extrapolated)
.add_t0_concentration <- function(obs, dose) {
  n_iv <- 0L
  add_rows <- list()

  for (sid in unique(obs$ID)) {
    s_obs  <- obs[obs$ID == sid, , drop = FALSE]
    s_dose <- dose[dose$ID == sid, , drop = FALSE]
    if (nrow(s_dose) == 0) next

    t_dose <- min(s_dose$TIME, na.rm = TRUE)
    route  <- s_dose$pknca_route[which.min(s_dose$TIME)]

    # Already have a sample at (or before) the dose time? leave it alone
    if (any(s_obs$TIME <= t_dose)) next

    s_obs <- s_obs[order(s_obs$TIME), , drop = FALSE]

    if (route == "intravascular") {
      # Log-linear back-extrapolation from first two positive, declining points
      pos <- s_obs[s_obs$CONC > 0, , drop = FALSE]
      c0 <- NA_real_
      if (nrow(pos) >= 2) {
        t1 <- pos$TIME[1]; c1 <- pos$CONC[1]
        t2 <- pos$TIME[2]; c2 <- pos$CONC[2]
        if (c2 < c1 && t2 > t1) {
          slope <- (log(c2) - log(c1)) / (t2 - t1)         # negative
          c0 <- exp(log(c1) - slope * (t1 - t_dose))
        }
      }
      if (is.na(c0) && nrow(pos) >= 1) c0 <- pos$CONC[1]   # fallback: first conc
      if (!is.na(c0)) {
        add_rows[[length(add_rows) + 1]] <-
          data.frame(ID = sid, TIME = t_dose, CONC = c0)
        n_iv <- n_iv + 1L
      }
    } else {
      # Extravascular: concentration at dose time is zero
      add_rows[[length(add_rows) + 1]] <-
        data.frame(ID = sid, TIME = t_dose, CONC = 0)
    }
  }

  if (length(add_rows) > 0) {
    obs <- rbind(obs, do.call(rbind, add_rows))
    obs <- obs[order(obs$ID, obs$TIME), , drop = FALSE]
    rownames(obs) <- NULL
  }
  list(obs = obs, n_iv = n_iv)
}
