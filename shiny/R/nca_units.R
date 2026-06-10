##############################################################################
# shiny/R/nca_units.R
# ===================
# Unit conversion layer for the Wadhams NCA module.
#
# PKNCA works best when all records share a consistent mass/volume/time basis.
# We standardise to:
#   - TIME          → hours
#   - CONCENTRATION → ng/mL   (mass/volume base)
#   - DOSE          → ng      (same mass base as concentration, so that
#                              CL = dose / AUC comes out in mL/time and
#                              Vd in mL without further juggling)
#
# Molar concentration/dose units require molecular weight (g/mol).
# Per-body-weight doses (mg/kg etc.) require body weight (kg).
#
# Every conversion function returns a numeric vector the same length as input.
##############################################################################

# ── Supported unit choices (exposed to the UI) ───────────────────────────────
NCA_TIME_UNITS <- c(
  "Hours"   = "h",
  "Minutes" = "min",
  "Days"    = "day"
)

NCA_CONC_UNITS <- c(
  "ng/mL"   = "ng/mL",
  "µg/mL"   = "ug/mL",
  "µg/L"    = "ug/L",
  "mg/L"    = "mg/L",
  "µmol/L"  = "umol/L",
  "nmol/L"  = "nmol/L"
)

NCA_DOSE_UNITS <- c(
  "mg"      = "mg",
  "µg"      = "ug",
  "g"       = "g",
  "ng"      = "ng",
  "mg/kg"   = "mg/kg",
  "µg/kg"   = "ug/kg",
  "µmol"    = "umol",
  "nmol"    = "nmol"
)

# Whether a given unit needs molecular weight to resolve
nca_unit_needs_mw <- function(unit) {
  unit %in% c("umol/L", "nmol/L", "umol", "nmol")
}

# Whether a given dose unit needs body weight to resolve to absolute mass
nca_unit_needs_bw <- function(unit) {
  unit %in% c("mg/kg", "ug/kg")
}

# ── TIME → hours ──────────────────────────────────────────────────────────────
convert_time_to_hours <- function(x, from_unit) {
  factor <- switch(from_unit,
    "h"   = 1,
    "min" = 1 / 60,
    "day" = 24,
    stop(sprintf("Unknown time unit: %s", from_unit))
  )
  x * factor
}

# ── CONCENTRATION → ng/mL ─────────────────────────────────────────────────────
# mw = molecular weight in g/mol (required for molar units, else NA)
convert_conc_to_ng_ml <- function(x, from_unit, mw = NA_real_) {
  if (nca_unit_needs_mw(from_unit) && (is.na(mw) || mw <= 0)) {
    stop("Molecular weight (g/mol) required for molar concentration units.")
  }
  factor <- switch(from_unit,
    "ng/mL"  = 1,
    "ug/mL"  = 1000,       # 1 µg/mL = 1000 ng/mL
    "ug/L"   = 1,          # 1 µg/L  = 1 ng/mL
    "mg/L"   = 1000,       # 1 mg/L  = 1000 ng/mL
    "umol/L" = mw,         # 1 µmol/L × MW(g/mol) = MW µg/L = MW ng/mL
    "nmol/L" = mw / 1000,  # 1 nmol/L × MW = MW ng/L = MW/1000 ng/mL
    stop(sprintf("Unknown concentration unit: %s", from_unit))
  )
  x * factor
}

# ── DOSE → ng (absolute mass) ─────────────────────────────────────────────────
# mw = molecular weight g/mol (for molar dose); bw = body weight kg (for /kg dose)
convert_dose_to_ng <- function(x, from_unit, mw = NA_real_, bw = NA_real_) {
  if (nca_unit_needs_mw(from_unit) && (is.na(mw) || mw <= 0)) {
    stop("Molecular weight (g/mol) required for molar dose units.")
  }
  if (nca_unit_needs_bw(from_unit) && (is.na(bw) || bw <= 0)) {
    stop("Body weight (kg) required for per-kg dose units.")
  }
  factor <- switch(from_unit,
    "mg"    = 1e6,             # 1 mg = 1e6 ng
    "ug"    = 1e3,             # 1 µg = 1e3 ng
    "g"     = 1e9,             # 1 g  = 1e9 ng
    "ng"    = 1,
    "mg/kg" = 1e6 * bw,        # mg/kg × kg → mg → ng
    "ug/kg" = 1e3 * bw,        # µg/kg × kg → µg → ng
    "umol"  = mw * 1e3,        # 1 µmol × MW(g/mol) = MW µg = MW×1e3 ng
    "nmol"  = mw,              # 1 nmol × MW = MW ng
    stop(sprintf("Unknown dose unit: %s", from_unit))
  )
  x * factor
}

# ── Human-readable summary of the standardised basis ─────────────────────────
nca_basis_label <- function() {
  list(
    time = "h",
    conc = "ng/mL",
    dose = "ng",
    cl   = "mL/h",     # dose(ng) / AUC(ng·h/mL) = mL/h
    vd   = "mL"        # mL/h × h = mL
  )
}
