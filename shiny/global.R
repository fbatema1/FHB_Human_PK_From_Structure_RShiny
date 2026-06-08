##############################################################################
# shiny/global.R
# ==============
# Loaded once on app start (before ui.R and server.R).
# Good place for package loading, options, and one-time setup.
##############################################################################

# ── Required packages ─────────────────────────────────────────────────────────
required_pkgs <- c(
  "shiny",
  "bslib",
  "bsicons",
  "shinyjs",
  "DT",
  "plotly",
  "httr2",
  "jsonlite",
  "dplyr",
  "tibble",
  "htmltools"
)

missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    "Missing R packages — install with:\n",
    "install.packages(c(",
    paste0('"', missing_pkgs, '"', collapse = ", "),
    "))"
  )
}

invisible(lapply(required_pkgs, library, character.only = TRUE))

# ── App-level options ─────────────────────────────────────────────────────────
options(
  shiny.maxRequestSize = 50 * 1024^2,   # 50 MB max CSV upload
  DT.options = list(pageLength = 20)
)

# ── Development mode ──────────────────────────────────────────────────────────
# Set PK_DEV_MOCK=1 to use mock API responses (no Python backend needed)
DEV_MOCK <- nchar(Sys.getenv("PK_DEV_MOCK")) > 0
if (DEV_MOCK) {
  message("[global.R] Development mock mode ON — API calls will return fake data")
}
