##############################################################################
# shiny/global.R
# ==============
# Loaded once on app start (before ui.R and server.R).
##############################################################################

# ── Required packages ─────────────────────────────────────────────────────────
required_pkgs <- c(
  "shiny", "bslib", "bsicons", "shinyjs",
  "DT", "plotly", "httr2", "jsonlite",
  "dplyr", "tibble", "htmltools",
  "r3dmol", "reticulate", "readxl", "digest"
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
  shiny.maxRequestSize = 50 * 1024^2,
  DT.options = list(pageLength = 20)
)

# ── Development mock mode ─────────────────────────────────────────────────────
DEV_MOCK <- nchar(Sys.getenv("PK_DEV_MOCK")) > 0
if (DEV_MOCK) message("[global.R] Dev mock mode ON")

# ── Load training reference dataset ──────────────────────────────────────────
# Check local shiny/data/ first (used on shinyapps.io), then fall back to
# the project-level processed folder (used in local development).
REFERENCE_PATH <- if (file.exists("data/training_reference.csv")) {
  "data/training_reference.csv"
} else {
  file.path("..", "data", "processed", "training_reference.csv")
}

TRAINING_REF <- tryCatch({
  df <- read.csv(REFERENCE_PATH, stringsAsFactors = FALSE)
  message(sprintf("[global.R] Training reference loaded: %d compounds", nrow(df)))
  df
}, error = function(e) {
  warning("[global.R] Could not load training_reference.csv: ", conditionMessage(e))
  NULL
})

# Named vector for selectize: "Compound Name" -> "SMILES"
REFERENCE_CHOICES <- if (!is.null(TRAINING_REF)) {
  setNames(TRAINING_REF$smiles, TRAINING_REF$name)
} else {
  character(0)
}
