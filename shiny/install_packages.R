# Run this once to install all required R packages
# Rscript shiny/install_packages.R

pkgs <- c(
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
  "htmltools",
  "r3dmol",
  "reticulate",
  "readxl",
  "digest",
  "PKNCA"
)

install.packages(pkgs, repos = "https://cloud.r-project.org")
message("All packages installed.")
