##############################################################################
# shiny/ui.R
# ==========
# PK Predictor — R Shiny UI
#
# Layout:
#   - bslib Bootstrap 5 theme for a clean, publication-ready look
#   - Sidebar: SMILES input OR batch CSV upload, model selector, predict button
#   - Main: tabbed panel → Results | Interval Plot | About
##############################################################################

library(shiny)
library(bslib)
library(DT)
library(plotly)

# ── Colour palette (Wong 2011 — colour-blind safe) ────────────────────────────
CLR <- list(
  RF     = "#0072B2",
  XGB    = "#E69F00",
  GNN    = "#009E73",
  Hybrid = "#CC79A7",
  ci     = "#D55E00",
  bg     = "#F8F9FA"
)

# ── Model choices ─────────────────────────────────────────────────────────────
MODEL_CHOICES <- c(
  "Hybrid (best per parameter)" = "hybrid",
  "Random Forest"               = "rf",
  "XGBoost"                     = "xgb",
  "GNN (AttentiveFP)"           = "gnn"
)

ui <- page_navbar(
  title = tags$span(
    tags$img(src = "logo.png", height = "30px",
             style = "margin-right:8px; vertical-align:middle;",
             onerror = "this.style.display='none'"),   # hide if no logo yet
    "Wadhams — A PK Prediction Platform"
  ),
  theme = bs_theme(
    bootswatch  = "flatly",
    primary     = "#0072B2",
    base_font   = font_google("Inter"),
    code_font   = font_google("Fira Code"),
    font_scale  = 0.95
  ),
  bg       = "#2C3E50",
  fillable = TRUE,

  # ── PREDICT tab ──────────────────────────────────────────────────────────────
  nav_panel(
    title = tagList(bsicons::bs_icon("activity"), " Predict"),
    value = "predict_tab",

    layout_sidebar(
      fillable = TRUE,

      # ── Sidebar ─────────────────────────────────────────────────────────────
      sidebar = sidebar(
        width = 320,
        bg    = CLR$bg,

        # Input mode toggle
        div(
          class = "mb-3",
          tags$label("Input mode", class = "form-label fw-semibold"),
          radioButtons(
            "input_mode",
            label   = NULL,
            choices = c("Single SMILES" = "single", "Batch CSV upload" = "batch"),
            inline  = TRUE
          )
        ),

        # ── Single SMILES panel ───────────────────────────────────────────────
        conditionalPanel(
          condition = "input.input_mode == 'single'",

          div(
            class = "mb-3",
            tags$label("Compound name", class = "form-label fw-semibold"),
            textInput(
              "compound_name",
              label       = NULL,
              placeholder = "e.g. Ibuprofen",
              width       = "100%"
            )
          ),

          div(
            class = "mb-3",
            tags$label("SMILES string", class = "form-label fw-semibold"),
            tags$textarea(
              id          = "smiles_input",
              class       = "form-control font-monospace",
              placeholder = "e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O",
              rows        = 3,
              style       = "font-size:0.82rem; resize:vertical;"
            ),
            tags$div(
              class = "mt-1",
              actionLink("load_example", "Load example (Ibuprofen)",
                         style = "font-size:0.82rem;")
            )
          )
        ),

        # ── Batch CSV panel ───────────────────────────────────────────────────
        conditionalPanel(
          condition = "input.input_mode == 'batch'",

          div(
            class = "mb-3",
            tags$label("Upload CSV", class = "form-label fw-semibold"),
            fileInput(
              "csv_upload",
              label       = NULL,
              accept      = ".csv",
              placeholder = "Choose CSV file…"
            ),
            tags$small(
              class = "text-muted",
              "Required columns: ", tags$code("smiles"),
              ". Optional: ", tags$code("name"), "."
            )
          ),

          # CSV preview
          uiOutput("csv_preview_ui")
        ),

        hr(),

        # ── Model selector ────────────────────────────────────────────────────
        div(
          class = "mb-3",
          tags$label("Model", class = "form-label fw-semibold"),
          selectInput(
            "model_choice",
            label   = NULL,
            choices = MODEL_CHOICES,
            width   = "100%"
          ),
          uiOutput("model_badge_ui")
        ),

        # ── Confidence interval toggle ────────────────────────────────────────
        div(
          class = "mb-3",
          checkboxInput(
            "show_ci",
            label = "Show 95% prediction intervals",
            value = TRUE
          )
        ),

        # ── Predict button ────────────────────────────────────────────────────
        actionButton(
          "predict_btn",
          label = tagList(bsicons::bs_icon("play-fill"), " Predict"),
          class = "btn btn-primary w-100 fw-semibold",
          style = "font-size:1rem;"
        ),

        tags$div(id = "predict_spinner", class = "text-center mt-2",
                 style = "display:none;",
                 tags$div(class = "spinner-border spinner-border-sm text-primary",
                          role = "status")),

        hr(),

        # ── Download buttons ──────────────────────────────────────────────────
        tags$label("Export", class = "form-label fw-semibold"),
        div(
          class = "d-grid gap-2",
          downloadButton("dl_csv",  "Download CSV",
                         class = "btn btn-outline-secondary btn-sm"),
          downloadButton("dl_json", "Download JSON",
                         class = "btn btn-outline-secondary btn-sm")
        )
      ),   # end sidebar

      # ── Main panel ───────────────────────────────────────────────────────────
      navset_card_tab(
        id = "results_tabs",

        # ── Results table ─────────────────────────────────────────────────────
        nav_panel(
          title = tagList(bsicons::bs_icon("table"), " Results"),

          uiOutput("results_header_ui"),
          br(),
          DTOutput("results_table"),

          # Footnote
          tags$div(
            class = "text-muted mt-2",
            style = "font-size:0.78rem;",
            tags$sup("†"),
            " Prediction intervals are 95% split conformal PIs (distribution-free). ",
            "All PK parameters predicted from 2D molecular structure (SMILES). ",
            "t½ and λz derived from CL and Vd estimates."
          )
        ),

        # ── Interval plot ─────────────────────────────────────────────────────
        nav_panel(
          title = tagList(bsicons::bs_icon("bar-chart-line"), " Interval Plot"),

          fluidRow(
            column(
              3,
              selectInput(
                "plot_param",
                "Parameter",
                choices = c("CL (mL/min/kg)" = "CL",
                            "Vd (L/kg)"       = "Vd",
                            "t½ (h)"          = "thalf",
                            "λz (1/h)"        = "lambdaz"),
                width = "100%"
              )
            ),
            column(
              3,
              selectInput(
                "plot_scale",
                "Scale",
                choices = c("Original" = "original", "Log₁₀" = "log10"),
                width = "100%"
              )
            ),
            column(6)
          ),

          plotlyOutput("interval_plot", height = "500px")
        ),

        # ── Structure viewer ──────────────────────────────────────────────────
        nav_panel(
          title = tagList(bsicons::bs_icon("eye"), " Structures"),

          uiOutput("structure_warning_ui"),

          fluidRow(
            column(
              3,
              numericInput(
                "struct_page",
                "Page",
                value = 1, min = 1, step = 1,
                width = "100%"
              )
            ),
            column(9)
          ),

          uiOutput("structure_grid_ui")
        )
      )   # end navset_card_tab
    )   # end layout_sidebar
  ),   # end Predict nav_panel

  # ── ABOUT tab ────────────────────────────────────────────────────────────────
  nav_panel(
    title = tagList(bsicons::bs_icon("info-circle"), " About"),
    value = "about_tab",

    layout_column_wrap(
      width = 1/2,
      fill  = FALSE,

      card(
        card_header(tagList(bsicons::bs_icon("cpu"), " Models")),
        card_body(
          tags$p(
            "Three machine-learning models were independently trained and validated ",
            "on a curated dataset of human intravenous PK measurements:"
          ),
          tags$ul(
            tags$li(tags$strong("Random Forest"), " — 300-trial Optuna tuning, SHAP-selected features, scikit-learn."),
            tags$li(tags$strong("XGBoost"), " — 300-trial Optuna tuning, same SHAP feature set."),
            tags$li(tags$strong("GNN (AttentiveFP)"), " — 175-trial Optuna tuning, PyTorch Geometric.")
          ),
          tags$p(
            "The ", tags$strong("Hybrid"), " predictor routes each PK parameter ",
            "(CL, Vd) to the best-performing model as determined on the held-out test set."
          )
        )
      ),

      card(
        card_header(tagList(bsicons::bs_icon("rulers"), " Features")),
        card_body(
          tags$p("Molecular features computed from 2D structure (SMILES):"),
          tags$ul(
            tags$li("162 RDKit 2D physicochemical descriptors"),
            tags$li("2048-bit Morgan fingerprint (radius 2)"),
            tags$li("PyG molecular graphs: 66 atom + 11 bond features (GNN only)")
          ),
          tags$p("All PK targets are modelled on the log₁₀ scale.")
        )
      ),

      card(
        card_header(tagList(bsicons::bs_icon("shield-check"), " Uncertainty")),
        card_body(
          tags$p(
            "95% prediction intervals use ", tags$strong("split conformal prediction"),
            " (Papadopoulos et al., 2002; Angelopoulos & Bates, 2023)."
          ),
          tags$ul(
            tags$li("Distribution-free — no parametric assumptions on residuals."),
            tags$li("Finite-sample marginal coverage guarantee: P(y ∈ PI) ≥ 0.95."),
            tags$li("Calibrated on a 15% holdout of the training set (~170 compounds).")
          )
        )
      ),

      card(
        card_header(tagList(bsicons::bs_icon("bar-chart"), " Performance")),
        card_body(
          tags$p("Target metrics (evaluated on held-out test set):"),
          tags$ul(
            tags$li("GMFE < 1.5"),
            tags$li("R² > 0.70"),
            tags$li("Within-2-fold > 60%")
          ),
          uiOutput("perf_table_ui")
        )
      )
    ),

    br(),

    card(
      card_header(tagList(bsicons::bs_icon("book"), " Citation")),
      card_body(
        tags$p(
          tags$em("Manuscript in preparation."),
          " If you use this tool, please cite:"
        ),
        tags$pre(
          class = "bg-light p-2 rounded",
          style = "font-size:0.82rem;",
          "Bateman F et al. (2025). PK Predictor: A multi-model ensemble for\n",
          "human pharmacokinetic parameter prediction from molecular structure.\n",
          "Manuscript in preparation."
        ),
        hr(),
        tags$p(
          class = "text-muted",
          style = "font-size:0.82rem;",
          "Source code: ",
          tags$a("github.com/francisbateman/pk-predictor",
                 href   = "https://github.com/francisbateman/pk-predictor",
                 target = "_blank"),
          " | Contact: fbatema1@unc.edu"
        )
      )
    )
  ),   # end About nav_panel

  # ── Persistent footer ─────────────────────────────────────────────────────────
  nav_spacer(),
  nav_item(
    tags$span(
      class = "text-white-50",
      style = "font-size:0.78rem; line-height:1.6; text-align:right;",
      "For research use only — not a clinical tool",
      tags$br(),
      "Created and maintained by Francis Henry Bateman"
    )
  )
)
