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
    "Wadhams — A Pharmacometrics Platform"
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
            choices = c("Single"  = "single",
                        "Compare" = "compare",
                        "Batch CSV" = "batch"),
            inline  = TRUE
          )
        ),

        # ── Single SMILES panel ───────────────────────────────────────────────
        conditionalPanel(
          condition = "input.input_mode == 'single'",

          # Repository search
          div(
            class = "mb-2",
            tags$label(
              tagList(bsicons::bs_icon("database"), " Search training library"),
              class = "form-label fw-semibold"
            ),
            selectizeInput(
              "repo_search",
              label   = NULL,
              choices = NULL,    # populated server-side for performance
              options = list(
                placeholder    = "Type a compound name…",
                maxOptions     = 20,
                searchField    = "label",
                valueField     = "value",
                labelField     = "label",
                render = I("{
                  option: function(item, escape) {
                    return '<div><strong>' + escape(item.label) + '</strong></div>';
                  }
                }")
              ),
              width = "100%"
            ),
            tags$small(
              class = "text-muted",
              sprintf("Search %s training compounds — autofills Name & SMILES",
                      if (exists("TRAINING_REF") && !is.null(TRAINING_REF))
                        formatC(nrow(TRAINING_REF), format="d", big.mark=",")
                      else "?")
            )
          ),

          tags$hr(style = "margin: 0.6rem 0;"),

          # Compound name + SMILES manual entry
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
              class = "mt-1 d-flex gap-3",
              actionLink("load_example", "Load example (Ibuprofen)",
                         style = "font-size:0.82rem;"),
              actionLink("clear_inputs", "Clear",
                         style = "font-size:0.82rem; color:#6c757d;")
            )
          )
        ),

        # ── Compare panel ─────────────────────────────────────────────────────
        conditionalPanel(
          condition = "input.input_mode == 'compare'",

          div(
            class = "mb-2",
            tags$label(
              tagList(bsicons::bs_icon("plus-circle"), " Compounds to compare"),
              class = "form-label fw-semibold"
            ),
            tags$small(class = "text-muted d-block mb-2",
                       "Up to 8 compounds. Each row: name + SMILES."),
            uiOutput("compare_rows_ui"),
            div(
              class = "d-flex gap-2 mt-2",
              actionButton("add_compound_btn", "＋ Add compound",
                           class = "btn btn-outline-primary btn-sm"),
              actionButton("remove_compound_btn", "－ Remove last",
                           class = "btn btn-outline-secondary btn-sm")
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
              accept      = c(".csv", "text/csv", "text/plain"),
              placeholder = "Choose CSV file…"
            ),
            tags$small(
              class = "text-muted",
              "Any CSV with a SMILES column. Column names are mapped below."
            )
          ),

          # ── Column mapping panel (appears after upload) ───────────────────
          uiOutput("csv_col_map_ui"),

          # ── Mapped preview ────────────────────────────────────────────────
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
                            "Vd (L/kg)"       = "Vd"),
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

          uiOutput("plot_page_ui"),
          plotlyOutput("interval_plot", height = "500px"),

          hr(),

          # Derived parameters table (t½ and λz — no CI, shown as table)
          tags$h6(class = "fw-semibold mt-2",
                  bsicons::bs_icon("table"), " Derived parameters"),
          tags$small(class = "text-muted d-block mb-2",
                     "t½ = 0.693 × Vd / CL    |    λz = CL / Vd    |    No prediction interval (derived from CL and Vd estimates)"),
          DTOutput("derived_table")
        ),

        # ── Structure viewer (3D — pure browser, no Python) ───────────────────
        nav_panel(
          title = tagList(bsicons::bs_icon("badge-3d"), " Structures"),

          # 3Dmol.js from CDN — works on any browser, no R packages needed
          tags$script(src = "https://3dmol.org/build/3Dmol-min.js"),
          tags$script(src = "molviewer.js"),

          fluidRow(
            # Left: compound selector + display controls
            column(
              4,
              div(
                class = "mb-3",
                tags$label("Select compound", class = "form-label fw-semibold"),
                uiOutput("struct_compound_select_ui")
              ),
              uiOutput("struct_metadata_ui"),
              hr(),
              tags$label("Display style", class = "form-label fw-semibold"),
              radioButtons(
                "viewer_style",
                label    = NULL,
                choices  = c("Stick" = "stick", "Sphere" = "sphere",
                             "Line"  = "line",  "Surface" = "surface"),
                selected = "stick",
                inline   = TRUE
              ),
              tags$label("Colour scheme", class = "form-label fw-semibold mt-1"),
              radioButtons(
                "viewer_colour",
                label    = NULL,
                choices  = c("Element"    = "element",
                             "Spectrum"   = "spectrum",
                             "Monochrome" = "mono"),
                selected = "element",
                inline   = TRUE
              ),
              # Colour picker — only visible in Monochrome mode
              conditionalPanel(
                condition = "input.viewer_colour == 'mono'",
                div(
                  class = "mt-1 d-flex align-items-center gap-2",
                  tags$label("Pick colour:",
                             class = "form-label mb-0",
                             style = "font-size:0.82rem; white-space:nowrap;"),
                  tags$input(
                    id      = "mono_colour",
                    type    = "color",
                    value   = "#0072B2",
                    class   = "form-control form-control-color",
                    style   = "width:42px; height:32px; padding:2px; cursor:pointer;",
                    oninput = "Shiny.setInputValue('mono_colour', this.value)"
                  )
                )
              ),
              hr(),
              tags$small(
                class = "text-muted",
                bsicons::bs_icon("globe"),
                " 3D structure from PubChem / NIH CACTUS.",
                tags$br(),
                "Works on mobile and all browsers."
              )
            ),

            # Right: 3D viewer div (3Dmol.js renders into this)
            column(
              8,
              # Spinner shown while fetching
              div(
                id    = "mol_3d_spinner",
                style = "display:none; align-items:center; justify-content:center;
                         height:60px; gap:10px;",
                tags$div(class = "spinner-border spinner-border-sm text-primary"),
                tags$span(id = "mol_3d_status",
                          class = "text-muted",
                          style = "font-size:0.85rem;",
                          "Loading…")
              ),
              # Status line (errors appear here)
              tags$p(id    = "mol_3d_status",
                     class = "text-muted mb-1",
                     style = "font-size:0.82rem; min-height:1.2em;"),

              # The viewer canvas — 3Dmol.js targets this div by id
              div(
                id    = VIEWER_ID <- "mol_3d_viewer",
                style = "width:100%; height:480px; border-radius:0.4rem;
                         border:1px solid #DEE2E6; background:#fff;
                         position:relative; overflow:hidden;"
              ),

              # Color legend (populated by molviewer.js)
              div(
                id    = "mol_legend",
                class = "mt-2 d-flex flex-wrap align-items-center",
                style = "min-height:1.4rem;"
              ),
              tags$small(
                class = "text-muted mt-1 d-block",
                "Drag to rotate · Scroll to zoom · Right-click to pan"
              ),
              div(
                class = "mt-2 p-2 rounded",
                style = "background:#FFF8E1; border:1px solid #FFE082; font-size:0.76rem; color:#795548;",
                bsicons::bs_icon("exclamation-triangle"),
                tags$strong(" Disclaimer:"),
                " 3D structures in Wadhams v1.0 are retrieved from PubChem / NIH CACTUS
                  and are ", tags$em("not"), " generated by a model developed by F.H. Bateman.
                  Structures may be incomplete or incorrect for novel or rare compounds.
                  Always verify structures independently before use."
              )
            )
          )
        )
      )   # end navset_card_tab
    )   # end layout_sidebar
  ),   # end Predict nav_panel

  # ── NCA tab ──────────────────────────────────────────────────────────────────
  nav_panel(
    title = tagList(bsicons::bs_icon("calculator"), " NCA"),
    value = "nca_tab",

    layout_sidebar(
      fillable = TRUE,

      # ── Sidebar: upload → map → units → limits → generate ──────────────────
      sidebar = sidebar(
        width = 340,
        bg    = CLR$bg,

        tags$h6(bsicons::bs_icon("1-circle-fill"), " Upload raw data",
                class = "fw-bold text-primary mt-1"),
        fileInput(
          "nca_file",
          label    = NULL,
          accept   = c(".csv", ".xlsx", ".xls"),
          buttonLabel = "Browse…",
          placeholder = "CSV or Excel"
        ),
        tags$p(
          class = "text-muted",
          style = "font-size:0.78rem;",
          "Raw LCMS output with subject IDs, time, concentration, and dose. ",
          "Any column names — you map them below."
        ),

        # Column mapping + units + limits are rendered once a file is loaded
        uiOutput("nca_config_ui"),

        # Generate button appears with config
        uiOutput("nca_generate_ui")
      ),

      # ── Main: two phases ────────────────────────────────────────────────────
      navset_card_tab(
        id = "nca_phase",

        # Phase 1 — formatted table
        nav_panel(
          title = tagList(bsicons::bs_icon("table"), " Formatted Data"),
          value = "nca_format_panel",
          div(
            class = "p-2",
            uiOutput("nca_format_status"),
            DTOutput("nca_formatted_table"),
            uiOutput("nca_format_downloads")
          )
        ),

        # Phase 2 — NCA results
        nav_panel(
          title = tagList(bsicons::bs_icon("clipboard-data"), " NCA Results"),
          value = "nca_results_panel",
          div(
            class = "p-2",
            uiOutput("nca_results_status"),
            DTOutput("nca_results_table"),
            uiOutput("nca_results_downloads")
          )
        )
      )
    )
  ),   # end NCA nav_panel

  # ── ABOUT tab ────────────────────────────────────────────────────────────────
  nav_panel(
    title = tagList(bsicons::bs_icon("info-circle"), " About"),
    value = "about_tab",

    # ── Intended use disclaimer ───────────────────────────────────────────────
    div(
      class = "alert alert-warning mx-3 mt-3",
      style = "font-size:0.85rem;",
      tags$strong(bsicons::bs_icon("exclamation-triangle-fill"), " Intended Use — Wadhams v1.0"),
      tags$p(
        class = "mb-0 mt-1",
        "Wadhams is designed as a ", tags$strong("decision-support tool for early-stage drug discovery"),
        " only. It provides rapid, structure-based estimation of human PK parameters to guide ",
        "SAR decisions ", tags$em("before"), " any in vitro or in vivo data are available. ",
        "Predictions should not be used for clinical dose selection, regulatory submissions, ",
        "or any purpose requiring precise PK characterisation. Always confirm with experimental data."
      )
    ),

    layout_column_wrap(
      width = 1/2,
      fill  = FALSE,

      # ── Hybrid model structure ────────────────────────────────────────────
      card(
        card_header(tagList(bsicons::bs_icon("diagram-3"), " Hybrid Model Architecture")),
        card_body(
          tags$p(
            "Three models were independently trained on a curated dataset of ",
            tags$strong("1,147 human intravenous PK measurements"),
            " and evaluated on a held-out test set of ", tags$strong("287 compounds."),
            " The Hybrid predictor routes each parameter to the best-performing model:"
          ),
          tags$table(
            class = "table table-sm table-bordered mb-2",
            style = "font-size:0.82rem;",
            tags$thead(class = "table-light",
              tags$tr(tags$th("Parameter"), tags$th("Model"), tags$th("Rationale"))
            ),
            tags$tbody(
              tags$tr(
                tags$td(tags$strong("CL")),
                tags$td(tags$span(class="badge rounded-pill", style="background:#0072B2;", "Random Forest")),
                tags$td("Best GMFE and within-2-fold on test set")
              ),
              tags$tr(
                tags$td(tags$strong("Vd")),
                tags$td(tags$span(class="badge rounded-pill", style="background:#E69F00;color:#000;", "XGBoost")),
                tags$td("Best GMFE, R², and within-2-fold on test set")
              )
            )
          ),
          tags$p(
            style = "font-size:0.8rem;",
            tags$strong("Features:"),
            " Top SHAP-selected RDKit 2D descriptors (CL: 50, Vd: 150 for XGB / 29 for RF) ",
            "+ full 2,048-bit Morgan fingerprint (radius 2). All targets modelled on log₁₀ scale."
          )
        )
      ),

      # ── Performance metrics ───────────────────────────────────────────────
      card(
        card_header(tagList(bsicons::bs_icon("bar-chart"), " Performance — Held-out Test Set (n=287)")),
        card_body(
          tags$p(class = "fw-semibold mb-1", style = "font-size:0.85rem;",
                 "CL (mL/min/kg) — all models:"),
          tags$table(
            class = "table table-sm table-bordered mb-3",
            style = "font-size:0.80rem;",
            tags$thead(class = "table-light",
              tags$tr(tags$th("Model"), tags$th("GMFE"), tags$th("R²"),
                      tags$th("Within 2-fold"), tags$th("Within 3-fold"))
            ),
            tags$tbody(
              tags$tr(
                tags$td(tags$span(style="color:#0072B2;font-weight:600;", "RF ✦")),
                tags$td("2.256"), tags$td("0.433"), tags$td("60.3%"), tags$td("77.4%")
              ),
              tags$tr(
                tags$td(tags$span(style="color:#E69F00;font-weight:600;", "XGB")),
                tags$td("2.272"), tags$td("0.417"), tags$td("56.8%"), tags$td("73.9%")
              ),
              tags$tr(
                tags$td(tags$span(style="color:#009E73;font-weight:600;", "GNN")),
                tags$td("2.550"), tags$td("0.312"), tags$td("48.4%"), tags$td("68.3%")
              )
            )
          ),
          tags$p(class = "fw-semibold mb-1", style = "font-size:0.85rem;",
                 "Vd (L/kg) — all models:"),
          tags$table(
            class = "table table-sm table-bordered mb-3",
            style = "font-size:0.80rem;",
            tags$thead(class = "table-light",
              tags$tr(tags$th("Model"), tags$th("GMFE"), tags$th("R²"),
                      tags$th("Within 2-fold"), tags$th("Within 3-fold"))
            ),
            tags$tbody(
              tags$tr(
                tags$td(tags$span(style="color:#0072B2;font-weight:600;", "RF")),
                tags$td("1.913"), tags$td("0.653"), tags$td("66.2%"), tags$td("82.2%")
              ),
              tags$tr(
                tags$td(tags$span(style="color:#E69F00;font-weight:600;", "XGB ✦")),
                tags$td("1.815"), tags$td("0.694"), tags$td("71.1%"), tags$td("85.0%")
              ),
              tags$tr(
                tags$td(tags$span(style="color:#009E73;font-weight:600;", "GNN")),
                tags$td("2.048"), tags$td("0.588"), tags$td("56.4%"), tags$td("78.7%")
              )
            )
          ),
          tags$small(class = "text-muted fst-italic",
                     "✦ = model used by Hybrid predictor for that parameter.")
        )
      ),

      # ── Derived parameters ────────────────────────────────────────────────
      card(
        card_header(tagList(bsicons::bs_icon("calculator"), " Derived Parameters (t½ and λz)")),
        card_body(
          tags$p(
            "t½ and λz are ", tags$strong("derived"), " from CL and Vd predictions — they are not ",
            "independently modelled:"
          ),
          tags$ul(
            style = "font-size:0.82rem;",
            tags$li("t½ (h) = 0.693 × Vd (L/kg) / CL (L/h/kg)"),
            tags$li("λz (1/h) = CL (L/h/kg) / Vd (L/kg)")
          ),
          tags$p(
            "Validation against published human t½ values (n=140, PubChem/FDA sources, ",
            "multi-compartment outliers excluded):"
          ),
          tags$table(
            class = "table table-sm table-bordered mb-2",
            style = "font-size:0.82rem;",
            tags$tbody(
              tags$tr(tags$th("GMFE"),            tags$td("2.44")),
              tags$tr(tags$th("Within 2-fold"),   tags$td("49.3%")),
              tags$tr(tags$th("Within 3-fold"),   tags$td("66.4%")),
              tags$tr(tags$th("Mean % error"),    tags$td("23.8%"))
            )
          ),
          div(
            class = "p-2 rounded",
            style = "background:#FFF8E1;border:1px solid #FFE082;font-size:0.78rem;color:#795548;",
            bsicons::bs_icon("exclamation-triangle"),
            " Approximately ", tags$strong("50% of t½ predictions fall within 2-fold"),
            " of published values. Derived parameters inherit combined uncertainty from both ",
            "CL and Vd estimates. t½ reflects one-compartment kinetics and may underestimate ",
            "the terminal half-life for multi-compartment or deep-tissue-distributed drugs."
          )
        )
      ),

      # ── Uncertainty ───────────────────────────────────────────────────────
      card(
        card_header(tagList(bsicons::bs_icon("shield-check"), " Uncertainty Quantification")),
        card_body(
          tags$p(
            "95% prediction intervals use ", tags$strong("split conformal prediction"),
            " (Papadopoulos et al., 2002; Angelopoulos & Bates, 2023)."
          ),
          tags$ul(
            tags$li("Distribution-free — no parametric assumptions on residuals."),
            tags$li("Finite-sample marginal coverage guarantee: P(y ∈ PI) ≥ 0.95."),
            tags$li("Calibrated on a 15% holdout of the training set (~172 compounds).")
          ),
          tags$p("95% PI fold widths (hybrid routing):"),
          tags$table(
            class = "table table-sm table-bordered mb-0",
            style = "font-size:0.82rem;",
            tags$thead(class="table-light",
              tags$tr(tags$th("Parameter"), tags$th("Model"), tags$th("PI fold width"))
            ),
            tags$tbody(
              tags$tr(tags$td("CL"), tags$td("RF"),  tags$td("±2.63×")),
              tags$tr(tags$td("Vd"), tags$td("XGB"), tags$td("±1.54×"))
            )
          )
        )
      ),

      # ── Features ──────────────────────────────────────────────────────────
      card(
        card_header(tagList(bsicons::bs_icon("rulers"), " Molecular Features")),
        card_body(
          tags$p("Features computed from 2D structure (SMILES) only — no 3D geometry required:"),
          tags$ul(
            tags$li("162 RDKit 2D physicochemical descriptors (MW, logP, TPSA, HBD/HBA, rotatable bonds, etc.)"),
            tags$li("SHAP-based feature selection via Optuna tuning — top descriptors retained per model/parameter"),
            tags$li("2,048-bit Morgan circular fingerprint (radius 2)"),
            tags$li("PyG molecular graphs: 66 atom + 11 bond features (GNN only)")
          ),
          tags$p(style="font-size:0.8rem;color:#6c757d;",
            "Hyperparameters tuned via 300-trial Optuna Bayesian optimisation with 5-fold ",
            "cross-validation (RF and XGB) or 175 trials (GNN)."
          )
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
      "Created and maintained by Francis Henry Bateman",
      tags$br(),
      tags$span(style = "font-size:0.68rem;", "v1.0")
    )
  )
)
