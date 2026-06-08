##############################################################################
# shiny/server.R
# ==============
# PK Predictor — R Shiny server
#
# Architecture:
#   - Calls the FastAPI backend (api/main.py) via httr2
#   - Falls back to a local mock if the API is unreachable (development mode)
#   - All heavy ML inference stays in Python; Shiny is purely presentation
##############################################################################

library(shiny)
library(httr2)
library(jsonlite)
library(DT)
library(plotly)
library(r3dmol)
library(dplyr)
library(tibble)

source("R/api_client.R")
source("R/plots.R")
source("R/utils.R")
source("R/conformer.R")

# ── Constants ─────────────────────────────────────────────────────────────────
EXAMPLE_SMILES  <- "CC(C)Cc1ccc(cc1)C(C)C(=O)O"   # Ibuprofen
EXAMPLE_NAME    <- "Ibuprofen"
STRUCTS_PER_PAGE <- 12

# Parameter display metadata
PARAM_META <- list(
  CL      = list(label = "CL",     unit = "mL/min/kg", digits = 3),
  Vd      = list(label = "Vd",     unit = "L/kg",      digits = 3),
  thalf   = list(label = "t½",     unit = "h",         digits = 2),
  lambdaz = list(label = "λz",unit = "1/h",       digits = 4)
)


server <- function(input, output, session) {

  # ── Reactive values ──────────────────────────────────────────────────────────
  rv <- reactiveValues(
    results   = NULL,    # data.frame of predictions
    api_error = NULL,    # error message string or NULL
    busy      = FALSE
  )

  # ── Populate repo search (server-side for 1,000+ compounds) ──────────────────
  updateSelectizeInput(
    session, "repo_search",
    choices  = REFERENCE_CHOICES,
    server   = TRUE
  )

  # ── Autofill from repository search ──────────────────────────────────────────
  observeEvent(input$repo_search, {
    req(nchar(input$repo_search) > 0)
    smiles <- input$repo_search
    # Look up the name from the reference table
    idx  <- which(TRAINING_REF$smiles == smiles)[1]
    name <- if (!is.na(idx)) TRAINING_REF$name[idx] else ""
    updateTextAreaInput(session, "smiles_input",  value = smiles)
    updateTextInput(session,    "compound_name",  value = name)
  }, ignoreInit = TRUE)

  # ── Load example ─────────────────────────────────────────────────────────────
  observeEvent(input$load_example, {
    updateTextAreaInput(session, "smiles_input", value = EXAMPLE_SMILES)
    updateTextInput(session, "compound_name",   value = EXAMPLE_NAME)
  })

  # ── Clear inputs ──────────────────────────────────────────────────────────────
  observeEvent(input$clear_inputs, {
    updateTextAreaInput(session, "smiles_input",  value = "")
    updateTextInput(session,    "compound_name",  value = "")
    updateSelectizeInput(session, "repo_search",  selected = "")
  })

  # ── CSV upload preview ────────────────────────────────────────────────────────
  csv_data <- reactive({
    req(input$csv_upload)
    df <- tryCatch(
      read.csv(input$csv_upload$datapath, stringsAsFactors = FALSE),
      error = function(e) NULL
    )
    validate(need(!is.null(df), "Could not read CSV. Please check the file format."))
    validate(need("smiles" %in% tolower(names(df)),
                  "CSV must contain a column named 'smiles'."))
    names(df) <- tolower(names(df))
    df
  })

  output$csv_preview_ui <- renderUI({
    req(input$input_mode == "batch", input$csv_upload)
    df <- csv_data()
    tagList(
      tags$small(class = "text-success fw-semibold",
                 sprintf("✓ %d compound(s) loaded", nrow(df))),
      br(), br(),
      DTOutput("csv_preview_tbl")
    )
  })

  output$csv_preview_tbl <- renderDT({
    req(csv_data())
    datatable(
      head(csv_data(), 5),
      options = list(dom = "t", scrollX = TRUE),
      rownames = FALSE
    )
  })

  # ── Model badge ───────────────────────────────────────────────────────────────
  output$model_badge_ui <- renderUI({
    col <- switch(input$model_choice,
      hybrid = "#CC79A7",
      rf     = "#0072B2",
      xgb    = "#E69F00",
      gnn    = "#009E73"
    )
    tags$span(
      class = "badge rounded-pill mt-1",
      style = sprintf("background-color:%s; font-size:0.72rem;", col),
      toupper(input$model_choice)
    )
  })

  # ── PREDICT ───────────────────────────────────────────────────────────────────
  observeEvent(input$predict_btn, {
    rv$api_error <- NULL
    rv$results   <- NULL

    # Build request payload ─────────────────────────────────────────────────────
    if (input$input_mode == "single") {
      smiles_vec <- trimws(input$smiles_input)
      cname      <- trimws(input$compound_name)
      names_vec  <- if (nchar(cname) > 0) cname else "Unnamed compound"
      if (nchar(smiles_vec) == 0) {
        showNotification("Please enter a SMILES string.", type = "warning")
        return()
      }
    } else {
      df <- csv_data()
      smiles_vec <- df$smiles
      names_vec  <- if ("name" %in% names(df)) df$name else paste0("Compound_", seq_len(nrow(df)))
    }

    payload <- list(
      smiles = as.list(smiles_vec),
      model  = input$model_choice,
      ci     = isTRUE(input$show_ci)
    )

    # Call API ──────────────────────────────────────────────────────────────────
    rv$busy <- TRUE
    shinyjs::show("predict_spinner")

    result <- tryCatch(
      pk_predict(payload),
      error = function(e) {
        list(error = conditionMessage(e))
      }
    )

    rv$busy <- FALSE
    shinyjs::hide("predict_spinner")

    if (!is.null(result$error)) {
      rv$api_error <- result$error
      showNotification(
        tagList(bsicons::bs_icon("exclamation-triangle-fill"), " ", result$error),
        type     = "error",
        duration = 8
      )
      return()
    }

    # Parse response ────────────────────────────────────────────────────────────
    rv$results <- parse_predictions(result, names_vec, show_ci = input$show_ci)

    # Switch to results tab
    nav_select("results_tabs", "Results")
    showNotification(
      tagList(bsicons::bs_icon("check-circle-fill"), " Prediction complete"),
      type     = "message",
      duration = 3
    )
  })

  # ── Results header ────────────────────────────────────────────────────────────
  output$results_header_ui <- renderUI({
    if (is.null(rv$results)) {
      return(
        div(
          class = "text-center text-muted py-5",
          bsicons::bs_icon("arrow-left-circle", size = "2em"),
          br(), br(),
          tags$p("Enter a SMILES string and click ", tags$strong("Predict"), " to begin.")
        )
      )
    }
    n  <- nrow(rv$results)
    md <- input$model_choice
    tagList(
      tags$span(
        class = "badge bg-secondary me-2",
        sprintf("%d compound%s", n, if (n == 1) "" else "s")
      ),
      tags$span(
        class = "badge rounded-pill",
        style = sprintf("background-color:%s;",
                        switch(md, hybrid="#CC79A7", rf="#0072B2",
                                   xgb="#E69F00", gnn="#009E73")),
        toupper(md)
      )
    )
  })

  # ── Results table ─────────────────────────────────────────────────────────────
  output$results_table <- renderDT({
    req(rv$results)
    df <- format_results_table(rv$results, show_ci = input$show_ci)

    datatable(
      df,
      rownames  = FALSE,
      escape    = FALSE,
      selection = "single",
      options   = list(
        dom        = "Bfrtip",
        buttons    = c("csv", "excel"),
        scrollX    = TRUE,
        pageLength = 20,
        columnDefs = list(
          list(className = "dt-center",
               targets   = seq(1, ncol(df) - 1))
        )
      ),
      extensions = "Buttons"
    ) |>
      formatStyle(
        columns    = "CL (mL/min/kg)",
        background = styleColorBar(c(0, 200), "#AED6F1"),
        backgroundSize   = "100% 80%",
        backgroundRepeat = "no-repeat",
        backgroundPosition = "center"
      )
  })

  # ── Interval plot ─────────────────────────────────────────────────────────────
  output$interval_plot <- renderPlotly({
    req(rv$results)
    param  <- input$plot_param
    scale  <- input$plot_scale
    make_interval_plot(rv$results, param, scale)
  })

  # ── 3D Structure viewer ────────────────────────────────────────────────────────

  # Compound selector — built from current prediction results
  output$struct_compound_select_ui <- renderUI({
    req(rv$results)
    ok_rows <- rv$results[rv$results$status == "ok", ]
    req(nrow(ok_rows) > 0)
    choices <- setNames(ok_rows$smiles, ok_rows$name)
    selectInput("struct_selected", label = NULL,
                choices = choices, width = "100%")
  })

  # Metadata card for selected compound
  output$struct_metadata_ui <- renderUI({
    req(rv$results, input$struct_selected)
    r <- rv$results[rv$results$smiles == input$struct_selected, ][1, ]
    req(!is.null(r), nrow(r) > 0)

    # Check if this compound is in the training reference
    ref_row <- if (!is.null(TRAINING_REF)) {
      TRAINING_REF[TRAINING_REF$smiles == input$struct_selected, ]
    } else NULL

    card(
      class = "mt-2",
      card_body(
        padding = "0.6rem",
        tags$table(
          class = "table table-sm table-borderless mb-0",
          style = "font-size:0.82rem;",
          tags$tbody(
            tags$tr(tags$th("CL pred"),
                    tags$td(sprintf("%.3f mL/min/kg", r$CL_pred))),
            tags$tr(tags$th("Vd pred"),
                    tags$td(sprintf("%.3f L/kg", r$Vd_pred))),
            tags$tr(tags$th("t½ pred"),
                    tags$td(sprintf("%.2f h", r$thalf_pred))),
            if (!is.null(ref_row) && nrow(ref_row) > 0) {
              tagList(
                tags$tr(tags$th(class="text-success", "CL obs"),
                        tags$td(class="text-success",
                                sprintf("%.3f mL/min/kg", ref_row$CL_measured[1]))),
                tags$tr(tags$th(class="text-success", "Vd obs"),
                        tags$td(class="text-success",
                                sprintf("%.3f L/kg", ref_row$Vd_measured[1])))
              )
            }
          )
        ),
        if (!is.null(ref_row) && nrow(ref_row) > 0) {
          tags$small(class="text-success",
                     bsicons::bs_icon("database-check"),
                     " In training set — observed values shown in green")
        }
      )
    )
  })

  # RDKit availability warning
  output$struct_rdkit_warning_ui <- renderUI({
    if (!rdkit_available()) {
      div(class = "alert alert-warning mb-2",
          bsicons::bs_icon("exclamation-triangle"),
          " RDKit not found in pkip-env. 3D viewer unavailable. ",
          "Run: ", tags$code("conda activate pkip-env"), " before launching Shiny.")
    }
  })

  # 3D viewer — render on compound selection or style change
  output$viewer_3d <- renderR3dmol({
    req(rv$results, input$struct_selected)

    smiles   <- input$struct_selected
    style_in <- input$viewer_style   %||% "stick"
    colour   <- input$viewer_colour  %||% "element"

    # Generate mol-block (cached after first call)
    mb <- get_molblock(smiles)

    if (is.null(mb)) {
      # Fallback: empty viewer with message
      return(
        r3dmol(backgroundColor = "white") |>
          m_set_style(style = m_style_cartoon()) |>
          m_zoom_to()
      )
    }

    # Build colour spec
    colour_spec <- switch(colour,
      element  = m_style_stick(colorscheme = "Jmol"),
      chain    = m_style_stick(colorscheme = "chainHetatm"),
      residue  = m_style_stick(colorscheme = "amino")
    )

    # Build style spec
    style_spec <- switch(style_in,
      stick   = m_style_stick(),
      sphere  = m_style_sphere(scale = 0.4),
      line    = m_style_line(),
      cartoon = m_style_cartoon()
    )

    r3dmol(
      backgroundColor = "white",
      elementId       = "mol_viewer"
    ) |>
      m_add_model(data = mb, format = "mol") |>
      m_set_style(style = style_spec) |>
      m_add_surface(
        type    = "SAS",
        opacity = 0.08,
        color   = "#0072B2"
      ) |>
      m_zoom_to() |>
      m_spin(axis = "y", speed = 0.3)
  })

  # ── Performance table (About tab) ─────────────────────────────────────────────
  output$perf_table_ui <- renderUI({
    perf_path <- file.path("..", "models", "saved", "results_summary.json")
    if (!file.exists(perf_path)) {
      return(tags$p(class = "text-muted fst-italic",
                    "Results available after model training completes."))
    }
    res  <- fromJSON(perf_path)
    rows <- lapply(names(res), function(key) {
      r <- res[[key]]
      tags$tr(
        tags$td(key),
        tags$td(sprintf("%.3f", r$gmfe)),
        tags$td(sprintf("%.3f", r$r2)),
        tags$td(sprintf("%.1f%%", r$within_2fold))
      )
    })
    tags$table(
      class = "table table-sm table-bordered",
      tags$thead(
        tags$tr(
          tags$th("Model / Param"),
          tags$th("GMFE"),
          tags$th("R²"),
          tags$th("Within 2-fold")
        )
      ),
      tags$tbody(!!!rows)
    )
  })

  # ── Downloads ─────────────────────────────────────────────────────────────────
  output$dl_csv <- downloadHandler(
    filename = function() sprintf("pk_predictions_%s.csv", Sys.Date()),
    content  = function(file) {
      req(rv$results)
      write.csv(rv$results, file, row.names = FALSE)
    }
  )

  output$dl_json <- downloadHandler(
    filename = function() sprintf("pk_predictions_%s.json", Sys.Date()),
    content  = function(file) {
      req(rv$results)
      write(toJSON(rv$results, pretty = TRUE, auto_unbox = TRUE), file)
    }
  )
}
