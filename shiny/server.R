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
library(dplyr)
library(tibble)

source("R/api_client.R")
source("R/plots.R")
source("R/utils.R")

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

  # ── Load example ─────────────────────────────────────────────────────────────
  observeEvent(input$load_example, {
    updateTextAreaInput(session, "smiles_input", value = EXAMPLE_SMILES)
    updateTextInput(session, "compound_name",   value = EXAMPLE_NAME)
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

  # ── Structure grid ────────────────────────────────────────────────────────────
  output$structure_warning_ui <- renderUI({
    if (is.null(rv$results)) {
      return(
        div(class = "alert alert-info",
            bsicons::bs_icon("info-circle"), " Run a prediction first.")
      )
    }
    div(class = "alert alert-secondary",
        bsicons::bs_icon("image"),
        " 2D structure depictions require the ",
        tags$code("ChemmineR"), " package or an internet connection for PubChem.",
        " SMILES strings are shown as fallback.")
  })

  output$structure_grid_ui <- renderUI({
    req(rv$results)
    page  <- input$struct_page
    total <- nrow(rv$results)
    n_pages <- ceiling(total / STRUCTS_PER_PAGE)

    # Update page input
    updateNumericInput(session, "struct_page", max = n_pages,
                       label = sprintf("Page (1–%d)", n_pages))

    start_i <- (page - 1) * STRUCTS_PER_PAGE + 1
    end_i   <- min(page * STRUCTS_PER_PAGE, total)
    rows    <- rv$results[start_i:end_i, ]

    cards <- lapply(seq_len(nrow(rows)), function(i) {
      r <- rows[i, ]
      card(
        full_screen = FALSE,
        style       = "font-size:0.8rem;",
        card_header(
          style = "padding:0.4rem 0.6rem;",
          tags$strong(r$name)
        ),
        card_body(
          padding = "0.5rem",
          # Inline SMILES display — swap for actual structure image when available
          tags$div(
            class = "font-monospace text-break text-muted",
            style = "font-size:0.68rem; word-break:break-all;",
            r$smiles
          ),
          tags$hr(style = "margin:0.4rem 0;"),
          tags$table(
            class = "table table-sm mb-0",
            tags$tbody(
              tags$tr(
                tags$td("CL"),
                tags$td(class="text-end", sprintf("%.3f", r$CL_pred))
              ),
              tags$tr(
                tags$td("Vd"),
                tags$td(class="text-end", sprintf("%.3f", r$Vd_pred))
              ),
              tags$tr(
                tags$td("t½"),
                tags$td(class="text-end", sprintf("%.2f", r$thalf_pred))
              )
            )
          )
        )
      )
    })

    layout_column_wrap(width = 1/3, !!!cards)
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
