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

# conformer.R requires reticulate + r3dmol — only load when available
if (exists("HAS_RETICULATE") && HAS_RETICULATE && exists("HAS_R3DMOL") && HAS_R3DMOL) {
  source("R/conformer.R")
} else {
  # Stub so the rest of server.R doesn't error
  rdkit_available <- function() FALSE
  get_molblock    <- function(smiles) NULL
}

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
    results      = NULL,
    api_error    = NULL,
    busy         = FALSE,
    n_compare    = 2       # number of rows in compare mode
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

  # ── Compare mode: dynamic rows ────────────────────────────────────────────────
  observeEvent(input$add_compound_btn, {
    rv$n_compare <- min(rv$n_compare + 1, 8)
  })
  observeEvent(input$remove_compound_btn, {
    rv$n_compare <- max(rv$n_compare - 1, 1)
  })

  output$compare_rows_ui <- renderUI({
    n <- rv$n_compare
    rows <- lapply(seq_len(n), function(i) {
      div(
        class = "mb-2 p-2 border rounded",
        style = "background:#fff;",
        fluidRow(
          column(4,
            selectizeInput(
              paste0("cmp_name_", i),
              label   = NULL,
              choices = NULL,          # populated server-side
              options = list(
                placeholder  = paste0("Name ", i, "…"),
                create       = TRUE,   # allow novel names not in dataset
                maxOptions   = 20,
                openOnFocus  = FALSE
              ),
              width = "100%"
            )
          ),
          column(8,
            textInput(paste0("cmp_smiles_", i), label = NULL,
                      placeholder = "SMILES (auto-filled or enter manually)",
                      width = "100%")
          )
        )
      )
    })
    tagList(
      tags$div(
        class = "d-flex mb-1",
        style = "font-size:0.78rem; color:#6c757d;",
        tags$span(style = "width:33%;", "Name"),
        tags$span("SMILES")
      ),
      rows
    )
  })

  # Populate each name selectize server-side and watch for autofill
  observe({
    n <- rv$n_compare
    for (i in seq_len(n)) {
      local({
        idx <- i
        # Populate dropdown with training reference names
        updateSelectizeInput(
          session,
          paste0("cmp_name_", idx),
          choices  = REFERENCE_CHOICES,   # named vector: name -> smiles
          server   = TRUE
        )
      })
    }
  })

  # Autofill SMILES when a known name is selected in any compare row
  observe({
    n <- rv$n_compare
    for (i in seq_len(n)) {
      local({
        idx       <- i
        input_id  <- paste0("cmp_name_",   idx)
        smiles_id <- paste0("cmp_smiles_", idx)
        selected  <- input[[input_id]]
        req(!is.null(selected), nchar(selected) > 0)

        # REFERENCE_CHOICES is named: names = compound names, values = SMILES
        # selectize returns the VALUE (smiles) when a known entry is picked
        is_known_smiles <- selected %in% REFERENCE_CHOICES
        if (is_known_smiles) {
          # Selected value is already the SMILES string
          updateTextInput(session, smiles_id, value = selected)
        }
        # If user typed a novel name (create=TRUE), leave SMILES blank
      })
    }
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

    } else if (input$input_mode == "compare") {
      n <- rv$n_compare
      smiles_vec <- character(0)
      names_vec  <- character(0)
      for (i in seq_len(n)) {
        # SMILES field takes priority (manual entry or autofilled)
        s  <- trimws(input[[paste0("cmp_smiles_", i)]])
        nm <- trimws(input[[paste0("cmp_name_",   i)]])

        # If SMILES is empty but name matched a known compound, use name value
        if (nchar(s) == 0 && nm %in% REFERENCE_CHOICES) s <- nm

        # Resolve display name: if nm is a SMILES string, look up the real name
        display_name <- if (nm %in% REFERENCE_CHOICES) {
          names(REFERENCE_CHOICES)[REFERENCE_CHOICES == nm][1]
        } else if (nchar(nm) > 0) {
          nm
        } else {
          paste0("Compound ", i)
        }

        if (nchar(s) > 0) {
          smiles_vec <- c(smiles_vec, s)
          names_vec  <- c(names_vec, display_name)
        }
      }
      if (length(smiles_vec) == 0) {
        showNotification("Enter at least one SMILES string.", type = "warning")
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
    make_interval_plot(rv$results, input$plot_param, input$plot_scale)
  })

  # ── Derived parameters table (t½ and λz) ──────────────────────────────────────
  output$derived_table <- renderDT({
    req(rv$results)
    ok <- rv$results[rv$results$status == "ok", ]
    req(nrow(ok) > 0)

    df <- data.frame(
      Name          = ok$name,
      "t½ (h)"      = ifelse(is.na(ok$thalf_pred),   "—", sprintf("%.2f", ok$thalf_pred)),
      "λz (1/h)"    = ifelse(is.na(ok$lambdaz_pred), "—", sprintf("%.4f", ok$lambdaz_pred)),
      "CL (mL/min/kg)" = ifelse(is.na(ok$CL_pred), "—", sprintf("%.3f", ok$CL_pred)),
      "Vd (L/kg)"   = ifelse(is.na(ok$Vd_pred),   "—", sprintf("%.3f", ok$Vd_pred)),
      check.names   = FALSE,
      stringsAsFactors = FALSE
    )

    datatable(df, rownames = FALSE, escape = FALSE,
              options = list(dom = "t", scrollX = TRUE, pageLength = 20)) |>
      formatStyle("t½ (h)",   color = "#0072B2", fontWeight = "bold") |>
      formatStyle("λz (1/h)", color = "#009E73", fontWeight = "bold")
  })

  # ── 3D Structure viewer ────────────────────────────────────────────────────────

  # Compound selector — uses prediction results if available, else training repo
  output$struct_compound_select_ui <- renderUI({
    if (!is.null(rv$results)) {
      ok_rows <- rv$results[rv$results$status == "ok", ]
      if (nrow(ok_rows) > 0) {
        choices <- setNames(ok_rows$smiles, ok_rows$name)
        return(selectInput("struct_selected", label = NULL,
                           choices = choices, width = "100%"))
      }
    }
    # Fallback: browse training reference directly
    if (!is.null(TRAINING_REF) && nrow(TRAINING_REF) > 0) {
      choices <- setNames(TRAINING_REF$smiles, TRAINING_REF$name)
      return(tagList(
        selectizeInput("struct_selected", label = NULL,
                       choices  = NULL,
                       options  = list(placeholder = "Type a compound name…",
                                       maxOptions  = 20),
                       width    = "100%"),
        tags$small(class = "text-muted",
                   "No prediction run yet — browsing training library")
      ))
    }
    tags$p(class = "text-muted fst-italic", "Run a prediction to view structures.")
  })

  # Populate training-repo selectize server-side when no predictions exist
  observe({
    if (is.null(rv$results) && !is.null(TRAINING_REF)) {
      updateSelectizeInput(session, "struct_selected",
                           choices = setNames(TRAINING_REF$smiles, TRAINING_REF$name),
                           server  = TRUE)
    }
  })

  # Metadata card
  output$struct_metadata_ui <- renderUI({
    req(input$struct_selected, nchar(input$struct_selected) > 0)

    # Predicted values (if available)
    pred_rows <- if (!is.null(rv$results)) {
      rv$results[rv$results$smiles == input$struct_selected, ]
    } else NULL

    # Observed values from training reference
    ref_row <- if (!is.null(TRAINING_REF)) {
      TRAINING_REF[TRAINING_REF$smiles == input$struct_selected, ]
    } else NULL

    has_pred <- !is.null(pred_rows) && nrow(pred_rows) > 0
    has_obs  <- !is.null(ref_row)   && nrow(ref_row)   > 0

    if (!has_pred && !has_obs) return(NULL)

    card(
      class = "mt-2",
      card_body(
        padding = "0.6rem",
        tags$table(
          class = "table table-sm table-borderless mb-0",
          style = "font-size:0.82rem;",
          tags$tbody(
            if (has_pred) tagList(
              tags$tr(tags$th("CL pred"),
                      tags$td(sprintf("%.3f mL/min/kg", pred_rows$CL_pred[1]))),
              tags$tr(tags$th("Vd pred"),
                      tags$td(sprintf("%.3f L/kg",      pred_rows$Vd_pred[1]))),
              tags$tr(tags$th("t½ pred"),
                      tags$td(sprintf("%.2f h",         pred_rows$thalf_pred[1])))
            ),
            if (has_obs) tagList(
              tags$tr(tags$th(class = "text-success", "CL obs"),
                      tags$td(class = "text-success",
                              sprintf("%.3f mL/min/kg", ref_row$CL_measured[1]))),
              tags$tr(tags$th(class = "text-success", "Vd obs"),
                      tags$td(class = "text-success",
                              sprintf("%.3f L/kg",     ref_row$Vd_measured[1])))
            )
          )
        ),
        if (has_obs)
          tags$small(class = "text-success",
                     bsicons::bs_icon("database-check"),
                     " Observed values from training set")
      )
    )
  })

  # 3D viewer — send SMILES to 3Dmol.js via custom message
  # Works on shinyapps.io, mobile, and all browsers (no Python required)
  # Helper: current mono colour (hex from colour picker, default blue)
  mono_colour <- reactive({
    col <- input$mono_colour %||% "#0072B2"
    if (is.null(col) || nchar(col) == 0) "#0072B2" else col
  })

  observe({
    req(input$struct_selected, nchar(input$struct_selected) > 0)
    session$sendCustomMessage("loadMolecule", list(
      smiles     = input$struct_selected,
      name       = input$struct_selected,
      style      = input$viewer_style  %||% "stick",
      colour     = input$viewer_colour %||% "element",
      monoColour = mono_colour()
    ))
  })

  # Restyle without re-fetching when display options change
  observeEvent(list(input$viewer_style, input$viewer_colour, mono_colour()), {
    req(input$struct_selected, nchar(input$struct_selected) > 0)
    session$sendCustomMessage("restyleMolecule", list(
      style      = input$viewer_style  %||% "stick",
      colour     = input$viewer_colour %||% "element",
      monoColour = mono_colour()
    ))
  }, ignoreInit = TRUE)

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
