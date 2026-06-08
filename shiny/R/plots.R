##############################################################################
# shiny/R/plots.R
# ===============
# Interactive Plotly visualisations for the Shiny UI.
#
# make_interval_plot() — horizontal CI strip plot for one PK parameter
##############################################################################

library(plotly)

# Wong 2011 colour palette
CLR_MAP <- c(
  rf     = "#0072B2",
  xgb    = "#E69F00",
  gnn    = "#009E73",
  hybrid = "#CC79A7"
)

# ── Interval strip plot ───────────────────────────────────────────────────────
#
# df    : data.frame from parse_predictions()
# param : one of "CL" | "Vd" | "thalf" | "lambdaz"
# scale : "original" | "log10"
#
make_interval_plot <- function(df, param = "CL", scale = "original") {

  meta <- list(
    CL      = list(pred = "CL_pred",      lo = "CL_lower",    hi = "CL_upper",
                   xlab = "CL (mL/min/kg)"),
    Vd      = list(pred = "Vd_pred",      lo = "Vd_lower",    hi = "Vd_upper",
                   xlab = "Vd (L/kg)"),
    thalf   = list(pred = "thalf_pred",   lo = NULL,          hi = NULL,
                   xlab = "t½ (h)"),
    lambdaz = list(pred = "lambdaz_pred", lo = NULL,          hi = NULL,
                   xlab = "λz (1/h)")
  )

  m   <- meta[[param]]
  ok  <- df$status == "ok"
  df  <- df[ok, , drop = FALSE]

  if (nrow(df) == 0) {
    return(plotly_empty(type = "scatter") |>
             layout(title = "No valid predictions to display"))
  }

  # Extract values
  y_pred <- df[[m$pred]]
  y_lo   <- if (!is.null(m$lo)) df[[m$lo]] else rep(NA_real_, nrow(df))
  y_hi   <- if (!is.null(m$hi)) df[[m$hi]] else rep(NA_real_, nrow(df))

  # Apply scale
  if (scale == "log10") {
    y_pred <- log10(pmax(y_pred, 1e-9))
    y_lo   <- log10(pmax(y_lo,   1e-9))
    y_hi   <- log10(pmax(y_hi,   1e-9))
    xlab   <- paste0("log₁₀(", m$xlab, ")")
  } else {
    xlab <- m$xlab
  }

  # Colour by model
  model_col <- unname(CLR_MAP[match(df$model_used, names(CLR_MAP))])
  model_col[is.na(model_col)] <- "#888888"

  # Compound labels (truncate if too long)
  labels <- ifelse(nchar(df$name) > 30,
                   paste0(substr(df$name, 1, 28), "…"),
                   df$name)

  n     <- nrow(df)
  y_pos <- seq(n, 1)   # top-to-bottom ordering

  fig <- plot_ly()

  # Error bars (CI) — only if available
  has_ci <- !all(is.na(y_lo))
  if (has_ci) {
    for (i in seq_len(n)) {
      if (!is.na(y_lo[i]) && !is.na(y_hi[i])) {
        fig <- fig |>
          add_segments(
            x    = y_lo[i],
            xend = y_hi[i],
            y    = y_pos[i],
            yend = y_pos[i],
            line = list(color = model_col[i], width = 2),
            showlegend = FALSE,
            hoverinfo  = "skip"
          )
      }
    }
  }

  # Point estimates
  hover_text <- sprintf(
    "<b>%s</b><br>%s = %.3f%s<br>Model: %s",
    labels,
    m$xlab,
    y_pred,
    if (scale == "log10") " (log₁₀)" else "",
    df$model_used
  )
  if (has_ci) {
    hover_text <- paste0(
      hover_text,
      sprintf("<br>95%% PI: [%.3f, %.3f]", y_lo, y_hi)
    )
  }

  fig <- fig |>
    add_trace(
      type   = "scatter",
      mode   = "markers",
      x      = y_pred,
      y      = y_pos,
      marker = list(
        color  = model_col,
        size   = 9,
        symbol = "circle",
        line   = list(color = "white", width = 1.5)
      ),
      text      = hover_text,
      hoverinfo = "text"
    )

  # Add vertical reference line at median
  med_x <- median(y_pred, na.rm = TRUE)

  fig |>
    layout(
      xaxis = list(
        title      = xlab,
        zeroline   = FALSE,
        showgrid   = TRUE,
        gridcolor  = "#EEEEEE"
      ),
      yaxis = list(
        tickvals   = y_pos,
        ticktext   = labels,
        showgrid   = FALSE,
        zeroline   = FALSE,
        autorange  = "reversed"
      ),
      shapes = list(
        list(
          type    = "line",
          x0 = med_x, x1 = med_x,
          y0 = 0,     y1 = 1,
          yref = "paper",
          line = list(color = "#999999", width = 1, dash = "dot")
        )
      ),
      annotations = list(
        list(
          x = med_x, y = 1.01,
          xref = "x", yref = "paper",
          text = sprintf("median = %.3f", if (scale == "log10") 10^med_x else med_x),
          showarrow = FALSE,
          font = list(size = 10, color = "#666666"),
          xanchor = "center"
        )
      ),
      margin    = list(l = 180, r = 20, t = 20, b = 50),
      showlegend = FALSE,
      plot_bgcolor  = "white",
      paper_bgcolor = "white"
    ) |>
    config(
      displayModeBar = TRUE,
      modeBarButtonsToRemove = c("select2d", "lasso2d", "toggleSpikelines"),
      toImageButtonOptions = list(
        format = "png",
        filename = paste0("pk_predictor_", param),
        width  = 900,
        height = max(300, n * 30),
        scale  = 2
      )
    )
}
