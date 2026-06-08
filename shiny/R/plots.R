##############################################################################
# shiny/R/plots.R
# ===============
# Interactive Plotly visualisations for the Shiny UI.
#
# make_interval_plot() — forest-style CI plot with shaded uncertainty bands
##############################################################################

library(plotly)

# Wong 2011 colour palette (model badges / legend)
CLR_MAP <- c(
  rf     = "#0072B2",
  xgb    = "#E69F00",
  gnn    = "#009E73",
  hybrid = "#CC79A7"
)

# Shaded band colours (semi-transparent fills) per model
FILL_MAP <- c(
  rf     = "rgba(0,114,178,0.15)",
  xgb    = "rgba(230,159,0,0.15)",
  gnn    = "rgba(0,158,115,0.15)",
  hybrid = "rgba(204,121,167,0.15)"
)

# Per-compound colour cycle (up to 8 compounds, colour-blind safe)
# Slot 1 = blue (single-compound default); slot 2+ alternate warm/cool
COMPOUND_COLORS <- c(
  "#0072B2",   # 1  blue
  "#D55E00",   # 2  vermilion
  "#009E73",   # 3  green
  "#E69F00",   # 4  orange
  "#CC79A7",   # 5  pink
  "#56B4E9",   # 6  sky blue
  "#F0E442",   # 7  yellow (dark border)
  "#000000"    # 8  black
)

COMPOUND_FILLS <- c(
  "rgba(0,114,178,0.15)",
  "rgba(213,94,0,0.15)",
  "rgba(0,158,115,0.15)",
  "rgba(230,159,0,0.15)",
  "rgba(204,121,167,0.15)",
  "rgba(86,180,233,0.15)",
  "rgba(240,228,66,0.15)",
  "rgba(0,0,0,0.10)"
)

# ── Forest-style CI plot ──────────────────────────────────────────────────────
#
# Each compound gets:
#   - A filled rectangle (shaded band) spanning the 95% PI
#   - A centre line through the prediction
#   - A filled diamond/circle at the point estimate
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
             layout(title = list(text = "No valid predictions to display",
                                 font = list(color = "#868E96"))))
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
    xlab   <- paste0("log₁₀ ", m$xlab)
  } else {
    xlab <- m$xlab
  }

  has_ci <- !all(is.na(y_lo))
  n      <- nrow(df)

  # Compound labels (truncate long names)
  labels <- ifelse(nchar(df$name) > 32,
                   paste0(substr(df$name, 1, 30), "…"),
                   df$name)

  # Y positions: top-to-bottom
  y_pos <- seq(n, 1)

  # Per-compound colours — cycle through COMPOUND_COLORS regardless of model
  # (model is shown via badge in UI; colour here distinguishes compounds)
  idx_cycle  <- ((seq_len(n) - 1) %% length(COMPOUND_COLORS)) + 1
  model_col  <- COMPOUND_COLORS[idx_cycle]
  model_fill <- COMPOUND_FILLS[idx_cycle]

  fig <- plot_ly()

  # ── 1. Shaded CI rectangles ───────────────────────────────────────────────
  if (has_ci) {
    for (i in seq_len(n)) {
      if (!is.na(y_lo[i]) && !is.na(y_hi[i])) {
        # Rectangle height = 0.45 on each side of centre
        fig <- fig |>
          add_trace(
            type   = "scatter",
            mode   = "none",
            x      = c(y_lo[i], y_hi[i], y_hi[i], y_lo[i], y_lo[i]),
            y      = c(y_pos[i] - 0.38, y_pos[i] - 0.38,
                       y_pos[i] + 0.38, y_pos[i] + 0.38,
                       y_pos[i] - 0.38),
            fill   = "toself",
            fillcolor = model_fill[i],
            line      = list(color = "transparent"),
            showlegend = FALSE,
            hoverinfo  = "skip"
          )
      }
    }
  }

  # ── 2. Horizontal CI lines ────────────────────────────────────────────────
  if (has_ci) {
    for (i in seq_len(n)) {
      if (!is.na(y_lo[i]) && !is.na(y_hi[i])) {
        fig <- fig |>
          add_segments(
            x    = y_lo[i],  xend = y_hi[i],
            y    = y_pos[i], yend = y_pos[i],
            line = list(color = model_col[i], width = 1.8),
            showlegend = FALSE,
            hoverinfo  = "skip"
          )
        # Whisker caps
        cap_h <- 0.18
        for (cap_x in c(y_lo[i], y_hi[i])) {
          fig <- fig |>
            add_segments(
              x    = cap_x, xend = cap_x,
              y    = y_pos[i] - cap_h, yend = y_pos[i] + cap_h,
              line = list(color = model_col[i], width = 1.8),
              showlegend = FALSE,
              hoverinfo  = "skip"
            )
        }
      }
    }
  }

  # ── 3. Point estimates (filled diamonds) ─────────────────────────────────
  hover_text <- sprintf(
    "<b>%s</b><br>%s = %.3f%s<br>Model: %s",
    labels,
    m$xlab,
    if (scale == "log10") 10^y_pred else y_pred,
    if (scale == "log10") sprintf(" (log₁₀ = %.3f)", y_pred) else "",
    df$model_used
  )
  if (has_ci) {
    orig_lo <- if (scale == "log10") 10^y_lo else y_lo
    orig_hi <- if (scale == "log10") 10^y_hi else y_hi
    hover_text <- paste0(
      hover_text,
      sprintf("<br>95%% PI: [%.3f, %.3f]", orig_lo, orig_hi),
      sprintf("<br>Fold range: %.2f×", orig_hi / pmax(orig_lo, 1e-9))
    )
  }

  fig <- fig |>
    add_trace(
      type   = "scatter",
      mode   = "markers",
      x      = y_pred,
      y      = y_pos,
      marker = list(
        color   = model_col,
        size    = 11,
        symbol  = "diamond",
        line    = list(color = "white", width = 1.5)
      ),
      text      = hover_text,
      hoverinfo = "text",
      showlegend = FALSE
    )

  # ── 4. Dotted vertical lines + value labels at CI bounds ─────────────────
  # One dotted line per compound at each bound, with the numeric value labelled
  # above the top compound and below the bottom compound (avoid per-row clutter).
  # For small n (≤ 8) we label every bound; for larger n, label only extremes.
  shapes      <- list()
  annotations <- list()

  if (has_ci) {
    label_all <- (n <= 8)

    for (i in seq_len(n)) {
      if (is.na(y_lo[i]) || is.na(y_hi[i])) next

      # Dotted vertical lines at lower and upper bounds
      for (bx in c(y_lo[i], y_hi[i])) {
        shapes <- c(shapes, list(list(
          type = "line",
          x0   = bx, x1 = bx,
          y0   = y_pos[i] - 0.42,
          y1   = y_pos[i] + 0.42,
          line = list(color = model_col[i], width = 1, dash = "dot")
        )))
      }

      # Numeric labels — show in the same units as the axis
      # Original scale → original values; log10 scale → log10 values
      if (label_all || y_pos[i] == n) {
        label_lo  <- if (scale == "log10") y_lo[i]   else y_lo[i]
        label_hi  <- if (scale == "log10") y_hi[i]   else y_hi[i]
        fmt_lo    <- if (scale == "log10") sprintf("<i>%.3f</i>", label_lo) else sprintf("<i>%.3g</i>", label_lo)
        fmt_hi    <- if (scale == "log10") sprintf("<i>%.3f</i>", label_hi) else sprintf("<i>%.3g</i>", label_hi)
        label_y   <- y_pos[i] + 0.52

        annotations <- c(annotations,
          list(list(
            x = y_lo[i], y = label_y,
            xref = "x", yref = "y",
            text = fmt_lo,
            showarrow = FALSE,
            font = list(size = 9, color = model_col[i]),
            xanchor = "center", yanchor = "bottom"
          )),
          list(list(
            x = y_hi[i], y = label_y,
            xref = "x", yref = "y",
            text = fmt_hi,
            showarrow = FALSE,
            font = list(size = 9, color = model_col[i]),
            xanchor = "center", yanchor = "bottom"
          ))
        )
      }
    }
  }

  # ── 5. Per-compound point-estimate reference lines ───────────────────────
  # One dotted vertical line per compound at its own predicted value,
  # spanning only that compound's row (no cross-plot median).
  for (i in seq_len(n)) {
    px <- y_pred[i]
    if (is.na(px)) next

    shapes <- c(shapes, list(list(
      type = "line",
      x0   = px, x1 = px,
      y0   = y_pos[i] - 0.42,
      y1   = y_pos[i] + 0.42,
      line = list(color = model_col[i], width = 1.5, dash = "dot")
    )))

    # Value label above the row  (original-scale when log10 active)
    orig_val <- if (scale == "log10") 10^px else px
    val_lab  <- if (scale == "log10") {
      sprintf("<i>%.3f</i>", px)          # show the log₁₀ value
    } else {
      sprintf("<i>%.3g</i>", orig_val)
    }

    annotations <- c(annotations, list(list(
      x         = px,
      y         = y_pos[i] + 0.55,
      xref      = "x",
      yref      = "y",
      text      = val_lab,
      showarrow = FALSE,
      font      = list(size = 9, color = model_col[i]),
      xanchor   = "center",
      yanchor   = "bottom"
    )))
  }

  # ── Layout ────────────────────────────────────────────────────────────────
  plot_h <- max(380, n * 44 + 100)

  fig |>
    layout(
      height = plot_h,
      xaxis  = list(
        title     = list(text = xlab, font = list(size = 13)),
        zeroline  = FALSE,
        showgrid  = TRUE,
        gridcolor = "#EEEEEE",
        gridwidth = 1
      ),
      yaxis = list(
        tickvals  = y_pos,
        ticktext  = labels,
        showgrid  = FALSE,
        zeroline  = FALSE,
        tickfont  = list(size = 11),
        range     = c(0.2, n + 0.8)
      ),
      shapes      = shapes,
      annotations = annotations,
      margin         = list(l = 200, r = 30, t = 50, b = 55),
      showlegend     = FALSE,
      plot_bgcolor   = "white",
      paper_bgcolor  = "white",
      hoverlabel     = list(
        bgcolor     = "white",
        bordercolor = "#CCCCCC",
        font        = list(size = 12)
      )
    ) |>
    config(
      displayModeBar = TRUE,
      modeBarButtonsToRemove = c("select2d", "lasso2d", "toggleSpikelines"),
      toImageButtonOptions = list(
        format   = "png",
        filename = paste0("wadhams_pk_", param),
        width    = 1000,
        height   = plot_h,
        scale    = 2
      )
    )
}
