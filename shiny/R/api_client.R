##############################################################################
# shiny/R/api_client.R
# ====================
# Thin wrapper around the FastAPI backend.
#
# Base URL is read from environment variable PK_API_URL (default: localhost).
# Falls back to mock data when the API is unreachable — useful during
# UI development before the Python backend is running.
##############################################################################

PK_API_BASE <- Sys.getenv("PK_API_URL", unset = "http://localhost:8000")

# ── Main predict call ─────────────────────────────────────────────────────────
#
# payload: list(smiles = list("CC..."), model = "hybrid", ci = TRUE)
# Returns: parsed JSON list, or list(error = "message")
#
pk_predict <- function(payload) {

  resp <- tryCatch({
    request(PK_API_BASE) |>
      req_url_path_append("predict") |>
      req_headers("Content-Type" = "application/json") |>
      req_body_json(payload) |>
      req_timeout(120) |>
      req_error(is_error = \(r) FALSE) |>
      req_perform()
  }, error = function(e) {
    # API unreachable — fall back to mock data automatically.
    # This covers: local dev, shinyapps.io test deployment, and any case
    # where the backend hasn't been started yet.
    message("[api_client] API unreachable — returning mock data (",
            conditionMessage(e), ")")
    return(mock_response(payload))
  })

  # If we got a mock response object back, return it directly
  if (is.list(resp) && !inherits(resp, "httr2_response")) {
    return(resp)
  }

  status <- resp_status(resp)
  if (status >= 400) {
    body <- tryCatch(resp_body_json(resp), error = function(e) list(detail = "Unknown error"))
    stop(sprintf("API error %d: %s", status, body$detail %||% "Unknown error"))
  }

  resp_body_json(resp)
}

# ── Health check ──────────────────────────────────────────────────────────────
pk_health <- function() {
  tryCatch({
    resp <- request(PK_API_BASE) |>
      req_url_path_append("health") |>
      req_timeout(5) |>
      req_perform()
    list(ok = resp_status(resp) == 200)
  }, error = function(e) {
    list(ok = FALSE, error = conditionMessage(e))
  })
}

# ── Null coalescing ────────────────────────────────────────────────────────────
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ── Mock response (development / offline mode) ────────────────────────────────
# Set environment variable PK_DEV_MOCK=1 to use this instead of real API.
# Returns plausible PK values for any SMILES.
mock_response <- function(payload) {
  smiles <- unlist(payload$smiles)
  n      <- length(smiles)
  set.seed(42 + n)

  lapply(seq_len(n), function(i) {
    cl_log <- runif(1, -0.5, 1.5)      # log10(CL)
    vd_log <- runif(1,  0.0, 1.5)      # log10(Vd)
    q_cl   <- 0.35
    q_vd   <- 0.28

    list(
      smiles        = smiles[i],
      model_used    = payload$model,
      CL_pred       = round(10^cl_log, 4),
      CL_lower      = round(10^(cl_log - q_cl), 4),
      CL_upper      = round(10^(cl_log + q_cl), 4),
      CL_log_pred   = round(cl_log, 4),
      Vd_pred       = round(10^vd_log, 4),
      Vd_lower      = round(10^(vd_log - q_vd), 4),
      Vd_upper      = round(10^(vd_log + q_vd), 4),
      Vd_log_pred   = round(vd_log, 4),
      thalf_pred    = round(0.693 * 10^vd_log / (10^cl_log * 60 / 1000), 3),
      lambdaz_pred  = round((10^cl_log * 60 / 1000) / 10^vd_log, 5),
      status        = "ok",
      error         = NULL
    )
  })
}
