# ADR-006: Deployment Architecture

**Date:** 2026-06-07  
**Status:** Decided

## Context
The tool must be accessible to academic users via a web interface. 
The R Shiny front-end is preferred for the user-facing layer. 
shinyapps.io does not support Python runtimes.

## Decision
- **ML backend:** FastAPI REST API, deployed on Render.com (free tier)
- **Front-end:** R Shiny app deployed on shinyapps.io
- **Integration:** Shiny POSTs a SMILES string to the FastAPI endpoint; 
  receives JSON response with point estimates and 95% CIs for all parameters

## Alternatives Considered
- **reticulate (Python in R directly):** Rejected for shinyapps.io — Python 
  runtime not available on the platform.
- **Posit Connect:** More capable but requires institutional license.
- **Single Python web app (Streamlit/Dash):** Rejected — client preference is R Shiny.

## Rationale
Decoupling the ML backend (FastAPI) from the UI (Shiny) is the correct architecture 
for this constraint. Render.com free tier is sufficient for inference-only workloads 
at academic usage levels. The API can also be called programmatically by researchers.

## Consequences
- Two separate deployments to maintain (API + Shiny)
- API must handle SMILES validation and return informative errors
- Cold start latency on Render.com free tier (~30s) should be communicated to users
- API should be versioned (e.g., /v1/predict) for future model updates
