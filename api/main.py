"""
api/main.py
===========
Wadhams PK Predictor — FastAPI backend.

Endpoints:
  POST /predict   — predict CL + Vd (+ optional 95% PIs) from SMILES
  GET  /health    — liveness check
  GET  /models    — model performance summary

The hybrid predictor is loaded once at startup and kept in memory.

Run locally:
    cd /path/to/pk-predictor
    uvicorn api.main:app --reload --port 8000

Environment variables:
  MODELS_DIR     — override default models/saved/ path
  DATA_DIR       — override default data/processed/ path
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid.predictor import HybridPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wadhams")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Wadhams PK Predictor",
    description = "Human pharmacokinetic parameter prediction from SMILES.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten after deployment if needed
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Global predictor (loaded once at startup) ─────────────────────────────────
predictor: Optional[HybridPredictor] = None
startup_time: Optional[float] = None


@app.on_event("startup")
def load_predictor():
    global predictor, startup_time
    t0 = time.time()
    log.info("Loading Wadhams hybrid predictor...")

    models_dir = Path(os.getenv("MODELS_DIR", str(ROOT / "models" / "saved")))
    data_dir   = Path(os.getenv("DATA_DIR",   str(ROOT / "data"   / "processed")))

    try:
        predictor    = HybridPredictor.load(models_dir=models_dir, data_dir=data_dir)
        startup_time = time.time() - t0
        log.info(f"Predictor ready in {startup_time:.1f}s")
    except Exception as e:
        log.error(f"Failed to load predictor: {e}")
        predictor = None


# ── Request / response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: List[str]
    model:  str  = "hybrid"   # hybrid | rf | xgb | gnn (only hybrid supported in v1)
    ci:     bool = True

    @field_validator("smiles")
    @classmethod
    def smiles_not_empty(cls, v):
        if not v:
            raise ValueError("smiles list must not be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 compounds per request")
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("All SMILES strings are empty")
        return cleaned


class CompoundResult(BaseModel):
    smiles:       str
    model_used:   str
    CL_pred:      Optional[float] = None
    CL_lower:     Optional[float] = None
    CL_upper:     Optional[float] = None
    CL_log_pred:  Optional[float] = None
    Vd_pred:      Optional[float] = None
    Vd_lower:     Optional[float] = None
    Vd_upper:     Optional[float] = None
    Vd_log_pred:  Optional[float] = None
    thalf_pred:   Optional[float] = None
    lambdaz_pred: Optional[float] = None
    status:       str
    error:        Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness + readiness check."""
    return {
        "status":       "ok" if predictor is not None else "degraded",
        "predictor":    "loaded" if predictor is not None else "not loaded",
        "startup_time": startup_time,
        "version":      "1.0.0",
    }


@app.get("/models")
def model_info():
    """Return model routing and performance summary."""
    import json
    results = {}
    for name, path in [
        ("rf",  ROOT / "models/saved/rf/rf_results.json"),
        ("xgb", ROOT / "models/saved/xgb/xgb_results.json"),
        ("gnn", ROOT / "models/saved/gnn/gnn_results.json"),
    ]:
        if path.exists():
            results[name] = json.loads(path.read_text())

    conformal = {}
    conf_summary = ROOT / "models/saved/conformal/conformal_summary.json"
    if conf_summary.exists():
        conformal = json.loads(conf_summary.read_text())

    return {
        "hybrid_routing": {"CL": "rf", "Vd": "xgb"},
        "model_results":  results,
        "conformal":      conformal,
    }


@app.post("/predict", response_model=List[CompoundResult])
def predict(req: PredictRequest):
    """
    Predict CL and Vd (+ optional 95% PIs) for a list of SMILES.

    - Maximum 100 compounds per request
    - Returns one result object per input SMILES
    - Compounds that fail featurization return status='error'
    """
    if predictor is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Predictor not loaded. Check server logs."
        )

    t0 = time.time()
    log.info(f"Predict request: {len(req.smiles)} compound(s), ci={req.ci}")

    results = predictor.predict(req.smiles, ci=req.ci)

    elapsed = time.time() - t0
    n_ok    = sum(1 for r in results if r.get("status") == "ok")
    log.info(f"  Done in {elapsed:.2f}s — {n_ok}/{len(results)} succeeded")

    return results
