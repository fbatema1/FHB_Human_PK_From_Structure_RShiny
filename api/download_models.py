"""
api/download_models.py
======================
Downloads model files from Hugging Face Hub at startup.
Run before starting the FastAPI server on Render.

Required environment variable:
  HF_REPO   — e.g. "fbatema1/wadhams-pk-models"
  HF_TOKEN  — Hugging Face read token (set in Render environment)

Files downloaded to models/saved/ and data/processed/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def download():
    from huggingface_hub import hf_hub_download

    repo   = os.environ["HF_REPO"]
    token  = os.environ.get("HF_TOKEN")

    files = [
        # (repo filename,                         local destination)
        ("rf_CL_best.pkl",    ROOT / "models/saved/rf/rf_CL_best.pkl"),
        ("rf_Vd_best.pkl",    ROOT / "models/saved/rf/rf_Vd_best.pkl"),
        ("xgb_CL_best.pkl",   ROOT / "models/saved/xgb/xgb_CL_best.pkl"),
        ("xgb_Vd_best.pkl",   ROOT / "models/saved/xgb/xgb_Vd_best.pkl"),
        ("featurizer_CL.pkl", ROOT / "data/processed/featurizer_CL.pkl"),
        ("featurizer_Vd.pkl", ROOT / "data/processed/featurizer_Vd.pkl"),
    ]

    for repo_file, local_path in files:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"  ✓ Already exists: {local_path.name}")
            continue
        print(f"  ↓ Downloading {repo_file}...")
        hf_hub_download(
            repo_id   = repo,
            filename  = repo_file,
            token     = token,
            local_dir = str(local_path.parent),
        )
        print(f"  ✓ {repo_file} → {local_path}")

    print("All model files ready.")


if __name__ == "__main__":
    download()
