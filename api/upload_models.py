"""
api/upload_models.py
====================
Uploads trained model files to the private HuggingFace Hub repo so that
Railway can download them at startup.

Run this from Longleaf AFTER the full pipeline completes:

    HF_TOKEN=hf_xxx HF_REPO=francishenrybateman/wadhams-pk-models \\
        python api/upload_models.py

Or set env vars in your shell first:
    export HF_TOKEN=hf_xxx
    export HF_REPO=francishenrybateman/wadhams-pk-models
    python api/upload_models.py

Files uploaded (matching what download_models.py expects):
    rf_CL_best.pkl
    rf_Vd_best.pkl
    xgb_CL_best.pkl
    xgb_Vd_best.pkl
    featurizer_CL.pkl
    featurizer_Vd.pkl
    conformal_rf_CL.pkl
    conformal_xgb_Vd.pkl
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def upload():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    token = os.environ.get("HF_TOKEN")
    repo  = os.environ.get("HF_REPO", "francishenrybateman/wadhams-pk-models")

    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Generate a write token at "
            "https://huggingface.co/settings/tokens and set:\n"
            "  export HF_TOKEN=hf_xxx"
        )

    api = HfApi()

    files = [
        # (local path,                                       repo filename)
        (ROOT / "models/saved/rf/rf_CL_best.pkl",           "rf_CL_best.pkl"),
        (ROOT / "models/saved/rf/rf_Vd_best.pkl",           "rf_Vd_best.pkl"),
        (ROOT / "models/saved/xgb/xgb_CL_best.pkl",        "xgb_CL_best.pkl"),
        (ROOT / "models/saved/xgb/xgb_Vd_best.pkl",        "xgb_Vd_best.pkl"),
        (ROOT / "data/processed/featurizer_CL.pkl",         "featurizer_CL.pkl"),
        (ROOT / "data/processed/featurizer_Vd.pkl",         "featurizer_Vd.pkl"),
        (ROOT / "models/saved/conformal/conformal_rf_CL.pkl",  "conformal_rf_CL.pkl"),
        (ROOT / "models/saved/conformal/conformal_xgb_Vd.pkl", "conformal_xgb_Vd.pkl"),
    ]

    print(f"Uploading to: {repo}\n")
    for local_path, repo_filename in files:
        if not local_path.exists():
            print(f"  ✗ MISSING: {local_path} — skipping")
            continue
        size_mb = local_path.stat().st_size / 1e6
        print(f"  ↑ {repo_filename} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=repo,
            repo_type="model",
            token=token,
        )
        print(f"    ✓ uploaded")

    print("\nAll files uploaded.")
    print("Railway will pick up the new versions on next deploy/restart.")
    print("\n⚠  Remember to revoke your write token and use a read-only")
    print("   token for Railway: https://huggingface.co/settings/tokens")


if __name__ == "__main__":
    upload()
