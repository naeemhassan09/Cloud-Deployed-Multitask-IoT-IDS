from __future__ import annotations
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from typing import Any, Tuple

import io
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------------------
# Load scaler (feature schema + normalization)
# --------------------------------------------------------------
SCALER_FILENAME = "multitask_cnn_mlp_scaler.json"
scaler_path = PROCESSED_DIR / SCALER_FILENAME

with open(scaler_path, "r") as f:
    scaler = json.load(f)

# scaler = {"means": {feat_name: mean, ...}, "stds": {feat_name: std, ...}}
feature_names = list(scaler["means"].keys())
means = np.array([scaler["means"][k] for k in feature_names], dtype=np.float32)
stds = np.array([scaler["stds"][k] for k in feature_names], dtype=np.float32)
num_features = len(feature_names)

# --------------------------------------------------------------
# Load label maps
# --------------------------------------------------------------
with open(PROCESSED_DIR / "attack_label_mapping.json") as f:
    attack_label_map = json.load(f)["id_to_attack"]

with open(PROCESSED_DIR / "device_label_mapping.json") as f:
    device_label_map = json.load(f)["id_to_device"]

num_attacks = len(attack_label_map)
num_devices = len(device_label_map)

# --------------------------------------------------------------
# Model
# --------------------------------------------------------------
from .multitask_cnn_model import MultiTaskCNN1D

device = torch.device("cpu")

model = MultiTaskCNN1D(
    num_features=num_features,
    num_attacks=num_attacks,
    num_devices=num_devices,
)

model_path = MODELS_DIR / "multitask_cnn_mlp_stage3_joint_finetuned.pt"
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# --------------------------------------------------------------
# Eval metrics (optional)
# --------------------------------------------------------------
metrics_path = MODELS_DIR / "multitask_cnn_mlp_eval_metrics.json"
try:
    with open(metrics_path, "r") as f:
        eval_metrics = json.load(f)
except FileNotFoundError:
    eval_metrics = {}

# --------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------
app = FastAPI(title="Multitask IoT IDS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later (e.g., ["http://localhost:5173"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------
# SPA serving (React build) - MUST be after app is defined
# --------------------------------------------------------------
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def spa_index():
        return FileResponse(FRONTEND_DIST / "index.html")

    # React Router fallback (SPA) - do not hijack API/docs
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/") or full_path in ("docs", "openapi.json", "redoc", "health"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(FRONTEND_DIST / "index.html")

class PacketFeatures(BaseModel):
    features: list[float]


# --------------------------------------------------------------
# Helpers (raw CSV -> matrix)
# --------------------------------------------------------------
def _safe_read_csv(upload_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(upload_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")


def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> Tuple[bool, list[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def _build_feature_matrix_from_raw(
    df: pd.DataFrame,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    clip_value: float = 10.0,
) -> np.ndarray:
    """
    RAW CIC CSV -> numeric matrix in the exact order expected by the model.
    Steps:
      1) select model feature columns in correct order
      2) coerce to numeric (invalid -> NaN)
      3) replace inf/-inf -> NaN
      4) fill NaN with training mean per feature
      5) standardize
      6) clip
    """
    Xdf = df[feature_names].copy()

    for c in feature_names:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # fill NaNs with training means (fast + deterministic)
    for i, c in enumerate(feature_names):
        Xdf[c] = Xdf[c].fillna(float(means[i]))

    X = Xdf.astype(np.float32).values

    X = (X - means) / stds
    X = np.clip(X, -clip_value, clip_value)
    return X


def _build_distributions_and_rows(
    probs_att: np.ndarray,
    probs_dev: np.ndarray,
    pred_att_ids: np.ndarray,
    pred_dev_ids: np.ndarray,
    attack_label_map: dict,
    device_label_map: dict,
    max_predictions_return: int = 5000,
) -> Tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    """
    Build:
      - attack_distribution (counts over all N)
      - device_distribution (counts over all N)
      - predictions list (capped for UI responsiveness)
    """
    attack_distribution: dict[str, int] = {}
    device_distribution: dict[str, int] = {}
    predictions: list[dict[str, Any]] = []

    n = int(pred_att_ids.shape[0])
    limit = min(n, int(max_predictions_return))

    for idx in range(n):
        att_id = int(pred_att_ids[idx])
        dev_id = int(pred_dev_ids[idx])

        att_label = attack_label_map[str(att_id)]
        dev_label = device_label_map[str(dev_id)]

        attack_distribution[att_label] = attack_distribution.get(att_label, 0) + 1
        device_distribution[dev_label] = device_distribution.get(dev_label, 0) + 1

        if idx < limit:
            predictions.append(
                {
                    "row_index": int(idx),
                    "device_id": dev_id,
                    "device_name": dev_label,
                    "attack_label": att_label,
                    "attack_score": float(probs_att[idx, att_id]),        # softmax prob in [0,1]
                    "device_confidence": float(probs_dev[idx, dev_id]),   # softmax prob in [0,1]
                }
            )

    return attack_distribution, device_distribution, predictions


# --------------------------------------------------------------
# Demo-friendly endpoints
# --------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "artifacts_loaded": model is not None,
        "num_features": num_features,
    }


@app.get("/api/schema")
def schema():
    return {
        "num_features": num_features,
        "feature_names": feature_names,
    }


@app.get("/api/labels")
def labels():
    # keep response light: counts + samples
    attack_sample = {k: attack_label_map[k] for k in list(attack_label_map.keys())[:10]}
    device_sample = {k: device_label_map[k] for k in list(device_label_map.keys())[:10]}
    return {
        "num_attacks": num_attacks,
        "num_devices": num_devices,
        "attack_label_sample": attack_sample,
        "device_label_sample": device_sample,
    }


@app.get("/api/info")
def info():
    return {
        "model": {
            "type": "MultiTaskCNN1D",
            "device": str(device),
            "num_features": num_features,
            "num_attacks": num_attacks,
            "num_devices": num_devices,
            "weights_file": model_path.name,
        },
        "schema": {
            "feature_names_count": len(feature_names),
        },
        "eval_summary": {
            "attack_accuracy": eval_metrics.get("attack_head", {}).get("accuracy"),
            "device_accuracy": eval_metrics.get("device_head", {}).get("accuracy"),
        } if eval_metrics else {},
    }


# --------------------------------------------------------------
# Single-row JSON predict (used for quick debugging)
# --------------------------------------------------------------
@app.post("/predict")
def predict(data: PacketFeatures, top_k: int = 3):
    start_time = time.perf_counter()

    x = np.array(data.features, dtype=np.float32)
    if x.shape[0] != num_features:
        raise HTTPException(status_code=400, detail=f"Expected {num_features} features, got {x.shape[0]}")

    x = (x - means) / stds
    x = np.clip(x, -10, 10)

    tensor_x = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        logits_att, logits_dev = model(tensor_x)
        probs_att = F.softmax(logits_att, dim=1).cpu().numpy()[0]
        probs_dev = F.softmax(logits_dev, dim=1).cpu().numpy()[0]

        pred_att = int(probs_att.argmax())
        pred_dev = int(probs_dev.argmax())

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    conf_att = float(probs_att[pred_att])
    conf_dev = float(probs_dev[pred_dev])

    k_att = min(top_k, len(probs_att))
    k_dev = min(top_k, len(probs_dev))

    topk_att_idx = probs_att.argsort()[::-1][:k_att]
    topk_dev_idx = probs_dev.argsort()[::-1][:k_dev]

    topk_att = [{"attack_id": int(i), "attack_label": attack_label_map[str(i)], "prob": float(probs_att[i])} for i in topk_att_idx]
    topk_dev = [{"device_id": int(i), "device_label": device_label_map[str(i)], "prob": float(probs_dev[i])} for i in topk_dev_idx]

    response: dict[str, Any] = {
        "attack": {"id": pred_att, "label": attack_label_map[str(pred_att)], "confidence": conf_att, "top_k": topk_att},
        "device": {"id": pred_dev, "label": device_label_map[str(pred_dev)], "confidence": conf_dev, "top_k": topk_dev},
        "latency_ms": float(latency_ms),
    }

    if eval_metrics:
        response["eval_summary"] = {
            "attack_accuracy": eval_metrics.get("attack_head", {}).get("accuracy"),
            "device_accuracy": eval_metrics.get("device_head", {}).get("accuracy"),
        }

    return response


# --------------------------------------------------------------
# RAW CSV upload endpoint for SPA demo
# --------------------------------------------------------------
@app.post("/api/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    max_rows: int = 50000,
    max_predictions_return: int = 5000,
):
    """
    Upload a RAW CIC packet CSV and run:
      raw CSV -> validate required features -> numeric coercion -> fill -> standardise -> clip -> model inference

    Returns:
      - stages: stepper-friendly timings
      - summary: distributions (over ALL rows used)
      - metrics: includes per-row latency + throughput
      - predictions: capped rows for UI table (default 5000)

    Requirements:
      - CSV must contain all columns in `feature_names`
      - Extra columns are allowed and ignored
    """
    t_start = time.perf_counter()

    # browsers sometimes send octet-stream
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # ---- Upload / parse
    upload_bytes = await file.read()
    df = _safe_read_csv(upload_bytes)

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    uploaded_rows = int(len(df))

    # ---- Validate schema
    ok, missing = _validate_required_columns(df, feature_names)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required feature columns for this model.",
                "missing_columns": missing[:50],
                "missing_count": len(missing),
                "hint": "Upload a CIC packet CSV containing the same numeric feature columns used during training.",
            },
        )

    # ---- Limit rows for responsiveness
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    used_rows = int(len(df))

    # ---- Preprocess
    t_pre = time.perf_counter()
    X = _build_feature_matrix_from_raw(df, feature_names, means, stds, clip_value=10.0)
    preprocessing_ms = (time.perf_counter() - t_pre) * 1000.0

    # ---- Inference
    t_inf = time.perf_counter()
    tensor_x = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits_att, logits_dev = model(tensor_x)
        probs_att = F.softmax(logits_att, dim=1).cpu().numpy()
        probs_dev = F.softmax(logits_dev, dim=1).cpu().numpy()
        pred_att_ids = probs_att.argmax(axis=1)
        pred_dev_ids = probs_dev.argmax(axis=1)

    inference_ms = (time.perf_counter() - t_inf) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    # ---- Build outputs
    attack_distribution, device_distribution, predictions = _build_distributions_and_rows(
        probs_att=probs_att,
        probs_dev=probs_dev,
        pred_att_ids=pred_att_ids,
        pred_dev_ids=pred_dev_ids,
        attack_label_map=attack_label_map,
        device_label_map=device_label_map,
        max_predictions_return=max_predictions_return,
    )

    benign_count = sum(v for k, v in attack_distribution.items() if str(k).lower() == "benign")
    n = int(X.shape[0])
    num_attacks_count = n - int(benign_count)

    # ---- Per-row + throughput metrics
    per_row_preprocess_ms = float(preprocessing_ms) / n if n else None
    per_row_inference_ms = float(inference_ms) / n if n else None
    per_row_total_ms = float(total_ms) / n if n else None
    throughput_rows_per_sec = float(n) / (float(inference_ms) / 1000.0) if inference_ms > 0 else None

    response: dict[str, Any] = {
        "stages": {
            "upload": {"status": "done", "rows": uploaded_rows},
            "validate": {"status": "done", "features_expected": int(num_features), "features_present": int(num_features)},
            "preprocess": {"status": "done", "ms": float(preprocessing_ms), "used_rows": used_rows},
            "predict": {"status": "done", "ms": float(inference_ms)},
            "results": {"status": "done"},
        },
        "summary": {
            "total_rows": n,
            "num_benign": int(benign_count),
            "num_attacks": int(num_attacks_count),
            "attack_distribution": attack_distribution,
            "device_distribution": device_distribution,
        },
        "metrics": {
            "preprocessing_ms": float(preprocessing_ms),
            "inference_ms": float(inference_ms),
            "total_ms": float(total_ms),
            "per_row_preprocess_ms": per_row_preprocess_ms,
            "per_row_inference_ms": per_row_inference_ms,
            "per_row_total_ms": per_row_total_ms,
            "throughput_rows_per_sec": throughput_rows_per_sec,
            "max_predictions_return": int(max_predictions_return),
        },
        "predictions": predictions,
    }

    if eval_metrics:
        response["eval_summary"] = {
            "attack_accuracy": eval_metrics.get("attack_head", {}).get("accuracy"),
            "device_accuracy": eval_metrics.get("device_head", {}).get("accuracy"),
        }

    return response