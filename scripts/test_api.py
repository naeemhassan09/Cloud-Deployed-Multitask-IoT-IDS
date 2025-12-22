import json
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 1) Load scaler to get feature column order
with open(PROCESSED_DIR / "multitask_cnn_mlp_scaler.json", "r") as f:
    scaler = json.load(f)

feature_cols = list(scaler["means"].keys())
print(f"Using {len(feature_cols)} features:")
print(feature_cols[:10], "...")

# 2) Load test data
df = pd.read_csv(PROCESSED_DIR / "packets_test.csv")

# Pick one row to test (index 1 in your earlier example)
row = df.iloc[1]

# 3) Build feature vector in the correct order
features = []
for col in feature_cols:
    val = row[col]
    if pd.isna(val):
        val = 0.0
    features.append(float(val))

payload = {"features": features}

# Dump payload to reuse in curl
print("\n=== JSON payload for curl ===")
print(json.dumps(payload))        # one-line JSON

# 4) Optional: call the API directly from Python (like you already did)
resp = requests.post("http://127.0.0.1:8000/predict?top_k=3", json=payload)
print("\nStatus:", resp.status_code)
print("Response:", resp.json())