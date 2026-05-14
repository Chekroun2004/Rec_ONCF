"""Measure end-to-end API latency via TestClient (no network overhead)."""
from __future__ import annotations
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from apps.api.main import app
from rec_oncf.recommender import Recommender
from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.schedule import build_liaison_station_map

paths = default_paths()
clean = read_parquet(paths.processed_dataset_parquet)

print("Loading model and ONNX session...")
rec = Recommender.from_paths(paths)
app.state.recommender = rec
app.state.liaison_map = build_liaison_station_map(clean)
app.state.redis = None

# Pick a warm user (one with lots of history)
top_user = clean.groupby("CodeClient").size().sort_values(ascending=False).index[0]
code_client = str(top_user)
print(f"Benchmarking user {code_client} (history size: {len(rec.history_lookup[code_client])})")

client = TestClient(app)

# Warmup
for _ in range(5):
    client.post("/recommend", json={"code_client": code_client, "k": 3})

RUNS = 200
latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    resp = client.post("/recommend", json={"code_client": code_client, "k": 3})
    ms = (time.perf_counter() - t0) * 1000
    latencies.append(ms)

latencies.sort()
p50 = latencies[RUNS // 2]
p95 = latencies[int(RUNS * 0.95)]
p99 = latencies[int(RUNS * 0.99)]
mn  = min(latencies)
mx  = max(latencies)

print(f"\n{RUNS} requests, k=3, mode={resp.json()['mode']}")
print(f"  min:  {mn:.2f} ms")
print(f"  p50:  {p50:.2f} ms")
print(f"  p95:  {p95:.2f} ms")
print(f"  p99:  {p99:.2f} ms")
print(f"  max:  {mx:.2f} ms")
