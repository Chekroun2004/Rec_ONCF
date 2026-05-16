"""Microbenchmark: profile each step of the hot recommendation path."""
from __future__ import annotations
import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
n = 50
dates = [datetime(2024, 1, 1) + timedelta(days=int(d)) for d in sorted(rng.integers(0, 500, n))]
history = pd.DataFrame({
    "CodeClient": ["12345"] * n,
    "AchteurId":  ["12345"] * n,
    "LiaisonId":  [str(rng.integers(1, 100)) for _ in range(n)],
    "DateHeureDepartVoyageSegment": dates,
    "TypeParcoursId": rng.choice(["1", "2"], n),
    "ClassificationId": rng.choice(["1", "2"], n),
    "ClassePhysiqueId": rng.choice(["1", "2"], n),
    "NiveauPrixId": rng.choice(["1", "2"], n),
    "TrainAutocarId": rng.choice(["10", "20"], n),
    "CarteClientId": rng.choice(["0", "1"], n),
    "PrixParLiaison": rng.uniform(50, 300, n),
    "NbrVoySegment": np.ones(n),
    "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
})

from rec_oncf.features import compute_inference_row
from rec_oncf.candidates import generate_candidates
from rec_oncf.training import load_artifacts, predict_proba_onnx
from rec_oncf.config import default_paths
from onnxruntime import InferenceSession

paths = default_paths()
arts = load_artifacts(model_path=paths.xgb_model_path, label_encoder_path=paths.label_encoder_path)
session = InferenceSession(str(paths.onnx_model_path))
preprocessor = arts.pipeline["pre"]

RUNS = 100

def bench(label, fn, runs=RUNS):
    t0 = time.perf_counter()
    for _ in range(runs):
        result = fn()
    ms = (time.perf_counter() - t0) / runs * 1000
    print(f"{label:<35} {ms:7.2f} ms")
    return result, ms

cands, t_cands = bench("generate_candidates", lambda: generate_candidates(history, user_id="12345", max_candidates=10))
feat_row, t_feat = bench("compute_inference_row", lambda: compute_inference_row(history))

drop_cols = [c for c in ["LiaisonId", "CodeClient"] if c in feat_row.columns]
drop_cols += [c for c in feat_row.columns if feat_row[c].dtype.kind == "M"]

_, t_drop = bench("feat_row.drop + dtype filter", lambda: feat_row.drop(columns=drop_cols))
X = feat_row.drop(columns=drop_cols)
_, t_pre = bench("preprocessor.transform", lambda: preprocessor.transform(X).astype(np.float32))
X_pre = preprocessor.transform(X).astype(np.float32)
_, t_onnx = bench("onnx session.run", lambda: session.run(["probabilities"], {"input": X_pre})[0])
proba, t_full = bench("predict_proba_onnx (full)", lambda: predict_proba_onnx(session, preprocessor, feat_row, label_col="LiaisonId")[0])

le = arts.label_encoder
known = set(le.classes_)
valid = [c for c in cands if c in known]
_, t_score = bench("filter+argsort candidates", lambda: (
    np.argsort(-proba[le.transform(np.asarray(valid))])[:3]
) if valid else [])

total = t_cands + t_feat + t_full + t_score
print("-" * 45)
print(f"{'Total (logic only)':<35} {total:7.2f} ms")
print()
print("Breakdown:")
for label, t in [("candidates", t_cands), ("feat row build", t_feat),
                  ("preprocessing", t_pre), ("onnx model", t_onnx), ("scoring", t_score)]:
    print(f"  {label:<20} {t/total*100:5.1f}%  ({t:.2f} ms)")

print()
print("=" * 45)
print("FastPreprocessor optimization:")
print()

from rec_oncf.training import FastPreprocessor

fp = FastPreprocessor(preprocessor)
row_dict = feat_row.iloc[0].to_dict()

_, t_fast = bench("FastPreprocessor.encode (NEW)", lambda: fp.encode(row_dict))

print()
print(f"Preprocessor savings: {t_pre:.2f} ms  ->  {t_fast:.2f} ms  ({t_pre/t_fast:.0f}x speedup)")
new_total = t_cands + t_feat + t_fast + t_onnx + t_score
print(f"Projected logic total: {new_total:.2f} ms  (was {total:.2f} ms)")
