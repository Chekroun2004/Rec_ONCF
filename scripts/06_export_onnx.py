from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import export_onnx, load_artifacts, predict_proba, predict_proba_onnx


def _benchmark(fn, n: int = 20) -> float:
    """Return median latency in ms over n calls."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    paths = default_paths()

    if not paths.xgb_model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {paths.xgb_model_path}. Run scripts/03_train_ranker.py first."
        )

    print("Loading artifacts ...")
    arts = load_artifacts(
        model_path=paths.xgb_model_path,
        label_encoder_path=paths.label_encoder_path,
    )

    print(f"Exporting ONNX -> {paths.onnx_model_path} ...")
    export_onnx(arts.pipeline, paths.onnx_model_path)
    size_mb = paths.onnx_model_path.stat().st_size / 1e6
    print(f"  Done ({size_mb:.1f} MB)")

    print("Loading ONNX session ...")
    from onnxruntime import InferenceSession
    session = InferenceSession(str(paths.onnx_model_path))

    print("Building benchmark row ...")
    clean = read_parquet(paths.processed_dataset_parquet)
    history_lookup = {
        str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
        for cid, grp in clean.groupby("CodeClient")
    }
    warm_users = [(uid, len(df)) for uid, df in history_lookup.items() if len(df) >= 10]
    warm_users.sort(key=lambda x: -x[1])
    uid, n_trips = warm_users[0]
    history = history_lookup[uid]
    feat_row = compute_inference_row(history)
    print(f"  User {uid} ({n_trips} trips)")

    preprocessor = arts.pipeline["pre"]

    p50_sklearn = _benchmark(lambda: predict_proba(arts, feat_row, label_col="LiaisonId"))
    p50_onnx = _benchmark(
        lambda: predict_proba_onnx(session, preprocessor, feat_row, label_col="LiaisonId")
    )

    speedup = p50_sklearn / p50_onnx
    print()
    print(f"XGBoost sklearn p50 : {p50_sklearn:.1f} ms")
    print(f"XGBoost ONNX    p50 : {p50_onnx:.1f} ms")
    print(f"Speedup             : {speedup:.1f}x")

    if p50_onnx >= 30:
        raise RuntimeError(
            f"ONNX p50 = {p50_onnx:.1f} ms >= 30 ms target. "
            "Check onnxruntime installation or model export."
        )
    print(f"\nTarget met: ONNX p50 {p50_onnx:.1f} ms < 30 ms")


if __name__ == "__main__":
    main()
