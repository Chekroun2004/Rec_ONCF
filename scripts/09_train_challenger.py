"""Train a challenger XGBoost model and export it for A/B testing.

Challenger hyperparameters vs production:
  max_depth      8  (prod: 6)   — deeper trees, more expressive
  n_estimators   250 (prod: 200) — more boosting rounds
  learning_rate  0.06 (prod: 0.08) — slower LR to compensate for extra rounds
  subsample      0.85 (prod: 0.90)
  colsample_bytree 0.75 (prod: 0.80)
  reg_lambda     1.5 (prod: 1.0)  — extra regularization for deeper trees

Outputs (in models/):
  xgb_ranker_challenger.json        — joblib pipeline
  label_encoder_challenger.joblib   — fitted LabelEncoder
  xgb_ranker_challenger.onnx        — ONNX export for fast inference
  xgb_ranker_challenger.meta.json   — metadata sidecar

Report: reports/challenger_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.metrics import hit_rate_at_k, mrr_at_k
from rec_oncf.training import (
    TrainArtifacts,
    build_metadata,
    export_onnx,
    fingerprint_dataframe,
    save_artifacts,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Challenger hyperparameters
# ---------------------------------------------------------------------------
CHALLENGER_PARAMS = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    device="cpu",
    n_estimators=250,
    learning_rate=0.06,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=1.5,
    n_jobs=-1,
    random_state=42,
)

PROD_PARAMS = dict(n_estimators=200, learning_rate=0.08, max_depth=6, subsample=0.90,
                   colsample_bytree=0.80, reg_lambda=1.0)


def _train(df_train: pd.DataFrame, *, label_col: str, time_col: str) -> TrainArtifacts:
    df_train = df_train.sort_values(time_col)

    y_raw = df_train[label_col].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    _drop = [c for c in [label_col, time_col, "CodeClient"] if c in df_train.columns]
    X = df_train.drop(columns=_drop)

    cat_cols = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    clf = xgb.XGBClassifier(**CHALLENGER_PARAMS)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return TrainArtifacts(pipeline=pipe, label_encoder=le)


def _predict_proba(arts: TrainArtifacts, df: pd.DataFrame, *, label_col: str) -> np.ndarray:
    drop = [c for c in [label_col, "CodeClient"] if c in df.columns]
    drop += [c for c in df.columns if df[c].dtype.kind == "M"]
    X = df.drop(columns=drop)
    return arts.pipeline.predict_proba(X)


def _benchmark_onnx(session, preprocessor, feat_row, *, label_col: str, n: int = 20) -> float:
    from rec_oncf.training import predict_proba_onnx
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_proba_onnx(session, preprocessor, feat_row, label_col=label_col)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    paths = default_paths()

    if not paths.features_parquet.exists():
        raise FileNotFoundError(
            f"Missing features: {paths.features_parquet}. Run scripts/02_build_features.py first."
        )

    print("Loading features ...")
    df = read_parquet(paths.features_parquet)
    dataset_fp = fingerprint_dataframe(df)

    label_col = "LiaisonId"
    time_col = "DateHeureDepartVoyageSegment"
    user_col = "CodeClient"

    train_df, test_df = temporal_split(df, time_col=time_col, train_frac=0.8)
    print(f"  Train: {len(train_df):,} rows  |  Test (before filter): {len(test_df):,} rows")

    # -----------------------------------------------------------------------
    # Train challenger
    # -----------------------------------------------------------------------
    print("\nTraining challenger model ...")
    print(f"  Params: {CHALLENGER_PARAMS}")
    t_start = time.time()
    arts = _train(train_df.drop(columns=[user_col]), label_col=label_col, time_col=time_col)
    elapsed = time.time() - t_start
    print(f"  Done in {elapsed / 60:.1f} min")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    known_classes = set(arts.label_encoder.classes_)
    n_before = len(test_df)
    test_df = test_df[test_df[label_col].astype(str).isin(known_classes)].copy()
    n_dropped = n_before - len(test_df)
    print(f"\nEvaluation — dropped {n_dropped} test rows with unseen labels")

    test_no_user = test_df.drop(columns=[user_col])
    proba = _predict_proba(arts, test_no_user, label_col=label_col)
    y_true = arts.label_encoder.transform(test_df[label_col].astype(str).to_numpy())

    hr1  = float(hit_rate_at_k(y_true, proba, k=1))
    hr3  = float(hit_rate_at_k(y_true, proba, k=3))
    mrr3 = float(mrr_at_k(y_true, proba, k=3))

    # Load prod metrics for comparison
    prod_metrics_path = PROJECT_ROOT / "reports" / "offline_metrics.json"
    prod = {}
    if prod_metrics_path.exists():
        prod = json.loads(prod_metrics_path.read_text(encoding="utf-8"))

    prod_hr1  = prod.get("hit_rate@1", float("nan"))
    prod_hr3  = prod.get("hit_rate@3", float("nan"))
    prod_mrr3 = prod.get("mrr@3",      float("nan"))

    def _delta(challenger_val: float, prod_val: float) -> str:
        if prod_val != prod_val:  # nan
            return "n/a"
        d = (challenger_val - prod_val) * 100
        return f"{d:+.2f} pp"

    print("\n=== Offline Metrics Comparison ===")
    print(f"{'Metric':<14} {'Prod (A)':>10} {'Challenger (B)':>15} {'Delta':>12}")
    print("-" * 55)
    print(f"{'hit_rate@1':<14} {prod_hr1:>10.4f} {hr1:>15.4f} {_delta(hr1, prod_hr1):>12}")
    print(f"{'hit_rate@3':<14} {prod_hr3:>10.4f} {hr3:>15.4f} {_delta(hr3, prod_hr3):>12}")
    print(f"{'mrr@3':<14} {prod_mrr3:>10.4f} {mrr3:>15.4f} {_delta(mrr3, prod_mrr3):>12}")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    challenger_model_path = paths.models_dir / "xgb_ranker_challenger.json"
    challenger_le_path    = paths.models_dir / "label_encoder_challenger.joblib"
    challenger_onnx_path  = paths.models_dir / "xgb_ranker_challenger.onnx"

    metadata = build_metadata(
        arts,
        train_rows=len(train_df),
        test_rows=len(test_df),
        metrics={"hit_rate@1": hr1, "hit_rate@3": hr3, "mrr@3": mrr3},
        dataset_fingerprint=dataset_fp,
    )
    metadata["challenger_params"] = CHALLENGER_PARAMS
    metadata["prod_params"] = PROD_PARAMS

    save_artifacts(
        arts,
        model_path=challenger_model_path,
        label_encoder_path=challenger_le_path,
        metadata=metadata,
    )
    print(f"\nSaved: {challenger_model_path}")
    print(f"Saved: {challenger_le_path}")
    print(f"Saved: {challenger_model_path.with_suffix('.meta.json')}")

    # -----------------------------------------------------------------------
    # ONNX export
    # -----------------------------------------------------------------------
    print(f"\nExporting ONNX -> {challenger_onnx_path} ...")
    export_onnx(arts.pipeline, challenger_onnx_path)
    size_mb = challenger_onnx_path.stat().st_size / 1e6
    print(f"  Done ({size_mb:.1f} MB)")

    print("\nBenchmarking ONNX p50 ...")
    from onnxruntime import InferenceSession
    from rec_oncf.features import compute_inference_row

    session = InferenceSession(str(challenger_onnx_path))
    clean = read_parquet(paths.processed_dataset_parquet)
    history_lookup = {
        str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
        for cid, grp in clean.groupby("CodeClient")
    }
    warm_users = [(uid, len(df)) for uid, df in history_lookup.items() if len(df) >= 10]
    warm_users.sort(key=lambda x: -x[1])
    uid, n_trips = warm_users[0]
    feat_row = compute_inference_row(history_lookup[uid])

    p50_ms = _benchmark_onnx(session, arts.pipeline["pre"], feat_row, label_col=label_col)
    print(f"  ONNX p50: {p50_ms:.1f} ms  (target < 30 ms)")
    if p50_ms >= 30:
        print("  WARNING: p50 >= 30 ms — check onnxruntime installation")

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "rows_test_unseen_dropped": int(n_dropped),
        "classes": int(len(arts.label_encoder.classes_)),
        "challenger_params": CHALLENGER_PARAMS,
        "hit_rate@1": hr1,
        "hit_rate@3": hr3,
        "mrr@3": mrr3,
        "prod_hit_rate@1": prod_hr1,
        "prod_hit_rate@3": prod_hr3,
        "prod_mrr@3": prod_mrr3,
        "delta_hit_rate@1_pp": round((hr1 - prod_hr1) * 100, 4) if prod_hr1 == prod_hr1 else None,
        "delta_hit_rate@3_pp": round((hr3 - prod_hr3) * 100, 4) if prod_hr3 == prod_hr3 else None,
        "delta_mrr@3_pp": round((mrr3 - prod_mrr3) * 100, 4) if prod_mrr3 == prod_mrr3 else None,
        "onnx_p50_ms": round(p50_ms, 2),
        "training_time_min": round(elapsed / 60, 2),
        "dataset_fingerprint": dataset_fp,
    }

    report_path = PROJECT_ROOT / "reports" / "challenger_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    print("\n=== Summary ===")
    verdict = "BETTER" if hr1 > prod_hr1 else "WORSE" if hr1 < prod_hr1 else "EQUAL"
    print(f"Challenger HR@1: {hr1:.4f}  vs  Prod HR@1: {prod_hr1:.4f}  -> {verdict}")
    print("Variant B is now active — restart the API to pick up the challenger.")


if __name__ == "__main__":
    main()
