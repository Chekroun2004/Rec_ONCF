from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.metrics import hit_rate_at_k, mrr_at_k
from rec_oncf.training import (
    build_metadata,
    fingerprint_dataframe,
    load_artifacts,
    predict_proba,
    save_artifacts,
    temporal_split,
    train_xgb_multiclass,
)


def _segment_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    user_col: str,
) -> dict[str, dict[str, float]]:
    """Compute HR@1/HR@3/MRR@3 per history-size bucket.

    Buckets are based on the number of bookings the user had in the TRAINING set
    (= bookings observed before the prediction time).
    """
    train_counts = train_df.groupby(user_col).size()
    test_users = test_df[user_col].astype(str).to_numpy()
    train_counts.index = train_counts.index.astype(str)
    user_history_size = np.array([
        train_counts.get(uid, 0) for uid in test_users
    ])

    buckets = {
        "0_2_unseen_in_train": user_history_size <= 2,
        "3_5_new_active":      (user_history_size >= 3) & (user_history_size <= 5),
        "6_20_regular":        (user_history_size >= 6) & (user_history_size <= 20),
        "21_plus_loyal":       user_history_size >= 21,
    }

    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n":          int(mask.sum()),
            "hit_rate@1": float(hit_rate_at_k(y_true[mask], proba[mask], k=1)),
            "hit_rate@3": float(hit_rate_at_k(y_true[mask], proba[mask], k=3)),
            "mrr@3":      float(mrr_at_k(y_true[mask], proba[mask], k=3)),
        }
    return out


def main() -> None:
    paths = default_paths()

    if not paths.features_parquet.exists():
        raise FileNotFoundError(
            f"Missing features: {paths.features_parquet}. Run scripts/02_build_features.py"
        )

    df = read_parquet(paths.features_parquet)

    label_col = "LiaisonId"
    time_col = "DateHeureDepartVoyageSegment"
    user_col = "CodeClient"

    dataset_fp = fingerprint_dataframe(df)
    train_df, test_df = temporal_split(df, time_col=time_col, train_frac=0.8)

    artifacts = train_xgb_multiclass(
        train_df.drop(columns=[user_col]),
        label_col=label_col,
        time_col=time_col,
    )
    # Save model + encoder; metadata is added below once metrics are computed.
    save_artifacts(
        artifacts,
        model_path=paths.xgb_model_path,
        label_encoder_path=paths.label_encoder_path,
    )

    artifacts = load_artifacts(
        model_path=paths.xgb_model_path,
        label_encoder_path=paths.label_encoder_path,
    )

    known_classes = set(artifacts.label_encoder.classes_)
    n_test_before = len(test_df)
    test_df = test_df[test_df[label_col].astype(str).isin(known_classes)].copy()
    n_test_dropped = n_test_before - len(test_df)

    test_df_no_user = test_df.drop(columns=[user_col])
    proba = predict_proba(artifacts, test_df_no_user, label_col=label_col)
    y_true = artifacts.label_encoder.transform(
        test_df[label_col].astype(str).to_numpy()
    )

    hr1 = hit_rate_at_k(y_true, proba, k=1)
    hr3 = hit_rate_at_k(y_true, proba, k=3)
    mrr3 = mrr_at_k(y_true, proba, k=3)

    segments = _segment_metrics(
        train_df, test_df, proba, y_true, user_col=user_col,
    )

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "rows_test_unseen_dropped": int(n_test_dropped),
        "classes": int(len(artifacts.label_encoder.classes_)),
        "n_estimators": 200,
        "max_depth": 6,
        "device": "cpu",
        "hit_rate@1": float(hr1),
        "hit_rate@3": float(hr3),
        "mrr@3": float(mrr3),
        "thresholds_met": {
            "hit_rate@1 > 0.30": hr1 > 0.30,
            "hit_rate@3 > 0.50": hr3 > 0.50,
            "mrr@3 > 0.35": mrr3 > 0.35,
        },
        "metrics_by_history_segment": segments,
    }

    out_path = paths.project_root / "reports" / "offline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Embed metadata sidecar next to the model for auditability
    metadata = build_metadata(
        artifacts,
        train_rows=len(train_df),
        test_rows=len(test_df),
        metrics={
            "hit_rate@1": float(hr1),
            "hit_rate@3": float(hr3),
            "mrr@3": float(mrr3),
        },
        dataset_fingerprint=dataset_fp,
    )
    save_artifacts(
        artifacts,
        model_path=paths.xgb_model_path,
        label_encoder_path=paths.label_encoder_path,
        metadata=metadata,
    )

    print(json.dumps(report, indent=2))
    print(f"Saved model: {paths.xgb_model_path}")
    print(f"Saved label encoder: {paths.label_encoder_path}")
    print(f"Saved metadata: {paths.xgb_model_path.with_suffix('.meta.json')}")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
