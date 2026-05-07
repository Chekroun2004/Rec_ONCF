from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from rec_oncf.cold_start import build_cold_start_recommender, save_cold_start
from rec_oncf.config import Paths
from rec_oncf.io import read_parquet
from rec_oncf.metrics import hit_rate_at_k, mrr_at_k
from rec_oncf.training import (
    TrainArtifacts,
    build_metadata,
    export_onnx,
    fingerprint_dataframe,
    predict_proba,
    save_artifacts,
    temporal_split,
    train_xgb_multiclass,
)


def load_current_metrics(meta_path: Path) -> dict:
    """Load metrics from a model sidecar JSON. Returns {} when file is missing (first run)."""
    if not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return data.get("metrics", {})


def check_guardrail(
    current: dict,
    new: dict,
    *,
    threshold: float = 0.05,
) -> tuple[bool, str]:
    """Return (passes, reason). Fails only if new HR@1 drops strictly more than threshold."""
    if not current:
        return True, "No current metrics — guardrail waived (first run)"
    current_hr1 = current.get("hit_rate@1", 0.0)
    new_hr1 = new.get("hit_rate@1", 0.0)
    drop = current_hr1 - new_hr1
    if drop - threshold > 1e-9:
        return (
            False,
            f"BLOCKED: HR@1 dropped {drop:.4f} "
            f"(current={current_hr1:.4f}, new={new_hr1:.4f}, threshold={threshold:.4f})",
        )
    return (
        True,
        f"OK: HR@1 current={current_hr1:.4f}, new={new_hr1:.4f}, drop={drop:.4f}",
    )


def evaluate_model(
    artifacts: TrainArtifacts,
    features_df: pd.DataFrame,
    *,
    label_col: str = "LiaisonId",
    time_col: str = "DateHeureDepartVoyageSegment",
    train_frac: float = 0.8,
) -> dict:
    """Evaluate artifacts on the temporal test split of features_df.

    Returns {"hit_rate@1": float, "hit_rate@3": float, "mrr@3": float, "test_rows": int}.
    """
    _, df_test = temporal_split(features_df, time_col=time_col, train_frac=train_frac)
    known = set(artifacts.label_encoder.classes_)
    df_test = df_test[df_test[label_col].astype(str).isin(known)]
    if df_test.empty:
        return {"hit_rate@1": 0.0, "hit_rate@3": 0.0, "mrr@3": 0.0, "test_rows": 0}
    proba = predict_proba(artifacts, df_test, label_col=label_col)
    y_true = artifacts.label_encoder.transform(df_test[label_col].astype(str).to_numpy())
    return {
        "hit_rate@1": hit_rate_at_k(y_true, proba, k=1),
        "hit_rate@3": hit_rate_at_k(y_true, proba, k=3),
        "mrr@3": mrr_at_k(y_true, proba, k=3),
        "test_rows": len(df_test),
    }


def promote_artifacts(staging_dir: Path, models_dir: Path) -> None:
    """Copy all files from staging_dir into models_dir, overwriting existing files."""
    for src in staging_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, models_dir / src.name)


def retrain_pipeline(
    paths: Paths,
    *,
    dry_run: bool = False,
    features_df: pd.DataFrame | None = None,
    clean_df: pd.DataFrame | None = None,
) -> dict:
    """Full pipeline: retrain → evaluate → guardrail → promote.

    Returns a report dict. Pass features_df / clean_df to skip disk reads (useful in tests).
    """
    if features_df is None:
        features_df = read_parquet(paths.features_parquet)
    if clean_df is None:
        clean_df = read_parquet(paths.processed_dataset_parquet)

    current_metrics = load_current_metrics(paths.xgb_model_path.with_suffix(".meta.json"))

    df_train, _ = temporal_split(features_df, time_col="DateHeureDepartVoyageSegment")

    new_arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    new_metrics = evaluate_model(new_arts, features_df)

    passes, reason = check_guardrail(current_metrics, new_metrics)

    report = {
        "current_metrics": current_metrics,
        "new_metrics": new_metrics,
        "guardrail_passes": passes,
        "guardrail_reason": reason,
        "promoted": False,
        "dry_run": dry_run,
    }

    if not passes:
        return report

    staging_dir = paths.models_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        new_arts,
        train_rows=len(df_train),
        test_rows=new_metrics["test_rows"],
        metrics={k: v for k, v in new_metrics.items() if k != "test_rows"},
        dataset_fingerprint=fingerprint_dataframe(features_df),
    )
    save_artifacts(
        new_arts,
        model_path=staging_dir / paths.xgb_model_path.name,
        label_encoder_path=staging_dir / paths.label_encoder_path.name,
        metadata=metadata,
    )

    cs = build_cold_start_recommender(clean_df)
    save_cold_start(cs, staging_dir / paths.cold_start_path.name)

    export_onnx(new_arts.pipeline, staging_dir / paths.onnx_model_path.name)

    if not dry_run:
        promote_artifacts(staging_dir, paths.models_dir)
        report["promoted"] = True

    return report
