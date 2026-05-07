from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_oncf.config import Paths
from rec_oncf.retrain import (
    check_guardrail,
    evaluate_model,
    load_current_metrics,
    promote_artifacts,
    retrain_pipeline,
)
from rec_oncf.training import temporal_split, train_xgb_multiclass


# ---------- shared helpers ----------

def _make_mini_features(n: int = 200) -> pd.DataFrame:
    """Mini features DataFrame with the same schema as features.parquet."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "LiaisonId": rng.choice(["R1", "R2", "R3"], n),
        "DateHeureDepartVoyageSegment": pd.date_range("2024-01-01", periods=n, freq="h"),
        "prev_liaison": rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId": rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId": rng.choice(["1", "2"], n),
        "TrainAutocarId": rng.choice(["10", "20"], n),
        "CarteClientId": rng.choice(["0", "1"], n),
        "PrixParLiaison": rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment": np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index": np.arange(n, dtype=float),
        "days_since_prev": rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour": rng.integers(0, 24, n).astype(float),
        "depart_dow": rng.integers(0, 7, n).astype(float),
        "depart_month": rng.integers(1, 13, n).astype(float),
        "depart_hour_sin": rng.standard_normal(n),
        "depart_hour_cos": rng.standard_normal(n),
        "depart_dow_sin": rng.standard_normal(n),
        "depart_dow_cos": rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n).astype(float),
    })


def _make_mini_clean(n: int = 100) -> pd.DataFrame:
    """Minimal oncf_clean DataFrame for cold-start CF building."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "CodeClient": rng.choice(["C1", "C2", "C3"], n).astype(str),
        "LiaisonId": rng.choice(["R1", "R2", "R3"], n).astype(str),
        "DateHeureDepartVoyageSegment": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _make_test_paths(base: Path) -> Paths:
    """Build a Paths dataclass pointing entirely to tmp_path subdirs."""
    models_dir = base / "models"
    models_dir.mkdir()
    processed_dir = base / "data" / "processed"
    processed_dir.mkdir(parents=True)
    return Paths(
        project_root=base,
        desktop=base,
        raw_oncf_data=base / "oncf_data.csv",
        raw_liaison=base / "Liaison.csv",
        processed_dir=processed_dir,
        processed_dataset_parquet=processed_dir / "oncf_clean.parquet",
        processed_dataset_csv=processed_dir / "oncf_clean.csv",
        features_parquet=processed_dir / "features.parquet",
        models_dir=models_dir,
        xgb_model_path=models_dir / "xgb_ranker.json",
        label_encoder_path=models_dir / "label_encoder.joblib",
        cold_start_path=models_dir / "cold_start.joblib",
        onnx_model_path=models_dir / "xgb_ranker.onnx",
    )


# ---------- load_current_metrics ----------

def test_load_current_metrics_returns_empty_when_file_missing(tmp_path):
    metrics = load_current_metrics(tmp_path / "xgb_ranker.meta.json")
    assert metrics == {}


def test_load_current_metrics_reads_metrics_key(tmp_path):
    meta = {"metrics": {"hit_rate@1": 0.76, "hit_rate@3": 0.90}, "other": "ignored"}
    (tmp_path / "xgb_ranker.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    metrics = load_current_metrics(tmp_path / "xgb_ranker.meta.json")
    assert metrics["hit_rate@1"] == pytest.approx(0.76)
    assert metrics["hit_rate@3"] == pytest.approx(0.90)
    assert "other" not in metrics


# ---------- check_guardrail ----------

def test_check_guardrail_passes_when_no_current():
    passes, reason = check_guardrail({}, {"hit_rate@1": 0.5})
    assert passes
    assert "waived" in reason


def test_check_guardrail_passes_small_drop():
    passes, _ = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.73})
    assert passes  # 0.02 drop < 0.05


def test_check_guardrail_passes_improvement():
    passes, _ = check_guardrail({"hit_rate@1": 0.70}, {"hit_rate@1": 0.76})
    assert passes


def test_check_guardrail_passes_at_exact_threshold():
    # Drop exactly equal to threshold passes (condition is strictly greater than)
    passes, _ = check_guardrail({"hit_rate@1": 0.80}, {"hit_rate@1": 0.75})
    assert passes  # 0.05 drop == 0.05 threshold → not strictly greater → passes


def test_check_guardrail_fails_above_threshold():
    passes, reason = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.69})
    assert not passes  # 0.06 drop > 0.05
    assert "BLOCKED" in reason


def test_check_guardrail_custom_threshold():
    passes, _ = check_guardrail(
        {"hit_rate@1": 0.80}, {"hit_rate@1": 0.75}, threshold=0.10
    )
    assert passes  # 0.05 drop < 0.10 custom threshold


# ---------- evaluate_model ----------

def test_evaluate_model_returns_correct_keys():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    metrics = evaluate_model(arts, df)
    assert set(metrics.keys()) == {"hit_rate@1", "hit_rate@3", "mrr@3", "test_rows"}


def test_evaluate_model_metrics_in_range():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    metrics = evaluate_model(arts, df)
    assert 0.0 <= metrics["hit_rate@1"] <= 1.0
    assert 0.0 <= metrics["hit_rate@3"] <= 1.0
    assert 0.0 <= metrics["mrr@3"] <= 1.0
    assert metrics["test_rows"] > 0


# ---------- promote_artifacts ----------

def test_promote_artifacts_copies_files(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (staging / "xgb_ranker.json").write_bytes(b"model-data")
    (staging / "label_encoder.joblib").write_bytes(b"le-data")
    (staging / "xgb_ranker.meta.json").write_text('{"ok": true}', encoding="utf-8")

    promote_artifacts(staging, models)

    assert (models / "xgb_ranker.json").read_bytes() == b"model-data"
    assert (models / "label_encoder.joblib").read_bytes() == b"le-data"
    assert (models / "xgb_ranker.meta.json").exists()


def test_promote_artifacts_overwrites_existing(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (models / "xgb_ranker.json").write_bytes(b"old")
    (staging / "xgb_ranker.json").write_bytes(b"new")

    promote_artifacts(staging, models)

    assert (models / "xgb_ranker.json").read_bytes() == b"new"


# ---------- retrain_pipeline ----------

def test_retrain_pipeline_promotes_on_first_run(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(
        paths,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["guardrail_passes"] is True
    assert report["promoted"] is True
    assert report["dry_run"] is False
    assert (paths.models_dir / "xgb_ranker.json").exists()
    assert (paths.models_dir / "label_encoder.joblib").exists()
    assert (paths.models_dir / "xgb_ranker.onnx").exists()
    assert (paths.models_dir / "cold_start.joblib").exists()


def test_retrain_pipeline_dry_run_does_not_promote(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(
        paths,
        dry_run=True,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["dry_run"] is True
    assert report["promoted"] is False
    assert not (paths.models_dir / "xgb_ranker.json").exists()


def test_retrain_pipeline_blocked_by_guardrail(tmp_path):
    paths = _make_test_paths(tmp_path)
    meta = {"metrics": {"hit_rate@1": 0.999, "hit_rate@3": 0.999, "mrr@3": 0.999}}
    (paths.models_dir / "xgb_ranker.meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    report = retrain_pipeline(
        paths,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["guardrail_passes"] is False
    assert report["promoted"] is False
    assert not (paths.models_dir / "xgb_ranker.json").exists()
