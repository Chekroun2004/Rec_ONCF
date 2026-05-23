"""ML tests: training, metrics, ONNX, retrain pipeline, popularity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.config import Paths
from rec_oncf.metrics import mrr_at_k
from rec_oncf.popularity import build_popularity_list, load_popularity, save_popularity
from rec_oncf.retrain import (
    check_guardrail,
    evaluate_model,
    load_current_metrics,
    promote_artifacts,
    retrain_pipeline,
    write_challenger_artifacts,
)
from rec_oncf.training import (
    FastPreprocessor,
    export_onnx,
    predict_proba,
    predict_proba_onnx,
    temporal_split,
    train_xgb_multiclass,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def _tiny_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "LiaisonId": rng.choice(["A", "B", "C"], n),
        "DateHeureDepartVoyageSegment": pd.date_range("2020-01-01", periods=n, freq="D"),
        "TypeParcoursId":   rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":     rng.choice(["1", "2"], n),
        "TrainAutocarId":   rng.choice(["10", "20"], n),
        "CarteClientId":    rng.choice(["0", "1"], n),
        "prev_liaison":     rng.choice(["A", "B", "nan"], n),
        "TrajetAllerRetour": rng.choice(["AR", "AS"], n),
        "PrixParLiaison":   rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":    np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index":  np.arange(n),
        "days_since_prev":  rng.choice([np.nan, 7.0, 14.0], n),
        "depart_hour":      rng.integers(0, 24, n),
        "depart_dow":       rng.integers(0, 7, n),
        "depart_month":     rng.integers(1, 13, n),
        "depart_hour_sin":  rng.standard_normal(n),
        "depart_hour_cos":  rng.standard_normal(n),
        "depart_dow_sin":   rng.standard_normal(n),
        "depart_dow_cos":   rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n),
    })


def _make_mini_features(n: int = 200) -> pd.DataFrame:
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
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "CodeClient": rng.choice(["C1", "C2", "C3"], n).astype(str),
        "LiaisonId": rng.choice(["R1", "R2", "R3"], n).astype(str),
        "DateHeureDepartVoyageSegment": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _make_test_paths(base: Path) -> Paths:
    models_dir = base / "models"
    models_dir.mkdir()
    raw_dir = base / "data" / "raw"
    raw_dir.mkdir(parents=True)
    clean_dir = base / "data" / "clean"
    (clean_dir / "parquet").mkdir(parents=True)
    features_dir = base / "data" / "features"
    (features_dir / "parquet").mkdir(parents=True)
    return Paths(
        project_root=base,
        raw_dir=raw_dir,
        raw_oncf_data=raw_dir / "oncf_data.csv",
        raw_liaison=raw_dir / "Liaison.csv",
        horaire_csv_path=raw_dir / "horaire.csv",
        clean_dir=clean_dir,
        processed_dataset_parquet=clean_dir / "parquet" / "oncf_clean.parquet",
        features_dir=features_dir,
        features_parquet=features_dir / "parquet" / "oncf_features.parquet",
        models_dir=models_dir,
        xgb_model_path=models_dir / "xgb_ranker.json",
        label_encoder_path=models_dir / "label_encoder.joblib",
        cold_start_path=models_dir / "cold_start.joblib",
        onnx_model_path=models_dir / "xgb_ranker.onnx",
        popularity_path=models_dir / "popularity.joblib",
        schedule_index_path=models_dir / "schedule_index.joblib",
    )


def _make_onnx_pipeline_and_row():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "prev_liaison":     rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId":   rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":     rng.choice(["1", "2"], n),
        "TrainAutocarId":   rng.choice(["10", "20"], n),
        "CarteClientId":    rng.choice(["0", "1"], n),
        "PrixParLiaison":   rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":    np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index":  np.arange(n, dtype=float),
        "days_since_prev":  rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour":      rng.integers(0, 24, n).astype(float),
        "depart_dow":       rng.integers(0, 7, n).astype(float),
        "depart_month":     rng.integers(1, 13, n).astype(float),
        "depart_hour_sin":  rng.standard_normal(n),
        "depart_hour_cos":  rng.standard_normal(n),
        "depart_dow_sin":   rng.standard_normal(n),
        "depart_dow_cos":   rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n).astype(float),
    })
    y_raw = rng.choice(["R1", "R2", "R3"], n)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    cat_cols = [
        "prev_liaison", "TypeParcoursId", "ClassificationId",
        "ClassePhysiqueId", "NiveauPrixId", "TrainAutocarId", "CarteClientId",
    ]
    num_cols = [c for c in df.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=5, device="cpu",
        eval_metric="mlogloss", random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(df, y)
    row = df.iloc[[0]].copy()
    row["LiaisonId"] = "__unknown__"
    row["CodeClient"] = 12345
    row["DateHeureDepartVoyageSegment"] = pd.Timestamp("2026-01-01")
    return pipe, row, len(le.classes_)


# ── training tests ────────────────────────────────────────────────────────────

def test_train_handles_nulls_and_returns_proba():
    df = _tiny_df()
    arts = train_xgb_multiclass(df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment")
    proba = predict_proba(arts, df, label_col="LiaisonId")
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_artifacts_have_correct_classes():
    df = _tiny_df()
    arts = train_xgb_multiclass(df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment")
    assert set(arts.label_encoder.classes_) == {"A", "B", "C"}


def test_train_xgb_accepts_hyperparam_overrides():
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment",
        n_estimators=10, max_depth=3, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.75, reg_lambda=1.5,
    )
    clf = arts.pipeline.named_steps["clf"]
    assert clf.n_estimators == 10
    assert clf.max_depth == 3
    assert clf.learning_rate == 0.06
    assert clf.subsample == 0.85
    assert clf.colsample_bytree == 0.75
    assert clf.reg_lambda == 1.5


def test_train_xgb_defaults_unchanged():
    df = _tiny_df()
    arts = train_xgb_multiclass(df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment")
    clf = arts.pipeline.named_steps["clf"]
    assert clf.n_estimators == 200
    assert clf.max_depth == 6
    assert clf.learning_rate == 0.08


# ── metrics tests ─────────────────────────────────────────────────────────────

def test_mrr_at_k_perfect():
    y_true = np.array([0, 1, 2])
    proba = np.eye(3)
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(1.0)


def test_mrr_at_k_second_place():
    y_true = np.array([0])
    proba = np.array([[0.4, 0.6]])
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.5)


def test_mrr_at_k_not_in_topk():
    y_true = np.array([2])
    proba = np.array([[0.6, 0.3, 0.1]])
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.0)


def test_mrr_at_k_mixed():
    y_true = np.array([0, 2])
    proba = np.array([[0.9, 0.05, 0.05], [0.5, 0.1, 0.4]])
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(0.75)


# ── ONNX tests ────────────────────────────────────────────────────────────────

def test_export_creates_onnx_file(tmp_path):
    pipe, _, _ = _make_onnx_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_onnx_probas_match_sklearn(tmp_path):
    pipe, row, _ = _make_onnx_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))
    extra = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra if c in row.columns])
    proba_sklearn = pipe.predict_proba(row_clean)
    proba_onnx = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")
    np.testing.assert_allclose(proba_onnx, proba_sklearn, atol=1e-4)


def test_onnx_output_shape(tmp_path):
    pipe, row, n_classes = _make_onnx_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))
    proba = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")
    assert proba.shape == (1, n_classes)


def test_fast_preprocessor_matches_sklearn():
    pipe, row, _ = _make_onnx_pipeline_and_row()
    ct = pipe["pre"]
    fp = FastPreprocessor(ct)
    extra = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra if c in row.columns])
    X_sklearn = ct.transform(row_clean).astype(np.float32)
    row_dict = row.iloc[0].to_dict()
    X_fast = fp.encode(row_dict)
    np.testing.assert_array_equal(X_sklearn, X_fast)


def test_fast_preprocessor_unknown_category():
    pipe, row, _ = _make_onnx_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])
    row_dict = row.iloc[0].to_dict()
    row_dict["TypeParcoursId"] = "UNSEEN_VALUE_XYZ"
    X_fast = fp.encode(row_dict)
    type_idx = cat_cols.index("TypeParcoursId")
    assert X_fast[0, type_idx] == -1.0


def test_fast_preprocessor_nan_numeric_passthrough():
    pipe, row, _ = _make_onnx_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])
    num_cols = list(pipe["pre"].transformers_[1][2])
    row_dict = row.iloc[0].to_dict()
    row_dict["PrixParLiaison"] = float("nan")
    X_fast = fp.encode(row_dict)
    prix_idx = len(cat_cols) + num_cols.index("PrixParLiaison")
    assert np.isnan(X_fast[0, prix_idx])


def test_fast_preprocessor_pandas_na_numeric_passthrough():
    pipe, row, _ = _make_onnx_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])
    num_cols = list(pipe["pre"].transformers_[1][2])
    row_dict = row.iloc[0].to_dict()
    row_dict["PrixParLiaison"] = pd.NA
    X_fast = fp.encode(row_dict)
    prix_idx = len(cat_cols) + num_cols.index("PrixParLiaison")
    assert np.isnan(X_fast[0, prix_idx])


# ── retrain guardrail tests ───────────────────────────────────────────────────

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


def test_check_guardrail_passes_when_no_current():
    passes, reason = check_guardrail({}, {"hit_rate@1": 0.5})
    assert passes
    assert "waived" in reason


def test_check_guardrail_passes_small_drop():
    passes, _ = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.73})
    assert passes


def test_check_guardrail_passes_improvement():
    passes, _ = check_guardrail({"hit_rate@1": 0.70}, {"hit_rate@1": 0.76})
    assert passes


def test_check_guardrail_passes_at_exact_threshold():
    passes, _ = check_guardrail({"hit_rate@1": 0.80}, {"hit_rate@1": 0.75})
    assert passes


def test_check_guardrail_fails_above_threshold():
    passes, reason = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.69})
    assert not passes
    assert "BLOCKED" in reason


def test_check_guardrail_custom_threshold():
    passes, _ = check_guardrail({"hit_rate@1": 0.80}, {"hit_rate@1": 0.75}, threshold=0.10)
    assert passes


def test_evaluate_model_returns_correct_keys():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment")
    metrics = evaluate_model(arts, df)
    assert set(metrics.keys()) == {"hit_rate@1", "hit_rate@3", "mrr@3", "test_rows"}


def test_evaluate_model_metrics_in_range():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment")
    metrics = evaluate_model(arts, df)
    assert 0.0 <= metrics["hit_rate@1"] <= 1.0
    assert 0.0 <= metrics["hit_rate@3"] <= 1.0
    assert 0.0 <= metrics["mrr@3"] <= 1.0
    assert metrics["test_rows"] > 0


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


def test_retrain_pipeline_promotes_on_first_run(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(paths, features_df=_make_mini_features(), clean_df=_make_mini_clean())
    assert report["guardrail_passes"] is True
    assert report["promoted"] is True
    assert report["challenger_updated"] is True
    assert report["dry_run"] is False
    assert (paths.models_dir / "xgb_ranker.json").exists()
    assert (paths.models_dir / "label_encoder.joblib").exists()
    assert (paths.models_dir / "xgb_ranker.onnx").exists()
    assert (paths.models_dir / "cold_start.joblib").exists()
    assert (paths.models_dir / "xgb_ranker_challenger.json").exists()
    assert (paths.models_dir / "label_encoder_challenger.joblib").exists()
    assert (paths.models_dir / "xgb_ranker_challenger.onnx").exists()


def test_retrain_pipeline_dry_run_does_not_promote(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(paths, dry_run=True, features_df=_make_mini_features(), clean_df=_make_mini_clean())
    assert report["dry_run"] is True
    assert report["promoted"] is False
    assert report["challenger_updated"] is False
    assert not (paths.models_dir / "xgb_ranker.json").exists()
    assert not (paths.models_dir / "xgb_ranker_challenger.json").exists()


def test_retrain_pipeline_blocked_by_guardrail(tmp_path):
    paths = _make_test_paths(tmp_path)
    meta = {"metrics": {"hit_rate@1": 0.999, "hit_rate@3": 0.999, "mrr@3": 0.999}}
    (paths.models_dir / "xgb_ranker.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report = retrain_pipeline(paths, features_df=_make_mini_features(), clean_df=_make_mini_clean())
    assert report["guardrail_passes"] is False
    assert report["promoted"] is False
    assert report["challenger_updated"] is True
    assert not (paths.models_dir / "xgb_ranker.json").exists()
    assert (paths.models_dir / "xgb_ranker_challenger.json").exists()
    assert (paths.models_dir / "xgb_ranker_challenger.onnx").exists()


def test_retrain_pipeline_rolling_window(tmp_path):
    paths = _make_test_paths(tmp_path)
    n = 400
    df = _make_mini_features(n=n)
    df["DateHeureDepartVoyageSegment"] = pd.date_range("2023-01-01", periods=n, freq="2D")
    report = retrain_pipeline(paths, window_months=12, features_df=df, clean_df=_make_mini_clean())
    assert report["window_months"] == 12
    assert report["guardrail_passes"] is True
    assert report["promoted"] is True
    assert report["train_rows"] < n


def test_write_challenger_artifacts(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (staging / "xgb_ranker.json").write_bytes(b"model")
    (staging / "label_encoder.joblib").write_bytes(b"le")
    (staging / "xgb_ranker.onnx").write_bytes(b"onnx")
    write_challenger_artifacts(staging, models)
    assert (models / "xgb_ranker_challenger.json").read_bytes() == b"model"
    assert (models / "label_encoder_challenger.joblib").read_bytes() == b"le"
    assert (models / "xgb_ranker_challenger.onnx").read_bytes() == b"onnx"
    assert not (models / "xgb_ranker.json").exists()


# ── popularity tests ──────────────────────────────────────────────────────────

def test_build_popularity_orders_by_descending_frequency():
    df = pd.DataFrame({"LiaisonId": ["A", "A", "A", "B", "B", "C"]})
    assert build_popularity_list(df) == ["A", "B", "C"]


def test_build_popularity_includes_every_liaison_as_str():
    df = pd.DataFrame({"LiaisonId": [10, 20, 30, 20]})
    result = build_popularity_list(df)
    assert set(result) == {"10", "20", "30"}
    assert all(isinstance(x, str) for x in result)


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "sub" / "popularity.joblib"
    save_popularity(["A", "B", "C"], p)
    assert load_popularity(p) == ["A", "B", "C"]
