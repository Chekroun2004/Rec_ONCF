"""HTTP integration tests for apps/api/main.py.

We bypass the production lifespan (which loads the trained model from disk)
by injecting a lightweight Recommender into app.state directly, then driving
the endpoints with FastAPI's TestClient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from unittest.mock import patch
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.main import app
from rec_oncf.recommender import Recommender
from rec_oncf.training import TrainArtifacts


def _build_artifacts() -> TrainArtifacts:
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "CodeClient": rng.integers(1000, 9999, n),
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
        "user_trip_index": np.arange(n),
        "days_since_prev": rng.choice([np.nan, 7.0, 14.0], n),
        "depart_hour": rng.integers(0, 24, n),
        "depart_dow": rng.integers(0, 7, n),
        "depart_month": rng.integers(1, 13, n),
        "depart_hour_sin": rng.standard_normal(n),
        "depart_hour_cos": rng.standard_normal(n),
        "depart_dow_sin": rng.standard_normal(n),
        "depart_dow_cos": rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n),
    })
    y_raw = rng.choice(["A", "B", "C"], n)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    feat_df = df.drop(columns=["CodeClient"])
    cat_cols = [c for c in feat_df.columns if not pd.api.types.is_numeric_dtype(feat_df[c])]
    num_cols = [c for c in feat_df.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=5,
        tree_method="hist", device="cpu", eval_metric="mlogloss",
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(feat_df, y)
    return TrainArtifacts(pipeline=pipe, label_encoder=le)


def _build_clean_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="30D")
    df = pd.DataFrame({
        "CodeClient": [1001] * 5 + [1002] * 5,
        "AchteurId":  [1001] * 5 + [1002] * 5,
        "LiaisonId":  ["A", "B", "A", "B", "A", "C", "C", "C", "C", "C"],
        "DateHeureDepartVoyageSegment": list(dates) + list(dates),
    })
    for col in ["TypeParcoursId", "ClassificationId", "ClassePhysiqueId",
                "NiveauPrixId", "TrainAutocarId", "CarteClientId"]:
        df[col] = 1
    df["PrixParLiaison"] = 100.0
    df["NbrVoySegment"] = 1.0
    df["DelaiAnticipation"] = 5.0
    return df


@pytest.fixture(scope="module")
def client():
    arts = _build_artifacts()
    clean = _build_clean_df()
    app.state.recommender = Recommender.from_data(arts, clean)
    app.state.liaison_map = {}   # empty — schedule calls return [] in unit tests
    app.state.redis = None
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_model_mode_for_known_user(client):
    payload = {"code_client": "1001", "k": 3}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "model"
    assert isinstance(body["recommendations"], list)
    assert 1 <= len(body["recommendations"]) <= 3


def test_recommend_returns_cold_start_for_unknown_user(client):
    payload = {"code_client": "00000", "k": 3}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "cold_start"
    assert body["recommendations"] == []


def test_recommend_validates_k_upper_bound(client):
    payload = {"code_client": "1001", "k": 99}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_recommend_validates_k_lower_bound(client):
    payload = {"code_client": "1001", "k": 0}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_recommend_no_schedules_field_by_default(client):
    resp = client.post("/recommend", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    assert "schedules" not in resp.json()


def test_recommend_include_schedule_adds_schedules_field(client):
    mock_sched = [{"depart": "07:00", "arrive": "09:30", "train": "1234"}]
    with patch("apps.api.main.get_schedule", return_value=mock_sched):
        resp = client.post(
            "/recommend",
            json={"code_client": "1001", "k": 1, "include_schedule": True},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "schedules" in body
    assert isinstance(body["schedules"], dict)
    for lid, deps in body["schedules"].items():
        assert isinstance(lid, str)
        assert isinstance(deps, list)


def test_recommend_include_schedule_unknown_user_no_schedules(client):
    resp = client.post(
        "/recommend",
        json={"code_client": "unknown_xyz", "k": 1, "include_schedule": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Unknown user -> cold_start -> empty recommendations -> schedules key absent
    assert body["recommendations"] == []
    assert "schedules" not in body
