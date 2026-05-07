from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.recommender import Recommender
from rec_oncf.training import TrainArtifacts


def _make_artifacts() -> TrainArtifacts:
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

    feature_df = df.drop(columns=["CodeClient"])
    cat_cols = [c for c in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[c])]
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
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
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(feature_df, y)
    return TrainArtifacts(pipeline=pipe, label_encoder=le)


def _make_clean_df() -> pd.DataFrame:
    """Clean df with all trip-context columns required for on-the-fly feature
    computation.
    """
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
def recommender():
    arts = _make_artifacts()
    clean = _make_clean_df()
    return Recommender.from_data(arts, clean)


def test_cold_start_unknown_user(recommender):
    result = recommender.recommend("9999", k=1)
    assert result["mode"] == "cold_start"
    assert result["recommendations"] == []


def test_warm_user_returns_model_mode(recommender):
    result = recommender.recommend("1001", k=1)
    assert result["mode"] == "model"
    assert len(result["recommendations"]) == 1


def test_k3_returns_up_to_3(recommender):
    result = recommender.recommend("1001", k=3)
    assert result["mode"] == "model"
    assert 1 <= len(result["recommendations"]) <= 3


def test_k_clamped_to_3(recommender):
    result = recommender.recommend("1001", k=10)
    assert len(result["recommendations"]) <= 3


def test_cold_start_few_trips_returns_cf_mode():
    """A user with 1-2 trips should get cold_start_cf, not an empty cold_start."""
    arts = _make_artifacts()
    dates = pd.date_range("2020-01-01", periods=2, freq="30D")
    clean = _make_clean_df()
    extra = pd.DataFrame({
        "CodeClient": [2001, 2001],
        "AchteurId":  [2001, 2001],
        "LiaisonId":  ["A", "B"],
        "DateHeureDepartVoyageSegment": list(dates),
    })
    for col in ["TypeParcoursId", "ClassificationId", "ClassePhysiqueId",
                "NiveauPrixId", "TrainAutocarId", "CarteClientId"]:
        extra[col] = 1
    extra["PrixParLiaison"] = 100.0
    extra["NbrVoySegment"] = 1.0
    extra["DelaiAnticipation"] = 5.0
    full_clean = pd.concat([clean, extra], ignore_index=True)

    rec = Recommender.from_data(arts, full_clean)
    result = rec.recommend("2001", k=1)
    assert result["mode"] == "cold_start_cf"
    assert len(result["recommendations"]) >= 1
