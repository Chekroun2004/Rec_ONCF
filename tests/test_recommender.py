from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.candidates import generate_candidates
from rec_oncf.cold_start import (
    build_cold_start_recommender,
    load_cold_start,
    save_cold_start,
)
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


def test_unknown_user_falls_back_to_popularity():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean, popularity=["C", "A", "B"])
    result = rec.recommend("9999", k=2)
    assert result["mode"] == "popularity"
    recs = result["recommendations"]
    assert len(recs) == 2
    assert set(recs) <= {"C", "A", "B"}  # sampled from popularity pool
    assert len(set(recs)) == 2            # no duplicates


def test_unknown_user_without_popularity_is_cold_start():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean)  # no popularity
    result = rec.recommend("9999", k=2)
    assert result["mode"] == "cold_start"
    assert result["recommendations"] == []


def test_recommend_result_always_has_labels_key():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean, popularity=["A", "B"])
    for cc in ("1001", "9999"):
        result = rec.recommend(cc, k=2)
        assert "labels" in result
        assert isinstance(result["labels"], dict)


def test_recommend_labels_map_liaison_to_station_pair():
    arts = _make_artifacts()
    clean = _make_clean_df()
    clean["DesignationFrGareDepart"] = "GARE X"
    clean["DesignationFrGareArrive"] = "GARE Y"
    rec = Recommender.from_data(arts, clean, popularity=["A"])
    result = rec.recommend("1001", k=3)
    for lid in result["recommendations"]:
        assert result["labels"][lid] == "GARE X → GARE Y"


def test_model_pads_to_k_with_popularity():
    """User with only 2 distinct routes in history must still get k=3 recs when popularity is set.
    Model-ranked recs come first; popularity fills the remainder."""
    arts = _make_artifacts()
    # _make_clean_df: user 1001 has 5 trips but only routes A and B (2 distinct).
    # generate_candidates → [A, B]; model can rank at most 2 → pad to 3 with C from popularity.
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean, popularity=["C", "A", "B"])
    result = rec.recommend("1001", k=3)
    assert result["mode"] == "model"
    assert len(result["recommendations"]) == 3
    # The padded entry (C) must not appear before the model-ranked entries (A and B)
    model_routes = {"A", "B"}
    recs = result["recommendations"]
    # A and B should be in positions 0 and 1 (model-ranked), C at position 2 (padded)
    assert set(recs[:2]) == model_routes
    assert recs[2] == "C"


# ── candidates tests ──────────────────────────────────────────────────────────

def _make_history(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["DateHeureDepartVoyageSegment"] = pd.to_datetime(df["DateHeureDepartVoyageSegment"])
    return df


def test_returns_list_of_strings():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "100", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "200", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "100", "DateHeureDepartVoyageSegment": "2020-01-03"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)


def test_most_frequent_first():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-03"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-04"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert result[0] == "B"


def test_recency_tiebreaker():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-06-01"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert result[0] == "B"


def test_max_candidates_respected():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": str(i), "DateHeureDepartVoyageSegment": f"2020-01-{i:02d}"}
        for i in range(1, 21)
    ])
    result = generate_candidates(history, user_id="U1", max_candidates=5)
    assert len(result) <= 5


def test_unknown_user_returns_empty():
    history = _make_history([{"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"}])
    result = generate_candidates(history, user_id="UNKNOWN")
    assert result == []


def test_empty_history_returns_empty():
    history = pd.DataFrame(columns=["CodeClient", "LiaisonId", "DateHeureDepartVoyageSegment"])
    result = generate_candidates(history, user_id="U1")
    assert result == []


def test_no_duplicates():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-02"},
        {"CodeClient": "U1", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-03"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert len(result) == len(set(result))


def test_per_user_isolation():
    history = _make_history([
        {"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"},
        {"CodeClient": "U2", "LiaisonId": "B", "DateHeureDepartVoyageSegment": "2020-01-01"},
    ])
    result = generate_candidates(history, user_id="U1")
    assert "B" not in result
    assert "A" in result


def test_integer_user_id_coerced():
    history = _make_history([{"CodeClient": "42", "LiaisonId": "X", "DateHeureDepartVoyageSegment": "2020-01-01"}])
    result = generate_candidates(history, user_id=42)
    assert result == ["X"]


def test_empty_result_is_falsy():
    history = _make_history([{"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"}])
    result = generate_candidates(history, user_id="UNKNOWN")
    assert not result


def test_non_empty_result_is_truthy():
    history = _make_history([{"CodeClient": "U1", "LiaisonId": "A", "DateHeureDepartVoyageSegment": "2020-01-01"}])
    result = generate_candidates(history, user_id="U1")
    assert result


# ── cold_start tests ──────────────────────────────────────────────────────────

def _make_cold_start_clean() -> pd.DataFrame:
    rows = [("A", "X"), ("A", "Y"), ("A", "Z"), ("B", "X"), ("B", "Y"), ("C", "Z"), ("C", "W")]
    df = pd.DataFrame(rows, columns=["CodeClient", "LiaisonId"])
    df["DateHeureDepartVoyageSegment"] = pd.date_range("2020-01-01", periods=len(df))
    return df


def test_global_top_order():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    assert len(rec.global_top) >= 1
    assert rec.global_top[0] in {"X", "Y", "Z"}


def test_global_top_excludes_nothing():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    assert "W" in rec.global_top


def test_cooccurrence_x_contains_y():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    assert "Y" in rec.cooccurrence.get("X", [])


def test_cooccurrence_z_contains_w():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    assert "W" in rec.cooccurrence.get("Z", [])


def test_cold_start_recommend_no_history_returns_global_top():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    result = rec.recommend(history_df=None, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_cold_start_recommend_with_history_uses_cooccurrence():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=3)
    assert "Y" in result or "Z" in result


def test_cold_start_recommend_k_respected():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=1)
    assert len(result) == 1


def test_cold_start_recommend_unknown_route_falls_back_to_global():
    rec = build_cold_start_recommender(_make_cold_start_clean())
    history = pd.DataFrame({"LiaisonId": ["UNKNOWN"]})
    result = rec.recommend(history_df=history, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_cold_start_save_load_roundtrip(tmp_path):
    rec = build_cold_start_recommender(_make_cold_start_clean())
    path = tmp_path / "cold_start.joblib"
    save_cold_start(rec, path)
    loaded = load_cold_start(path)
    assert loaded.global_top == rec.global_top
    assert loaded.cooccurrence == rec.cooccurrence
