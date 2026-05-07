from __future__ import annotations

import pandas as pd

from rec_oncf.cold_start import (
    build_cold_start_recommender,
    load_cold_start,
    save_cold_start,
)


def _make_clean() -> pd.DataFrame:
    rows = [
        ("A", "X"), ("A", "Y"), ("A", "Z"),
        ("B", "X"), ("B", "Y"),
        ("C", "Z"), ("C", "W"),
    ]
    df = pd.DataFrame(rows, columns=["CodeClient", "LiaisonId"])
    df["DateHeureDepartVoyageSegment"] = pd.date_range("2020-01-01", periods=len(df))
    return df


def test_global_top_order():
    rec = build_cold_start_recommender(_make_clean())
    assert len(rec.global_top) >= 1
    assert rec.global_top[0] in {"X", "Y", "Z"}


def test_global_top_excludes_nothing():
    rec = build_cold_start_recommender(_make_clean())
    assert "W" in rec.global_top


def test_cooccurrence_x_contains_y():
    rec = build_cold_start_recommender(_make_clean())
    assert "Y" in rec.cooccurrence.get("X", [])


def test_cooccurrence_z_contains_w():
    rec = build_cold_start_recommender(_make_clean())
    assert "W" in rec.cooccurrence.get("Z", [])


def test_recommend_no_history_returns_global_top():
    rec = build_cold_start_recommender(_make_clean())
    result = rec.recommend(history_df=None, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_recommend_with_history_uses_cooccurrence():
    rec = build_cold_start_recommender(_make_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=3)
    assert "Y" in result or "Z" in result


def test_recommend_k_respected():
    rec = build_cold_start_recommender(_make_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=1)
    assert len(result) == 1


def test_recommend_unknown_route_falls_back_to_global():
    rec = build_cold_start_recommender(_make_clean())
    history = pd.DataFrame({"LiaisonId": ["UNKNOWN"]})
    result = rec.recommend(history_df=history, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_save_load_roundtrip(tmp_path):
    rec = build_cold_start_recommender(_make_clean())
    path = tmp_path / "cold_start.joblib"
    save_cold_start(rec, path)
    loaded = load_cold_start(path)
    assert loaded.global_top == rec.global_top
    assert loaded.cooccurrence == rec.cooccurrence
