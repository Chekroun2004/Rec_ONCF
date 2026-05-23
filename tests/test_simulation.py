from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from rec_oncf.extract_days import extract_last_n_days
from rec_oncf.simulation import (
    baseline_frame,
    day_frame,
    eval_on_next_day,
    filter_sliding_window,
    history_through,
    last_n_dates,
    load_simulation_log,
    log_simulation_entry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _two_year_df() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2020-01-01", "2021-12-31", freq="D"):
        for h in (8, 18):
            rows.append({
                "CodeClient": 100,
                "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=h),
                "LiaisonId": "L1",
            })
    return pd.DataFrame(rows)


def _ten_day_df() -> pd.DataFrame:
    rows = []
    for i, day in enumerate(pd.date_range("2021-12-22", "2021-12-31", freq="D")):
        rows.append({
            "CodeClient": 100 + i,
            "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=9),
            "LiaisonId": "L1",
        })
    return pd.DataFrame(rows)


def test_last_n_dates_returns_last_seven():
    dates = last_n_dates(_ten_day_df(), n=7)
    assert dates == [
        "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28",
        "2021-12-29", "2021-12-30", "2021-12-31",
    ]


def test_last_n_dates_skips_sparse_trailing_days():
    """The real test1 tail is sparse advance-bookings (1/day); dense days must win."""
    rows = []
    for day in pd.date_range("2021-12-10", "2021-12-12", freq="D"):  # 3 dense days
        for _ in range(5):
            rows.append({"CodeClient": 1, "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=8), "LiaisonId": "L1"})
    for day in pd.date_range("2022-02-01", "2022-02-02", freq="D"):  # 2 sparse trailing days
        rows.append({"CodeClient": 1, "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=8), "LiaisonId": "L1"})
    df = pd.DataFrame(rows)

    assert last_n_dates(df, n=2) == ["2022-02-01", "2022-02-02"]          # default keeps sparse
    assert last_n_dates(df, n=2, min_count=3) == ["2021-12-11", "2021-12-12"]  # density filter


def test_baseline_frame_excludes_sim_days():
    df = _two_year_df()
    sim_dates = last_n_dates(df, n=7)  # last 7 of 2021
    base = baseline_frame(df, sim_dates=sim_dates)
    assert base["DateHeureDepartVoyageSegment"].max() < pd.Timestamp(min(sim_dates))
    assert base["DateHeureDepartVoyageSegment"].dt.date.astype(str).max() == "2021-12-24"


def test_filter_sliding_window_keeps_one_year():
    window = filter_sliding_window(_two_year_df(), end_date="2021-12-31", window_days=365)
    assert window["DateHeureDepartVoyageSegment"].min() >= pd.Timestamp("2021-01-01")
    assert window["DateHeureDepartVoyageSegment"].max() <= pd.Timestamp("2021-12-31 23:59:59")


def test_filter_sliding_window_empty_when_out_of_range():
    window = filter_sliding_window(_two_year_df(), end_date="2019-01-01", window_days=365)
    assert len(window) == 0


def test_day_frame_returns_single_day():
    df = day_frame(_two_year_df(), date="2021-06-15")
    assert len(df) == 2
    assert (df["DateHeureDepartVoyageSegment"].dt.date.astype(str) == "2021-06-15").all()


def test_history_through_is_inclusive_and_leak_free():
    hist = history_through(_two_year_df(), date="2021-06-15")
    assert hist["DateHeureDepartVoyageSegment"].max() <= pd.Timestamp("2021-06-15 23:59:59")
    # the next day must NOT be present (no future leak)
    assert (hist["DateHeureDepartVoyageSegment"] < pd.Timestamp("2021-06-16")).all()


def test_eval_on_next_day_computes_metrics():
    class FakeRecommender:
        def recommend(self, code_client, k=3):
            return {"recommendations": ["L1", "L2", "L3"], "labels": {}}

    next_day = pd.DataFrame({"CodeClient": [100, 100], "LiaisonId": ["L1", "L4"]})
    m = eval_on_next_day(FakeRecommender(), next_day)
    assert m["n_eval"] == 2
    assert m["hr@1"] == 0.5   # L1 hits at rank 1, L4 misses
    assert m["hr@3"] == 0.5
    assert m["mrr@3"] == 0.5


def test_eval_on_next_day_empty():
    class FakeRecommender:
        def recommend(self, code_client, k=3):
            return {"recommendations": []}

    m = eval_on_next_day(FakeRecommender(), pd.DataFrame({"CodeClient": [], "LiaisonId": []}))
    assert m == {"hr@1": 0.0, "hr@3": 0.0, "mrr@3": 0.0, "n_eval": 0}


def test_log_appends_and_replaces_same_day(tmp_path):
    log_path = tmp_path / "simulation_daily.json"
    log_simulation_entry(log_path, {"day": 1, "hr@1": 0.70})
    log_simulation_entry(log_path, {"day": 2, "hr@1": 0.76})
    log_simulation_entry(log_path, {"day": 1, "hr@1": 0.78})  # re-run day 1
    log = load_simulation_log(log_path)
    assert [e["day"] for e in log] == [1, 2]
    assert log[0]["hr@1"] == 0.78  # replaced, not duplicated


# ── extract_days tests ────────────────────────────────────────────────────────

@pytest.fixture
def sample_days_df() -> pd.DataFrame:
    rows = []
    for day_idx, day in enumerate(pd.date_range("2021-12-22", "2021-12-31", freq="D")):
        for hour in (8, 12, 18):
            rows.append({
                "CodeClient": 1000 + day_idx,
                "DateHeureDepartVoyageSegment": day + pd.Timedelta(hours=hour),
                "LiaisonVoyageurSegmentIdSTG": f"L{day_idx % 4}",
                "PrixParLiaison": 100.0,
            })
    return pd.DataFrame(rows)


def test_extract_returns_base_and_seven_days(sample_days_df):
    base, days = extract_last_n_days(sample_days_df, n=7, date_col="DateHeureDepartVoyageSegment")
    assert isinstance(base, pd.DataFrame)
    assert isinstance(days, dict)
    assert len(days) == 7


def test_extract_keys_are_dates(sample_days_df):
    _, days = extract_last_n_days(sample_days_df, n=7, date_col="DateHeureDepartVoyageSegment")
    expected_dates = ["2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28", "2021-12-29", "2021-12-30", "2021-12-31"]
    assert sorted(days.keys()) == expected_dates


def test_extract_preserves_total_rows(sample_days_df):
    base, days = extract_last_n_days(sample_days_df, n=7, date_col="DateHeureDepartVoyageSegment")
    total = len(base) + sum(len(d) for d in days.values())
    assert total == len(sample_days_df)


def test_extract_base_has_only_early_dates(sample_days_df):
    base, _ = extract_last_n_days(sample_days_df, n=7, date_col="DateHeureDepartVoyageSegment")
    base_dates = base["DateHeureDepartVoyageSegment"].dt.date.unique()
    assert set(str(d) for d in base_dates) == {"2021-12-22", "2021-12-23", "2021-12-24"}


def test_extract_day_csv_has_consistent_rows(sample_days_df):
    _, days = extract_last_n_days(sample_days_df, n=7, date_col="DateHeureDepartVoyageSegment")
    for date_str, day_df in days.items():
        unique = day_df["DateHeureDepartVoyageSegment"].dt.date.astype(str).unique()
        assert list(unique) == [date_str]
