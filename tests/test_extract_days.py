from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.extract_days import extract_last_n_days


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """30 rows over 10 distinct departure dates (3 rows/day), 2021-12-22 → 2021-12-31."""
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


def test_extract_returns_base_and_seven_days(sample_df):
    base, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    assert isinstance(base, pd.DataFrame)
    assert isinstance(days, dict)
    assert len(days) == 7


def test_extract_keys_are_dates(sample_df):
    _, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    expected_dates = [
        "2021-12-25", "2021-12-26", "2021-12-27", "2021-12-28",
        "2021-12-29", "2021-12-30", "2021-12-31",
    ]
    assert sorted(days.keys()) == expected_dates


def test_extract_preserves_total_rows(sample_df):
    base, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    total = len(base) + sum(len(d) for d in days.values())
    assert total == len(sample_df)


def test_extract_base_has_only_early_dates(sample_df):
    base, _ = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    base_dates = base["DateHeureDepartVoyageSegment"].dt.date.unique()
    assert set(str(d) for d in base_dates) == {"2021-12-22", "2021-12-23", "2021-12-24"}


def test_extract_day_csv_has_consistent_rows(sample_df):
    _, days = extract_last_n_days(sample_df, n=7, date_col="DateHeureDepartVoyageSegment")
    for date_str, day_df in days.items():
        unique = day_df["DateHeureDepartVoyageSegment"].dt.date.astype(str).unique()
        assert list(unique) == [date_str]
