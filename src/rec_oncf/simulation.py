"""Sliding-window daily-retrain simulation.

The whole universe (oncf 2018-2020 + test1 2021-2022) is cleaned and
feature-built ONCE; the phases below are pure date-filters on that table:

- baseline (Phase A): everything strictly before the first simulation day
  (= a model trained on 2018->2022 minus the last 7 days).
- daily test (Phase B): for each "today" D, a 365-day window ending at D;
  the model is evaluated honestly on D+1 with a history truncated to <= D.

No per-day re-cleaning happens here, so cold-start / cancellation / per-user
feature chaining stay globally correct (see project_test1_purpose).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

TIME_COL = "DateHeureDepartVoyageSegment"


def last_n_dates(
    df: pd.DataFrame, *, n: int, min_count: int = 1, time_col: str = TIME_COL
) -> list[str]:
    """The last n calendar dates (YYYY-MM-DD) that have at least `min_count` rows.

    `min_count` skips sparse trailing days. test1 is dense through ~Dec 2021 then
    trails off into single-booking advance reservations up to Mar 2022; the
    simulation uses min_count to land on real, high-volume days.
    """
    counts = pd.to_datetime(df[time_col]).dt.date.astype(str).value_counts()
    dense = sorted(d for d, c in counts.items() if c >= min_count)
    if len(dense) < n:
        raise ValueError(
            f"Need at least {n} dates with >= {min_count} rows; got {len(dense)}"
        )
    return dense[-n:]


def baseline_frame(
    df: pd.DataFrame, *, sim_dates: list[str], time_col: str = TIME_COL
) -> pd.DataFrame:
    """Rows strictly before the first simulation day (training universe minus sim days)."""
    first = pd.Timestamp(min(sim_dates)).normalize()
    return df.loc[df[time_col] < first].reset_index(drop=True)


def filter_sliding_window(
    df: pd.DataFrame,
    *,
    end_date: str | pd.Timestamp,
    window_days: int = 365,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Rows whose time_col falls in [end_date - (window_days-1), end_date], full day inclusive."""
    end_ts = pd.Timestamp(end_date).normalize()
    end_inclusive = end_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    start_ts = end_ts - pd.Timedelta(days=window_days - 1)
    mask = (df[time_col] >= start_ts) & (df[time_col] <= end_inclusive)
    return df.loc[mask].reset_index(drop=True)


def day_frame(
    df: pd.DataFrame, *, date: str | pd.Timestamp, time_col: str = TIME_COL
) -> pd.DataFrame:
    """All rows whose departure calendar date equals `date`."""
    d = pd.Timestamp(date).normalize()
    mask = df[time_col].dt.normalize() == d
    return df.loc[mask].reset_index(drop=True)


def history_through(
    df: pd.DataFrame, *, date: str | pd.Timestamp, time_col: str = TIME_COL
) -> pd.DataFrame:
    """All rows up to and including `date` (used to build a leak-free recommender)."""
    end_inclusive = pd.Timestamp(date).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return df.loc[df[time_col] <= end_inclusive].reset_index(drop=True)


def eval_on_next_day(recommender: Any, next_day_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate recommender on the true bookings of the next day.

    For each row, recommender.recommend(code_client, k=3) is compared against the
    row's real LiaisonId. Returns {"hr@1", "hr@3", "mrr@3", "n_eval"}.
    """
    n = len(next_day_df)
    if n == 0:
        return {"hr@1": 0.0, "hr@3": 0.0, "mrr@3": 0.0, "n_eval": 0}

    hits1 = hits3 = 0
    rr_sum = 0.0
    for _, row in next_day_df.iterrows():
        true_lid = str(row["LiaisonId"])
        recs = [str(r) for r in recommender.recommend(str(row["CodeClient"]), k=3).get("recommendations", [])]
        if recs and recs[0] == true_lid:
            hits1 += 1
        if true_lid in recs[:3]:
            hits3 += 1
            rr_sum += 1.0 / (recs.index(true_lid) + 1)
    return {"hr@1": hits1 / n, "hr@3": hits3 / n, "mrr@3": rr_sum / n, "n_eval": n}


def load_simulation_log(path: str | Path) -> list[dict[str, Any]]:
    """Load the cumulative log; [] if the file is missing or empty."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text) if text else []


def log_simulation_entry(path: str | Path, entry: dict[str, Any]) -> None:
    """Append entry, or replace the existing one with the same 'day' (idempotent re-runs)."""
    path = Path(path)
    log = [e for e in load_simulation_log(path) if e.get("day") != entry.get("day")]
    log.append(entry)
    log.sort(key=lambda e: e.get("day", 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
