from __future__ import annotations

import pandas as pd


def extract_last_n_days(
    df: pd.DataFrame,
    *,
    n: int = 7,
    date_col: str = "DateHeureDepartVoyageSegment",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Splits df into a base DataFrame and a dict of the last n calendar days.

    Returns:
        base: all rows whose departure date is NOT among the last n dates
        days: dict mapping each of the last n dates (YYYY-MM-DD) to its rows

    The "day" is the calendar date of date_col (departure date). Rows with
    the same date are kept in their original order. The split is idempotent.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing column: {date_col}")

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"{int(parsed.isna().sum())} rows have unparseable {date_col} values"
        )

    df = df.copy()
    df[date_col] = parsed
    df["_day"] = parsed.dt.date.astype(str)

    unique_days = sorted(df["_day"].unique())
    if len(unique_days) < n:
        raise ValueError(
            f"Need at least {n} distinct days; got {len(unique_days)}"
        )

    last_days = unique_days[-n:]
    base = df[~df["_day"].isin(last_days)].drop(columns="_day").reset_index(drop=True)
    days = {
        day: df[df["_day"] == day].drop(columns="_day").reset_index(drop=True)
        for day in last_days
    }
    return base, days
