from __future__ import annotations

import pandas as pd


def generate_candidates(
    history_df: pd.DataFrame,
    *,
    user_id: str,
    max_candidates: int = 10,
    lookback: int = 50,
) -> list[str]:
    """Return up to `max_candidates` LiaisonId based on user's past bookings.

    Simple, strong baseline: most frequent + most recent in a lookback window.
    No GPS. No global popularity fallback here (cold-start rule is handled upstream).
    """

    user_hist = history_df[history_df["CodeClient"].astype(str) == str(user_id)]
    if user_hist.empty:
        return []

    user_hist = user_hist.sort_values("DateHeureDepartVoyageSegment").tail(lookback)
    # by frequency
    freq = (
        user_hist.groupby("LiaisonId")
        .size()
        .sort_values(ascending=False)
    )

    # by recency
    last_ts = user_hist.groupby("LiaisonId")["DateHeureDepartVoyageSegment"].max()
    score = pd.DataFrame({"freq": freq, "last": last_ts}).fillna(0)
    score = score.sort_values(["freq", "last"], ascending=[False, False])

    return score.index.astype(str).tolist()[:max_candidates]