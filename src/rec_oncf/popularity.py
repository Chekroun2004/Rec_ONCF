from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def build_popularity_list(clean_df: pd.DataFrame) -> list[str]:
    """LiaisonId values ordered by descending global frequency.

    Cancellations are already removed upstream by cleaning.py, so a plain
    value_counts on the clean frame is the global popularity ranking.
    """
    counts = clean_df["LiaisonId"].astype(str).value_counts()
    return [str(x) for x in counts.index]


def save_popularity(popularity: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump([str(x) for x in popularity], path)


def load_popularity(path: Path) -> list[str]:
    return [str(x) for x in joblib.load(path)]
