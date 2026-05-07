from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class ColdStartRecommender:
    cooccurrence: dict[str, list[str]]
    global_top: list[str]

    def recommend(self, history_df: pd.DataFrame | None, k: int) -> list[str]:
        if history_df is None or history_df.empty:
            return self.global_top[:k]

        known_routes = history_df["LiaisonId"].astype(str).unique().tolist()

        scores: Counter[str] = Counter()
        for route in known_routes:
            for co_route in self.cooccurrence.get(route, []):
                scores[co_route] += 1

        if not scores:
            return self.global_top[:k]

        return [r for r, _ in scores.most_common(k)]


def build_route_cooccurrence(clean_df: pd.DataFrame, top_n: int = 20) -> dict[str, list[str]]:
    user_routes = (
        clean_df.groupby("CodeClient")["LiaisonId"]
        .apply(lambda s: s.astype(str).unique().tolist())
    )
    cooc: dict[str, Counter[str]] = {}
    for routes in user_routes:
        for route in routes:
            if route not in cooc:
                cooc[route] = Counter()
            for other in routes:
                if other != route:
                    cooc[route][other] += 1

    return {
        route: [r for r, _ in counter.most_common(top_n)]
        for route, counter in cooc.items()
    }


def build_global_top(clean_df: pd.DataFrame, n: int = 20) -> list[str]:
    return (
        clean_df["LiaisonId"]
        .astype(str)
        .value_counts()
        .head(n)
        .index.tolist()
    )


def build_cold_start_recommender(clean_df: pd.DataFrame) -> ColdStartRecommender:
    return ColdStartRecommender(
        cooccurrence=build_route_cooccurrence(clean_df),
        global_top=build_global_top(clean_df),
    )


def save_cold_start(rec: ColdStartRecommender, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rec, path)


def load_cold_start(path: Path) -> ColdStartRecommender:
    return joblib.load(Path(path))
