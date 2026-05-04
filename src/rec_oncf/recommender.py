from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from rec_oncf.candidates import generate_candidates
from rec_oncf.config import Paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import TrainArtifacts, load_artifacts, predict_proba

_COLD_START = {"mode": "cold_start", "recommendations": []}


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    Features for the next-trip prediction are computed ON THE FLY from the
    user's live history, so they always reflect the most recent bookings ---
    no stale snapshots from training time.
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]

    @classmethod
    def from_paths(cls, paths: Paths) -> Recommender:
        artifacts = load_artifacts(
            model_path=paths.xgb_model_path,
            label_encoder_path=paths.label_encoder_path,
        )
        clean = read_parquet(paths.processed_dataset_parquet)
        return cls.from_data(artifacts, clean)

    @classmethod
    def from_data(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,  # kept for backward compat, unused
    ) -> Recommender:
        history_lookup = {
            str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
            for cid, grp in clean_df.groupby("CodeClient")
        }
        return cls(artifacts=artifacts, history_lookup=history_lookup)

    def recommend(
        self,
        code_client: str,
        k: int = 1,
        *,
        asof: datetime | pd.Timestamp | None = None,
    ) -> dict:
        code_client = str(code_client)
        k = min(max(k, 1), 3)

        history = self.history_lookup.get(code_client)
        if history is None or len(history) < 3:
            return _COLD_START

        candidates = generate_candidates(
            history, user_id=code_client, max_candidates=10
        )
        if not candidates:
            return _COLD_START

        # Compute features on the fly from live history (no stale snapshot).
        feat_row = compute_inference_row(history, asof=asof)

        # Two-stage ranking: score the model, then KEEP ONLY the candidates
        # produced by generate_candidates() and rank them by probability.
        le = self.artifacts.label_encoder
        known = set(le.classes_)
        valid_candidates = [c for c in candidates if c in known]

        if not valid_candidates:
            # No candidate is known to the model -> fall back on the raw
            # heuristic order produced by generate_candidates.
            return {"mode": "model", "recommendations": candidates[:k]}

        proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]
        cand_idx = le.transform(np.asarray(valid_candidates))
        cand_scores = proba[cand_idx]

        order = np.argsort(-cand_scores)[:k]
        recs = [valid_candidates[i] for i in order]
        return {"mode": "model", "recommendations": recs}
