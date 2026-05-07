from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from rec_oncf.candidates import generate_candidates
from rec_oncf.cold_start import ColdStartRecommender, build_cold_start_recommender, load_cold_start
from rec_oncf.config import Paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx

_COLD_START = {"mode": "cold_start", "recommendations": []}


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    For users with 1-2 trips, falls back to co-occurrence collaborative
    filtering. For warm users, features are computed ON THE FLY from live
    history and scored via ONNX Runtime (fast path) or sklearn (fallback).
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None  # onnxruntime.InferenceSession

    @classmethod
    def from_paths(cls, paths: Paths) -> Recommender:
        from onnxruntime import InferenceSession
        artifacts = load_artifacts(
            model_path=paths.xgb_model_path,
            label_encoder_path=paths.label_encoder_path,
        )
        clean = read_parquet(paths.processed_dataset_parquet)
        cold_start_rec = load_cold_start(paths.cold_start_path)
        if not paths.onnx_model_path.exists():
            raise RuntimeError(
                f"ONNX model not found: {paths.onnx_model_path}. "
                "Run scripts/06_export_onnx.py first."
            )
        onnx_session = InferenceSession(str(paths.onnx_model_path))
        return cls._build(artifacts, clean, cold_start_rec, onnx_session)

    @classmethod
    def from_data(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,  # kept for backward compat, unused
    ) -> Recommender:
        cold_start_rec = build_cold_start_recommender(clean_df)
        return cls._build(artifacts, clean_df, cold_start_rec)

    @classmethod
    def _build(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        cold_start_rec: ColdStartRecommender,
        onnx_session: object | None = None,
    ) -> Recommender:
        history_lookup = {
            str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
            for cid, grp in clean_df.groupby("CodeClient")
        }
        return cls(
            artifacts=artifacts,
            history_lookup=history_lookup,
            cold_start_rec=cold_start_rec,
            onnx_session=onnx_session,
        )

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

        if history is None:
            return _COLD_START

        if len(history) < 3:
            recs = self.cold_start_rec.recommend(history, k)
            if recs:
                return {"mode": "cold_start_cf", "recommendations": recs}
            return _COLD_START

        candidates = generate_candidates(
            history, user_id=code_client, max_candidates=10
        )
        if not candidates:
            return _COLD_START

        feat_row = compute_inference_row(history, asof=asof)

        le = self.artifacts.label_encoder
        known = set(le.classes_)
        valid_candidates = [c for c in candidates if c in known]

        if not valid_candidates:
            return {"mode": "model", "recommendations": candidates[:k]}

        if self.onnx_session is not None:
            proba = predict_proba_onnx(
                self.onnx_session,
                self.artifacts.pipeline["pre"],
                feat_row,
                label_col="LiaisonId",
            )[0]
        else:
            proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]

        cand_idx = le.transform(np.asarray(valid_candidates))
        cand_scores = proba[cand_idx]

        order = np.argsort(-cand_scores)[:k]
        recs = [valid_candidates[i] for i in order]
        return {"mode": "model", "recommendations": recs}
