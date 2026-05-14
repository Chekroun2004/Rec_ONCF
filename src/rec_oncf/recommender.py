from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from rec_oncf.candidates import generate_candidates
from rec_oncf.cold_start import ColdStartRecommender, build_cold_start_recommender, load_cold_start
from rec_oncf.config import Paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.popularity import load_popularity
from rec_oncf.training import FastPreprocessor, TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx

_LABEL_COLS = ("LiaisonId", "DesignationFrGareDepart", "DesignationFrGareArrive")


def _build_label_lookup(clean_df: pd.DataFrame) -> dict[str, str]:
    """LiaisonId -> "GARE DEPART → GARE ARRIVEE". Empty if columns absent."""
    if not set(_LABEL_COLS) <= set(clean_df.columns):
        return {}
    sub = (
        clean_df.loc[:, list(_LABEL_COLS)]
        .dropna(subset=["DesignationFrGareDepart", "DesignationFrGareArrive"])
        .drop_duplicates("LiaisonId")
    )
    return {
        str(dep_arr.LiaisonId): f"{dep_arr.DesignationFrGareDepart} → {dep_arr.DesignationFrGareArrive}"
        for dep_arr in sub.itertuples(index=False)
    }


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    For users with 1-2 trips, falls back to co-occurrence collaborative
    filtering. For warm users, features are computed ON THE FLY from live
    history and scored via ONNX Runtime (fast path) or sklearn (fallback).
    When the model cannot serve a user at all, falls back to global popularity
    (so the recommendation list is never empty), unless the popularity list is
    itself empty (then the legacy ``cold_start`` / empty-list result).
    Every result carries a ``labels`` dict mapping each recommended LiaisonId to
    a human-readable "GARE → GARE" string (missing ids are simply omitted).
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None        # onnxruntime.InferenceSession
    fast_preprocessor: FastPreprocessor | None = None
    popularity: list[str] = field(default_factory=list)
    liaison_label_lookup: dict[str, str] = field(default_factory=dict)

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
        fast_preprocessor = FastPreprocessor(artifacts.pipeline["pre"])
        popularity = (
            load_popularity(paths.popularity_path) if paths.popularity_path.exists() else []
        )
        return cls._build(
            artifacts, clean, cold_start_rec, onnx_session, fast_preprocessor, popularity
        )

    @classmethod
    def from_data(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,  # kept for backward compat, unused
        *,
        popularity: list[str] | None = None,
    ) -> Recommender:
        cold_start_rec = build_cold_start_recommender(clean_df)
        return cls._build(artifacts, clean_df, cold_start_rec, popularity=popularity)

    @classmethod
    def _build(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        cold_start_rec: ColdStartRecommender,
        onnx_session: object | None = None,
        fast_preprocessor: FastPreprocessor | None = None,
        popularity: list[str] | None = None,
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
            fast_preprocessor=fast_preprocessor,
            popularity=[str(x) for x in (popularity or [])],
            liaison_label_lookup=_build_label_lookup(clean_df),
        )

    # --- internals -------------------------------------------------------

    def _fallback(self, k: int) -> dict:
        if self.popularity:
            return {"mode": "popularity", "recommendations": list(self.popularity[:k])}
        return {"mode": "cold_start", "recommendations": []}

    def _labels_for(self, recs: list[str]) -> dict[str, str]:
        return {r: self.liaison_label_lookup[r] for r in recs if r in self.liaison_label_lookup}

    def _recommend_core(
        self,
        code_client: str,
        k: int,
        asof: datetime | pd.Timestamp | None,
    ) -> dict:
        history = self.history_lookup.get(code_client)
        if history is None:
            return self._fallback(k)

        if len(history) < 3:
            recs = self.cold_start_rec.recommend(history, k)
            if recs:
                return {"mode": "cold_start_cf", "recommendations": recs}
            return self._fallback(k)

        candidates = generate_candidates(history, user_id=code_client, max_candidates=10)
        if not candidates:
            return self._fallback(k)

        feat_row = compute_inference_row(history, asof=asof)

        le = self.artifacts.label_encoder
        known = set(le.classes_)
        valid_candidates = [c for c in candidates if c in known]
        if not valid_candidates:
            return {"mode": "model", "recommendations": candidates[:k]}

        if self.onnx_session is not None:
            if self.fast_preprocessor is not None:
                row_dict = feat_row.iloc[0].to_dict()
                X_pre = self.fast_preprocessor.encode(row_dict)
                proba = self.onnx_session.run(["probabilities"], {"input": X_pre})[0][0]
            else:
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
        if len(recs) < k and self.popularity:
            existing = set(recs)
            for lid in self.popularity:
                if lid not in existing:
                    recs.append(lid)
                    if len(recs) == k:
                        break
        return {"mode": "model", "recommendations": recs}

    # --- public API ------------------------------------------------------

    def recommend(
        self,
        code_client: str,
        k: int = 1,
        *,
        asof: datetime | pd.Timestamp | None = None,
    ) -> dict:
        code_client = str(code_client)
        k = min(max(k, 1), 3)
        result = self._recommend_core(code_client, k, asof)
        return {**result, "labels": self._labels_for(result["recommendations"])}
