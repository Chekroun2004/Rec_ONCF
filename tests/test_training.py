from __future__ import annotations

import numpy as np
import pandas as pd
from rec_oncf.training import predict_proba, train_xgb_multiclass


def _tiny_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "LiaisonId": rng.choice(["A", "B", "C"], n),
        "DateHeureDepartVoyageSegment": pd.date_range("2020-01-01", periods=n, freq="D"),
        "TypeParcoursId":   rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":     rng.choice(["1", "2"], n),
        "TrainAutocarId":   rng.choice(["10", "20"], n),
        "CarteClientId":    rng.choice(["0", "1"], n),
        "prev_liaison":     rng.choice(["A", "B", "nan"], n),
        "TrajetAllerRetour": rng.choice(["AR", "AS"], n),
        "PrixParLiaison":   rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":    np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index":  np.arange(n),
        "days_since_prev":  rng.choice([np.nan, 7.0, 14.0], n),
        "depart_hour":      rng.integers(0, 24, n),
        "depart_dow":       rng.integers(0, 7, n),
        "depart_month":     rng.integers(1, 13, n),
        "depart_hour_sin":  rng.standard_normal(n),
        "depart_hour_cos":  rng.standard_normal(n),
        "depart_dow_sin":   rng.standard_normal(n),
        "depart_dow_cos":   rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n),
    })


def test_train_handles_nulls_and_returns_proba():
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    proba = predict_proba(arts, df, label_col="LiaisonId")
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_artifacts_have_correct_classes():
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    assert set(arts.label_encoder.classes_) == {"A", "B", "C"}


def test_train_xgb_accepts_hyperparam_overrides():
    """Simulation/baseline must train with challenger hyperparams, passed explicitly."""
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df,
        label_col="LiaisonId",
        time_col="DateHeureDepartVoyageSegment",
        n_estimators=10,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.5,
    )
    clf = arts.pipeline.named_steps["clf"]
    assert clf.n_estimators == 10
    assert clf.max_depth == 3
    assert clf.learning_rate == 0.06
    assert clf.subsample == 0.85
    assert clf.colsample_bytree == 0.75
    assert clf.reg_lambda == 1.5


def test_train_xgb_defaults_unchanged():
    """Defaults must stay Sprint-2 so script 03 behavior is untouched."""
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    clf = arts.pipeline.named_steps["clf"]
    assert clf.n_estimators == 200
    assert clf.max_depth == 6
    assert clf.learning_rate == 0.08
