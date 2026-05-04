from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rec_oncf.features import build_training_rows, compute_inference_row


def _minimal_clean_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "CodeClient": [1001, 1001, 1001, 1002, 1002, 1002],
        "AchteurId":  [1001, 1001, 9999, 1002, 1002, 1002],
        "LiaisonId": ["10", "20", "10", "30", "30", "10"],
        "TrajetAllerRetour": [101, 102, 103, 201, 202, 203],
        "DateHeureDepartVoyageSegment": pd.date_range("2020-01-01", periods=n, freq="30D"),
        "TypeParcoursId": [1, 1, 1, 2, 2, 2],
        "ClassificationId": [1, 1, 1, 1, 1, 1],
        "ClassePhysiqueId": [2, 2, 2, 1, 1, 1],
        "NiveauPrixId": [1, 1, 1, 2, 2, 2],
        "TrainAutocarId": [10, 10, 10, 20, 20, 20],
        "CarteClientId": [0, 0, 0, 1, 1, 1],
        "PrixParLiaison": [100.0, 200.0, np.nan, 150.0, 150.0, 100.0],
        "NbrVoySegment": [1, 1, 1, 2, 2, 1],
        "DelaiAnticipation": [5, 10, 3, 7, 2, 14],
        "depart_hour": [8, 18, 8, 7, 7, 9],
        "depart_dow": [0, 4, 0, 1, 1, 3],
        "depart_month": [1, 2, 3, 1, 2, 3],
        "DatePaiement": pd.date_range("2019-12-25", periods=n, freq="30D"),
    })


def test_is_self_purchase_column_exists():
    df = build_training_rows(_minimal_clean_df())
    assert "is_self_purchase" in df.columns


def test_is_self_purchase_values():
    df = build_training_rows(_minimal_clean_df())
    c1 = df[df["CodeClient"] == 1001].sort_values("DateHeureDepartVoyageSegment")  # int64 after build
    vals = c1["is_self_purchase"].tolist()
    assert vals[0] == 1
    assert vals[1] == 1
    assert vals[2] == 0


def test_trajet_aller_retour_not_a_feature():
    df = build_training_rows(_minimal_clean_df())
    assert "TrajetAllerRetour" not in df.columns


def test_codeclient_is_numeric():
    df = build_training_rows(_minimal_clean_df())
    assert pd.api.types.is_integer_dtype(df["CodeClient"])


def test_output_has_26_columns():
    df = build_training_rows(_minimal_clean_df())
    assert len(df.columns) == 26


def test_user_top_liaison_share_present_and_in_unit_interval():
    df = build_training_rows(_minimal_clean_df())
    assert "user_top_liaison_share" in df.columns
    valid = df["user_top_liaison_share"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


# --- compute_inference_row tests --------------------------------------------

def _history_for_user(code_client: int = 1001) -> pd.DataFrame:
    base = _minimal_clean_df()
    return base[base["CodeClient"] == code_client].reset_index(drop=True)


def test_inference_row_has_single_row():
    history = _history_for_user()
    row = compute_inference_row(history, asof=pd.Timestamp("2020-04-01"))
    assert len(row) == 1


def test_inference_row_prev_liaison_is_last_seen():
    history = _history_for_user()  # liaisons 10, 20, 10 in chronological order
    row = compute_inference_row(history, asof=pd.Timestamp("2020-04-01"))
    assert row["prev_liaison"].iloc[0] == "10"


def test_inference_row_user_trip_index_equals_history_size():
    history = _history_for_user()
    row = compute_inference_row(history, asof=pd.Timestamp("2020-04-01"))
    assert row["user_trip_index"].iloc[0] == float(len(history))


def test_inference_row_days_since_prev_is_positive():
    history = _history_for_user()
    row = compute_inference_row(history, asof=pd.Timestamp("2020-04-01"))
    assert row["days_since_prev"].iloc[0] >= 0.0


def test_inference_row_temporal_features_match_asof():
    history = _history_for_user()
    asof = pd.Timestamp("2020-06-15 14:30:00")  # Monday in June, hour 14
    row = compute_inference_row(history, asof=asof)
    assert row["depart_hour"].iloc[0] == 14.0
    assert row["depart_month"].iloc[0] == 6.0
    # Cyclic encoding sanity check (cos^2 + sin^2 = 1)
    s = row["depart_hour_sin"].iloc[0]
    c = row["depart_hour_cos"].iloc[0]
    np.testing.assert_allclose(s * s + c * c, 1.0, atol=1e-9)


def test_inference_row_raises_on_empty_history():
    empty = _minimal_clean_df().iloc[0:0]
    with pytest.raises(ValueError):
        compute_inference_row(empty)
