"""Pipeline integration tests: cleaning, features, extra-CSV concat, retrain data contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_oncf.cleaning import MIN_TRIPS_FOR_TRAINING, make_clean_dataset
from rec_oncf.features import build_training_rows, compute_inference_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── helpers ──────────────────────────────────────────────────────────────────

def _liaison_lookup(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "LiaisonId": [str(i) for i in ids],
        "DesignationFrGareDepart": [f"Gare_D_{i}" for i in ids],
        "DesignationFrGareArrive": [f"Gare_A_{i}" for i in ids],
    })


def _row(
    *,
    code_client: int,
    liaison_id: int,
    date: str,
    nbr: int = 1,
    prix: float = 100.0,
    delai: int = 5,
    achteur: int | None = None,
    ar: int = 1,
) -> dict:
    return {
        "TrajetAllerRetour": ar,
        "TypeParcoursId": 1,
        "CodeClient": code_client,
        "AchteurId": code_client if achteur is None else achteur,
        "ClassificationId": 1,
        "ClassePhysiqueId": 2,
        "NiveauPrixId": 1,
        "TrainAutocarId": 10,
        "LiaisonVoyageurSegmentIdSTG": liaison_id,
        "CarteClientId": 0,
        "PrixParLiaison": prix,
        "NbrVoySegment": nbr,
        "DatePaiement": date,
        "DateHeureDepartVoyageSegment": date,
        "DelaiAnticipation": delai,
    }


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


def _load_make_dataset():
    spec = importlib.util.spec_from_file_location(
        "make_dataset", PROJECT_ROOT / "scripts" / "01_make_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ── cleaning tests ────────────────────────────────────────────────────────────

def test_cancellation_propagates_to_previous_booking():
    raw = pd.DataFrame([
        _row(code_client=1, liaison_id=10, date="2020-01-01", ar=1),
        _row(code_client=1, liaison_id=10, date="2020-01-02", nbr=0, ar=1),
        _row(code_client=1, liaison_id=20, date="2020-01-03", ar=1),
        _row(code_client=1, liaison_id=30, date="2020-01-04", ar=1),
        _row(code_client=1, liaison_id=40, date="2020-01-05", ar=1),
        _row(code_client=1, liaison_id=50, date="2020-01-06", ar=1),
    ])
    clean, report, _ = make_clean_dataset(raw, _liaison_lookup([10, 20, 30, 40, 50]))
    remaining_dates = clean["DateHeureDepartVoyageSegment"].dt.strftime("%Y-%m-%d").tolist()
    assert "2020-01-01" not in remaining_dates
    assert "2020-01-02" not in remaining_dates
    assert "2020-01-03" in remaining_dates
    assert "2020-01-04" in remaining_dates
    assert report["rows_removed_negative"] >= 2


def test_cold_start_clients_are_removed():
    rows: list[dict] = []
    rows.append(_row(code_client=1, liaison_id=10, date="2020-01-01"))
    rows.append(_row(code_client=1, liaison_id=20, date="2020-02-01"))
    for i in range(MIN_TRIPS_FOR_TRAINING):
        rows.append(_row(code_client=2, liaison_id=10, date=f"2020-0{i + 1}-15"))
    clean, report, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10, 20]))
    remaining_clients = set(clean["CodeClient"].unique())
    assert 1 not in remaining_clients
    assert 2 in remaining_clients
    assert report["clients_removed_cold_start"] >= 1


def test_liaison_join_overlap_rate_reported():
    rows = [_row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-01") for i in range(MIN_TRIPS_FOR_TRAINING)]
    clean, report, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))
    assert report["join_status"] == "merged"
    assert report["join_overlap_rate"] == 1.0
    assert "DesignationFrGareDepart" in clean.columns
    assert "DesignationFrGareArrive" in clean.columns


def test_essential_missing_rows_dropped():
    rows = [_row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-01") for i in range(MIN_TRIPS_FOR_TRAINING + 2)]
    raw = pd.DataFrame(rows)
    raw.loc[len(raw)] = _row(code_client=1, liaison_id=10, date="not-a-date")
    clean, report, _ = make_clean_dataset(raw, _liaison_lookup([10]))
    assert clean["DateHeureDepartVoyageSegment"].notna().all()


def test_cyclic_features_are_added_and_within_unit_circle():
    rows = [_row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-15") for i in range(MIN_TRIPS_FOR_TRAINING + 1)]
    clean, _, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))
    for prefix in ("depart_hour", "depart_dow", "depart_month"):
        sin_col, cos_col = f"{prefix}_sin", f"{prefix}_cos"
        assert sin_col in clean.columns
        assert cos_col in clean.columns
        norm = clean[sin_col] ** 2 + clean[cos_col] ** 2
        np.testing.assert_allclose(norm.to_numpy(), 1.0, atol=1e-9)


# ── features tests ────────────────────────────────────────────────────────────

def test_is_self_purchase_column_exists():
    df = build_training_rows(_minimal_clean_df())
    assert "is_self_purchase" in df.columns


def test_is_self_purchase_values():
    df = build_training_rows(_minimal_clean_df())
    c1 = df[df["CodeClient"] == 1001].sort_values("DateHeureDepartVoyageSegment")
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


def _straddling_clean_df() -> pd.DataFrame:
    old = _minimal_clean_df().iloc[0:3].copy()
    new = _minimal_clean_df().iloc[0:2].copy()
    new["DateHeureDepartVoyageSegment"] = pd.to_datetime(["2021-06-01", "2021-07-01"])
    new["DatePaiement"] = pd.to_datetime(["2021-05-25", "2021-06-25"])
    return pd.concat([old, new], ignore_index=True)


def test_user_trip_index_chains_across_appended_corpora():
    df = build_training_rows(_straddling_clean_df())
    seq = df[df["CodeClient"] == 1001].sort_values("DateHeureDepartVoyageSegment")["user_trip_index"].tolist()
    assert seq == [0, 1, 2, 3, 4]


def _history_for_user(code_client: int = 1001) -> pd.DataFrame:
    base = _minimal_clean_df()
    return base[base["CodeClient"] == code_client].reset_index(drop=True)


def test_inference_row_has_single_row():
    row = compute_inference_row(_history_for_user(), asof=pd.Timestamp("2020-04-01"))
    assert len(row) == 1


def test_inference_row_prev_liaison_is_last_seen():
    row = compute_inference_row(_history_for_user(), asof=pd.Timestamp("2020-04-01"))
    assert row["prev_liaison"].iloc[0] == "10"


def test_inference_row_user_trip_index_equals_history_size():
    history = _history_for_user()
    row = compute_inference_row(history, asof=pd.Timestamp("2020-04-01"))
    assert row["user_trip_index"].iloc[0] == float(len(history))


def test_inference_row_days_since_prev_is_positive():
    row = compute_inference_row(_history_for_user(), asof=pd.Timestamp("2020-04-01"))
    assert row["days_since_prev"].iloc[0] >= 0.0


def test_inference_row_temporal_features_match_asof():
    asof = pd.Timestamp("2020-06-15 14:30:00")
    row = compute_inference_row(_history_for_user(), asof=asof)
    assert row["depart_hour"].iloc[0] == 14.0
    assert row["depart_month"].iloc[0] == 6.0
    s = row["depart_hour_sin"].iloc[0]
    c = row["depart_hour_cos"].iloc[0]
    np.testing.assert_allclose(s * s + c * c, 1.0, atol=1e-9)


def test_inference_row_raises_on_empty_history():
    empty = _minimal_clean_df().iloc[0:0]
    with pytest.raises(ValueError):
        compute_inference_row(empty)


# ── dataset extra-CSV concat tests ────────────────────────────────────────────

def test_concat_main_and_extra(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({"CodeClient": [1, 2], "AchteurId": [1, 2], "LiaisonVoyageurSegmentIdSTG": ["L1", "L2"]}), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({"CodeClient": [3], "AchteurId": [3], "LiaisonVoyageurSegmentIdSTG": ["L3"]}), tmp_path / "extra.csv")
    combined = mod.load_and_concat(main, extra)
    assert len(combined) == 3
    assert list(combined.columns) == ["CodeClient", "AchteurId", "LiaisonVoyageurSegmentIdSTG"]


def test_concat_no_extra_returns_main(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({"CodeClient": [1, 2], "AchteurId": [1, 2], "LiaisonVoyageurSegmentIdSTG": ["L1", "L2"]}), tmp_path / "main.csv")
    combined = mod.load_and_concat(main, None)
    assert len(combined) == 2


def test_concat_normalizes_aliased_buyer_column(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({"CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"]}), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({"CodeClient": [2], "Acheteurid": [2], "LiaisonVoyageurSegmentIdSTG": ["L2"]}), tmp_path / "extra.csv")
    combined = mod.load_and_concat(main, extra)
    assert len(combined) == 2
    assert "AchteurId" in combined.columns
    assert "Acheteurid" not in combined.columns
    assert combined["AchteurId"].notna().all()


def test_concat_parses_each_file_dates_with_its_own_convention(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({
        "CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"],
        "DatePaiement": ["3/25/2019"], "DateHeureDepartVoyageSegment": ["3/25/2019 08:00"],
    }), tmp_path / "main.csv")
    extra = _write(pd.DataFrame({
        "CodeClient": [2], "Acheteurid": [2], "LiaisonVoyageurSegmentIdSTG": ["L2"],
        "DatePaiement": ["25/3/2021"], "DateHeureDepartVoyageSegment": ["25/3/2021 08:00"],
    }), tmp_path / "extra.csv")
    combined = mod.load_and_concat(main, extra)
    dep = combined["DateHeureDepartVoyageSegment"]
    assert pd.api.types.is_datetime64_any_dtype(dep)
    assert dep.iloc[0] == pd.Timestamp("2019-03-25 08:00")
    assert dep.iloc[1] == pd.Timestamp("2021-03-25 08:00")


def test_concat_real_schema_mismatch_raises(tmp_path):
    mod = _load_make_dataset()
    main = _write(pd.DataFrame({"CodeClient": [1], "AchteurId": [1], "LiaisonVoyageurSegmentIdSTG": ["L1"]}), tmp_path / "main.csv")
    bad = _write(pd.DataFrame({"foo": [1], "bar": [2]}), tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="schema"):
        mod.load_and_concat(main, bad)


# ── retrain data contract tests ───────────────────────────────────────────────

def _oncf_row(*, code_client: int, liaison_id: int, date: str, i: int = 0, achteur: int | None = None) -> dict:
    return {
        "TrajetAllerRetour": 1 + (i % 2),
        "TypeParcoursId": 1 + (i % 2),
        "CodeClient": code_client,
        "AchteurId": code_client if achteur is None else achteur,
        "ClassificationId": 1 + (i % 3),
        "ClassePhysiqueId": 2 + (i % 2),
        "NiveauPrixId": 1 + (i % 3),
        "TrainAutocarId": 10 + (i % 2),
        "LiaisonVoyageurSegmentIdSTG": liaison_id,
        "CarteClientId": i % 2,
        "PrixParLiaison": 100.0 + i,
        "NbrVoySegment": 1 + (i % 2),
        "DatePaiement": date,
        "DateHeureDepartVoyageSegment": date,
        "DelaiAnticipation": 5 + (i % 4),
    }


def _sample_oncf_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(MIN_TRIPS_FOR_TRAINING + 2):
        rows.append(_oncf_row(code_client=1, liaison_id=10 + (i % 2) * 10, date=f"2020-0{i + 1}-15", i=i))
        rows.append(_oncf_row(code_client=2, liaison_id=10 + (i % 3) * 10, date=f"2020-0{i + 1}-20", i=i + 1, achteur=999))
    return pd.DataFrame(rows)


def _to_test1_layout(oncf_df: pd.DataFrame) -> pd.DataFrame:
    renamed = oncf_df.rename(columns={"AchteurId": "Acheteurid"})
    return renamed[list(reversed(renamed.columns))]


def test_cleaning_aliases_acheteurid_column():
    oncf = _sample_oncf_frame()
    test1 = oncf.rename(columns={"AchteurId": "Acheteurid"})
    assert "AchteurId" not in test1.columns
    clean, _, _ = make_clean_dataset(test1, _liaison_lookup([10, 20, 30]))
    assert "AchteurId" in clean.columns
    assert clean["AchteurId"].notna().all()


def test_features_identical_across_oncf_and_test1_layouts():
    oncf = _sample_oncf_frame()
    test1 = _to_test1_layout(oncf)
    liaison = _liaison_lookup([10, 20, 30])
    feat_oncf = build_training_rows(make_clean_dataset(oncf, liaison)[0])
    feat_test1 = build_training_rows(make_clean_dataset(test1, liaison)[0])
    assert list(feat_oncf.columns) == list(feat_test1.columns)
    assert feat_oncf.dtypes.equals(feat_test1.dtypes)
    pd.testing.assert_frame_equal(feat_oncf.reset_index(drop=True), feat_test1.reset_index(drop=True))


def test_feature_dtypes_are_stable_without_nan():
    oncf = _sample_oncf_frame()
    clean, _, _ = make_clean_dataset(oncf, _liaison_lookup([10, 20, 30]))
    feats = build_training_rows(clean)
    assert str(feats["NbrVoySegment"].dtype) == "float64"
    assert str(feats["DelaiAnticipation"].dtype) == "float64"
    assert str(feats["PrixParLiaison"].dtype) == "float64"
    assert str(feats["DateHeureDepartVoyageSegment"].dtype) == "datetime64[ns]"


def test_is_self_purchase_correct_when_achteurid_is_float64():
    oncf = _sample_oncf_frame().copy()
    oncf["AchteurId"] = oncf["AchteurId"].astype(float)
    liaison = _liaison_lookup([10, 20, 30])
    clean, _, _ = make_clean_dataset(oncf, liaison)
    feats = build_training_rows(clean)
    self_rows = feats[feats["CodeClient"] == 1]["is_self_purchase"]
    assert (self_rows == 1).all()


def test_is_self_purchase_survives_concat_dtype_promotion():
    liaison = _liaison_lookup([10, 20, 30])
    oncf_rows = [_oncf_row(code_client=100 + i, liaison_id=10 + (i % 3) * 10, date=f"2020-0{i+1}-15", i=i, achteur=999_000 + i) for i in range(MIN_TRIPS_FOR_TRAINING + 2)]
    oncf_raw = pd.DataFrame(oncf_rows)
    oncf_raw["AchteurId"] = oncf_raw["AchteurId"].astype(float)
    oncf_clean, _, _ = make_clean_dataset(oncf_raw, liaison)
    test1_rows = [_oncf_row(code_client=200 + i, liaison_id=10 + (i % 3) * 10, date=f"2021-0{i+1}-15", i=i) for i in range(MIN_TRIPS_FOR_TRAINING + 2)]
    test1_raw = pd.DataFrame(test1_rows).rename(columns={"AchteurId": "Acheteurid"})
    test1_clean, _, _ = make_clean_dataset(test1_raw, liaison)
    combined = pd.concat([oncf_clean, test1_clean], ignore_index=True)
    assert str(combined["AchteurId"].dtype) == "float64"
    feats = build_training_rows(combined)
    oncf_feats = feats[feats["CodeClient"].isin(range(100, 100 + MIN_TRIPS_FOR_TRAINING + 3))]
    test1_feats = feats[feats["CodeClient"].isin(range(200, 200 + MIN_TRIPS_FOR_TRAINING + 3))]
    assert (oncf_feats["is_self_purchase"] == 0).all()
    assert (test1_feats["is_self_purchase"] == 1).all()


def test_dmy_dates_are_parsed_day_first():
    dates = ["13/01/2021 07:30", "14/01/2021 17:30", "15/01/2021 08:00"]
    rows = [_oncf_row(code_client=1, liaison_id=10, date=d) for d in dates]
    clean, _, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))
    clean = clean.sort_values("DateHeureDepartVoyageSegment")
    assert clean["depart_hour"].tolist() == [7, 17, 8]
    assert clean["depart_month"].unique().tolist() == [1]
