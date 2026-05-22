"""Contract: any retrain data (test1.csv, future files) must yield features
structurally identical to oncf_data after going through the SAME pipeline.

test1.csv differs from oncf_data.csv only in:
  - column spelling: 'Acheteurid' (test1) vs 'AchteurId' (oncf_data)
  - column order
  - date convention: D/M/Y (test1) vs M/D/Y (oncf_data)

None of these may change the extracted features.
"""

from __future__ import annotations

import pandas as pd

from rec_oncf.cleaning import MIN_TRIPS_FOR_TRAINING, make_clean_dataset
from rec_oncf.features import build_training_rows


def _liaison_lookup(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "LiaisonId": [str(i) for i in ids],
        "DesignationFrGareDepart": [f"Gare_D_{i}" for i in ids],
        "DesignationFrGareArrive": [f"Gare_A_{i}" for i in ids],
    })


def _oncf_row(
    *,
    code_client: int,
    liaison_id: int,
    date: str,
    i: int = 0,
    achteur: int | None = None,
) -> dict:
    """A booking row in the canonical oncf_data layout (column 'AchteurId').

    Categorical/numeric fields vary with ``i`` so they are not dropped as
    constant columns (mirrors real oncf_data, which has varied values).
    """
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
    """Two clients, each above the cold-start threshold, varied routes/fields."""
    rows: list[dict] = []
    for i in range(MIN_TRIPS_FOR_TRAINING + 2):
        rows.append(_oncf_row(code_client=1, liaison_id=10 + (i % 2) * 10,
                              date=f"2020-0{i + 1}-15", i=i))
        rows.append(_oncf_row(code_client=2, liaison_id=10 + (i % 3) * 10,
                              date=f"2020-0{i + 1}-20", i=i + 1, achteur=999))
    return pd.DataFrame(rows)


def _to_test1_layout(oncf_df: pd.DataFrame) -> pd.DataFrame:
    """Mimic test1.csv: rename AchteurId -> Acheteurid and reverse column order."""
    renamed = oncf_df.rename(columns={"AchteurId": "Acheteurid"})
    return renamed[list(reversed(renamed.columns))]


def test_cleaning_aliases_acheteurid_column():
    """test1 spells the buyer column 'Acheteurid'; cleaning must alias it to
    'AchteurId' so is_self_purchase is real, not silently all-NA."""
    oncf = _sample_oncf_frame()
    test1 = oncf.rename(columns={"AchteurId": "Acheteurid"})
    assert "AchteurId" not in test1.columns  # only the test1 spelling present

    clean, _, _ = make_clean_dataset(test1, _liaison_lookup([10, 20, 30]))

    assert "AchteurId" in clean.columns
    assert clean["AchteurId"].notna().all(), "alias lost the buyer values"


def test_features_identical_across_oncf_and_test1_layouts():
    """The core contract: features from a test1-layout file must be byte-identical
    to features from the oncf_data layout (same columns, dtypes, values)."""
    oncf = _sample_oncf_frame()
    test1 = _to_test1_layout(oncf)
    liaison = _liaison_lookup([10, 20, 30])

    feat_oncf = build_training_rows(make_clean_dataset(oncf, liaison)[0])
    feat_test1 = build_training_rows(make_clean_dataset(test1, liaison)[0])

    assert list(feat_oncf.columns) == list(feat_test1.columns)
    assert feat_oncf.dtypes.equals(feat_test1.dtypes)
    pd.testing.assert_frame_equal(
        feat_oncf.reset_index(drop=True),
        feat_test1.reset_index(drop=True),
    )


def test_feature_dtypes_are_stable_without_nan():
    """NbrVoySegment/DelaiAnticipation must stay float64 and the departure column
    datetime64[ns] even when the source file has no missing values. In oncf_data
    these are float64/ns; if a clean retrain file (test1) flips them to int64/us,
    the feature schema diverges -- which is not allowed.
    """
    oncf = _sample_oncf_frame()  # no NaN in the numeric columns
    clean, _, _ = make_clean_dataset(oncf, _liaison_lookup([10, 20, 30]))
    feats = build_training_rows(clean)

    assert str(feats["NbrVoySegment"].dtype) == "float64"
    assert str(feats["DelaiAnticipation"].dtype) == "float64"
    assert str(feats["PrixParLiaison"].dtype) == "float64"
    assert str(feats["DateHeureDepartVoyageSegment"].dtype) == "datetime64[ns]"


def test_is_self_purchase_correct_when_achteurid_is_float64():
    """When AchteurId is float64 (dtype promotion in the combined oncf+test1 clean),
    is_self_purchase must be 1 where AchteurId numerically equals CodeClient.
    Bug: astype(str) converts 104078.0 -> '104078.0' which != '104078' -> silent 0."""
    oncf = _sample_oncf_frame()
    # Force AchteurId to float64 to simulate pandas dtype promotion in concat
    oncf = oncf.copy()
    oncf["AchteurId"] = oncf["AchteurId"].astype(float)
    liaison = _liaison_lookup([10, 20, 30])

    clean, _, _ = make_clean_dataset(oncf, liaison)
    feats = build_training_rows(clean)

    # Client 1: all rows have AchteurId == CodeClient == 1 (self-purchases)
    self_rows = feats[feats["CodeClient"] == 1]["is_self_purchase"]
    assert (self_rows == 1).all(), (
        f"is_self_purchase should be 1 for self-purchasers when AchteurId is float64, "
        f"got {self_rows.value_counts().to_dict()}"
    )


def test_dmy_dates_are_parsed_day_first():
    """test1 dates are D/M/Y. Cleaning must parse them day-first so depart_hour
    and depart_month match the real schedule (not month/day swapped)."""
    # All days > 12 so month-first parsing fails -> day-first fallback engages.
    dates = ["13/01/2021 07:30", "14/01/2021 17:30", "15/01/2021 08:00"]
    rows = [
        _oncf_row(code_client=1, liaison_id=10, date=d)
        for d in dates
    ]
    clean, _, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))

    clean = clean.sort_values("DateHeureDepartVoyageSegment")
    assert clean["depart_hour"].tolist() == [7, 17, 8]
    assert clean["depart_month"].unique().tolist() == [1]
