from __future__ import annotations

import numpy as np
import pandas as pd

from rec_oncf.cleaning import MIN_TRIPS_FOR_TRAINING, make_clean_dataset


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


def test_cancellation_propagates_to_previous_booking():
    """A NbrVoySegment <= 0 row must remove the cancellation AND the booking before it.

    We pad with extra bookings so the client stays above MIN_TRIPS_FOR_TRAINING
    after the cancellation+booking pair is removed; otherwise the cold-start
    filter would drop the whole client and obscure the propagation behaviour.
    """
    raw = pd.DataFrame([
        _row(code_client=1, liaison_id=10, date="2020-01-01", ar=1),  # booking
        _row(code_client=1, liaison_id=10, date="2020-01-02", nbr=0, ar=1),  # cancellation
        _row(code_client=1, liaison_id=20, date="2020-01-03", ar=1),
        _row(code_client=1, liaison_id=30, date="2020-01-04", ar=1),
        _row(code_client=1, liaison_id=40, date="2020-01-05", ar=1),
        _row(code_client=1, liaison_id=50, date="2020-01-06", ar=1),
    ])
    clean, report, _ = make_clean_dataset(raw, _liaison_lookup([10, 20, 30, 40, 50]))
    remaining_dates = clean["DateHeureDepartVoyageSegment"].dt.strftime("%Y-%m-%d").tolist()
    # The Jan 1 booking and the Jan 2 cancellation must both be gone.
    assert "2020-01-01" not in remaining_dates
    assert "2020-01-02" not in remaining_dates
    # Other bookings preserved
    assert "2020-01-03" in remaining_dates
    assert "2020-01-04" in remaining_dates
    assert report["rows_removed_negative"] >= 2


def test_cold_start_clients_are_removed():
    """Clients with strictly fewer than MIN_TRIPS_FOR_TRAINING bookings are excluded."""
    rows: list[dict] = []
    # Client 1 has only 2 bookings -> cold start -> removed
    rows.append(_row(code_client=1, liaison_id=10, date="2020-01-01"))
    rows.append(_row(code_client=1, liaison_id=20, date="2020-02-01"))
    # Client 2 has MIN_TRIPS bookings -> kept
    for i in range(MIN_TRIPS_FOR_TRAINING):
        rows.append(_row(code_client=2, liaison_id=10, date=f"2020-0{i + 1}-15"))

    clean, report, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10, 20]))
    remaining_clients = set(clean["CodeClient"].unique())
    assert 1 not in remaining_clients, "cold-start client 1 should have been removed"
    assert 2 in remaining_clients, "active client 2 should be kept"
    assert report["clients_removed_cold_start"] >= 1


def test_liaison_join_overlap_rate_reported():
    """The cleaning report must accurately measure the overlap with Liaison.csv."""
    rows = [
        _row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-01")
        for i in range(MIN_TRIPS_FOR_TRAINING)
    ]
    clean, report, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))
    assert report["join_status"] == "merged"
    assert report["join_overlap_rate"] == 1.0
    assert "DesignationFrGareDepart" in clean.columns
    assert "DesignationFrGareArrive" in clean.columns


def test_essential_missing_rows_dropped():
    """Rows missing CodeClient, LiaisonId or departure date are removed."""
    rows = [
        _row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-01")
        for i in range(MIN_TRIPS_FOR_TRAINING + 2)
    ]
    raw = pd.DataFrame(rows)
    # Inject one row with missing departure date
    raw.loc[len(raw)] = _row(code_client=1, liaison_id=10, date="not-a-date")
    clean, report, _ = make_clean_dataset(raw, _liaison_lookup([10]))
    # All cleaned rows must have a parsed datetime
    assert clean["DateHeureDepartVoyageSegment"].notna().all()


def test_cyclic_features_are_added_and_within_unit_circle():
    """Cyclic encodings sin/cos must be created and their squared sum equal 1."""
    rows = [
        _row(code_client=1, liaison_id=10, date=f"2020-0{i + 1}-15")
        for i in range(MIN_TRIPS_FOR_TRAINING + 1)
    ]
    clean, _, _ = make_clean_dataset(pd.DataFrame(rows), _liaison_lookup([10]))
    for prefix in ("depart_hour", "depart_dow", "depart_month"):
        sin_col = f"{prefix}_sin"
        cos_col = f"{prefix}_cos"
        assert sin_col in clean.columns
        assert cos_col in clean.columns
        norm = clean[sin_col] ** 2 + clean[cos_col] ** 2
        np.testing.assert_allclose(norm.to_numpy(), 1.0, atol=1e-9)
