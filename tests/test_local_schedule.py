# tests/test_local_schedule.py
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rec_oncf.local_schedule import (
    build_od_index,
    get_local_schedule,
    load_schedule_index,
    parse_horaire_csv,
    save_schedule_index,
)

# ── fixture CSV minimal ────────────────────────────────────────────────────
# Train 1 : CASA PORT → MOHAMMEDIA → RABAT VILLE → KENITRA
# Train 10: KENITRA → RABAT VILLE → CASA PORT
SAMPLE_CSV = (
    "CASA PORT;00:00:00;06:15:00;1;1\n"
    "MOHAMMEDIA;06:36:00;06:37:00;5;1\n"
    "RABAT VILLE;07:31:00;07:33:00;12;1\n"
    "KENITRA;08:07:00;00:00:00;17;1\n"
    "KENITRA;00:00:00;08:07:00;1;10\n"
    "RABAT VILLE;08:40:00;08:42:00;5;10\n"
    "CASA PORT;09:20:00;00:00:00;12;10\n"
)


@pytest.fixture
def horaire_csv_path(tmp_path):
    p = tmp_path / "horaire.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return p


@pytest.fixture
def df(horaire_csv_path):
    return parse_horaire_csv(horaire_csv_path)


@pytest.fixture
def od_index(df):
    return build_od_index(df)


# ── parse_horaire_csv ──────────────────────────────────────────────────────

def test_parse_columns_exist(df):
    assert set(df.columns) == {"gare", "arrivee", "depart", "ordre", "num_commercial"}


def test_parse_row_count(df):
    assert len(df) == 7


def test_parse_normalizes_station_names(df):
    assert all(name == name.strip().upper() for name in df["gare"])


def test_parse_whitespace_padding(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(
        "BENI NSAR VILLE               ;00:00:00;10:00:00;1;5\n"
        "NADOR VILLE;11:00:00;00:00:00;3;5\n",
        encoding="utf-8",
    )
    df2 = parse_horaire_csv(p)
    assert df2.iloc[0]["gare"] == "BENI NSAR VILLE"


# ── build_od_index ─────────────────────────────────────────────────────────

def test_od_index_is_dict(od_index):
    assert isinstance(od_index, dict)


def test_od_index_basic_pairs(od_index):
    assert ("CASA PORT", "KENITRA") in od_index
    assert ("CASA PORT", "MOHAMMEDIA") in od_index
    assert ("MOHAMMEDIA", "RABAT VILLE") in od_index


def test_od_index_no_reverse_pair(od_index):
    assert ("KENITRA", "CASA PORT") in od_index      # Train 10
    assert ("CASA PORT", "KENITRA") in od_index      # Train 1
    assert od_index[("KENITRA", "CASA PORT")][0]["depart"] == "08:07"
    assert od_index[("CASA PORT", "KENITRA")][0]["depart"] == "06:15"


def test_od_index_terminus_not_origin(od_index):
    kenitra_origins = [v for (o, _), v in od_index.items() if o == "KENITRA"]
    for trips in kenitra_origins:
        for t in trips:
            assert t["depart"] != "00:00"


def test_od_index_origin_not_destination(od_index):
    casa_dests = [v for (_, d), v in od_index.items() if d == "CASA PORT"]
    for trips in casa_dests:
        for t in trips:
            assert t["arrive"] != "00:00"


def test_od_index_times_are_hhmm(od_index):
    for trips in od_index.values():
        for t in trips:
            assert len(t["depart"]) == 5
            assert t["depart"][2] == ":"
            assert len(t["arrive"]) == 5
            assert t["arrive"][2] == ":"


def test_od_index_sorted_by_departure(od_index):
    for trips in od_index.values():
        deps = [t["depart"] for t in trips]
        assert deps == sorted(deps)


# ── get_local_schedule ─────────────────────────────────────────────────────

LIAISON_MAP = {
    "R1": ("CASA PORT", "KENITRA"),
    "R2": ("KENITRA", "CASA PORT"),
    "R3": ("CASA PORT", "NOWHERE"),
}


def test_get_local_schedule_hit(od_index):
    result = get_local_schedule("R1", LIAISON_MAP, od_index)
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"
    assert result[0]["arrive"] == "08:07"


def test_get_local_schedule_unknown_liaison(od_index):
    assert get_local_schedule("UNKNOWN", LIAISON_MAP, od_index) == []


def test_get_local_schedule_unmapped_station(od_index):
    assert get_local_schedule("R3", LIAISON_MAP, od_index) == []


def test_get_local_schedule_time_filter_upcoming(od_index):
    now = datetime(2026, 5, 19, 5, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", LIAISON_MAP, od_index, now=now)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"


def test_get_local_schedule_time_filter_past(od_index):
    now = datetime(2026, 5, 19, 9, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", LIAISON_MAP, od_index, now=now)
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_local_schedule_no_now_returns_all(od_index):
    result = get_local_schedule("R2", LIAISON_MAP, od_index)
    assert len(result) >= 1


# ── save / load roundtrip ──────────────────────────────────────────────────

def test_save_load_roundtrip(od_index, tmp_path):
    path = tmp_path / "schedule_index.joblib"
    save_schedule_index(od_index, path)
    loaded = load_schedule_index(path)
    assert loaded == od_index


def test_load_missing_file_returns_empty(tmp_path):
    result = load_schedule_index(tmp_path / "nonexistent.joblib")
    assert result == {}
