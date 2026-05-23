from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from rec_oncf.schedule import (
    STATION_CODES,
    _parse_schedule_html,
    build_liaison_station_map,
    fetch_departures,
    get_schedule,
    normalize_station_name,
)

# ---------------------------------------------------------------------------
# HTML fixture (used in Tasks 2 and 3 tests)
# ---------------------------------------------------------------------------

_SCHEDULE_HTML = """
<!DOCTYPE html>
<html><body>
<div class="container">
  <table>
    <tbody>
      <tr><td>06:00</td><td>09:30</td><td>1234</td></tr>
      <tr><td>09:00</td><td>12:30</td><td>5678</td></tr>
    </tbody>
  </table>
</div>
</body></html>
"""

_EMPTY_HTML = """
<!DOCTYPE html>
<html><body>
<div class="container">
  <table><tbody></tbody></table>
</div>
</body></html>
"""

_NO_TABLE_HTML = "<html><body><div class='container'></div></body></html>"


def _mini_clean() -> pd.DataFrame:
    return pd.DataFrame({
        "LiaisonId":               ["14347", "17786", "14347"],
        "DesignationFrGareDepart": ["CASA VOYAGEURS", "MARRAKECH", "CASA VOYAGEURS"],
        "DesignationFrGareArrive": ["MARRAKECH", "RABAT AGDAL", "MARRAKECH"],
    })


def _mock_response(html: str) -> MagicMock:
    mock = MagicMock()
    mock.text = html
    return mock


@pytest.fixture(autouse=True)
def clear_mem_cache():
    import rec_oncf.schedule as _sched
    _sched._mem_cache.clear()
    yield
    _sched._mem_cache.clear()


# ---------------------------------------------------------------------------
# Task 1 tests: STATION_CODES + normalize + build_liaison_station_map
# ---------------------------------------------------------------------------

def test_station_codes_contains_casa_voyageurs():
    code_g, code_r, _ = STATION_CODES["CASA VOYAGEURS"]
    assert code_g == "00200"
    assert code_r == "0093"


def test_station_codes_contains_marrakech():
    code_g, code_r, _ = STATION_CODES["MARRAKECH"]
    assert code_g == "00110"
    assert code_r == "0093"


def test_station_codes_contains_martil_supratours():
    _, code_r, _ = STATION_CODES["MARTIL"]
    assert code_r == "0011"  # Supratours network


def test_normalize_station_name():
    assert normalize_station_name("Casa Oasis") == "CASA OASIS"
    assert normalize_station_name("  KENITRA  ") == "KENITRA"
    assert normalize_station_name("fes") == "FES"


def test_build_liaison_station_map_basic():
    m = build_liaison_station_map(_mini_clean())
    assert m["14347"] == ("CASA VOYAGEURS", "MARRAKECH")
    assert m["17786"] == ("MARRAKECH", "RABAT AGDAL")


def test_build_liaison_station_map_deduplicates():
    m = build_liaison_station_map(_mini_clean())
    assert len(m) == 2  # "14347" appears twice but maps to one entry


# ---------------------------------------------------------------------------
# Task 2 tests: _parse_schedule_html
# ---------------------------------------------------------------------------

def test_parse_schedule_html_returns_departures():
    result = _parse_schedule_html(_SCHEDULE_HTML)
    assert len(result) == 2
    assert result[0] == {"depart": "06:00", "arrive": "09:30", "train": "1234"}
    assert result[1] == {"depart": "09:00", "arrive": "12:30", "train": "5678"}


def test_parse_schedule_html_empty_tbody():
    assert _parse_schedule_html(_EMPTY_HTML) == []


def test_parse_schedule_html_no_table():
    assert _parse_schedule_html(_NO_TABLE_HTML) == []


# ---------------------------------------------------------------------------
# Task 3 tests: fetch_departures + get_schedule + caching
# ---------------------------------------------------------------------------

def test_fetch_departures_builds_oncf_url_with_codes():
    with patch("rec_oncf.schedule.requests.get", return_value=_mock_response(_SCHEDULE_HTML)) as mock_get:
        result = fetch_departures(
            "CASA VOYAGEURS", "00200", "0093",
            "MARRAKECH", "00110", "0093",
            datetime(2026, 5, 7, 8, 0),
        )
    assert len(result) == 2
    called_url = mock_get.call_args[0][0]
    assert "oncf.ma" in called_url
    assert "00200" in called_url
    assert "00110" in called_url


def test_fetch_departures_returns_empty_on_network_error():
    with patch("rec_oncf.schedule.requests.get", side_effect=requests.RequestException("timeout")):
        result = fetch_departures(
            "CASA VOYAGEURS", "00200", "0093",
            "MARRAKECH", "00110", "0093",
            datetime(2026, 5, 7, 8, 0),
        )
    assert result == []


def test_get_schedule_unknown_liaison_returns_empty():
    assert get_schedule("99999", {}, datetime(2026, 5, 7, 8, 0)) == []


def test_get_schedule_unknown_station_returns_empty():
    liaison_map = {"99999": ("SMARA", "LAAYOUNE")}  # not in STATION_CODES
    assert get_schedule("99999", liaison_map, datetime(2026, 5, 7, 8, 0)) == []


def test_get_schedule_uses_memory_cache_on_second_call():
    # Use a unique date (2026-05-09) to avoid collisions with other test entries
    liaison_map = {"14347": ("CASA VOYAGEURS", "MARRAKECH")}
    mock_sched = [{"depart": "07:00", "arrive": "09:30", "train": "1234"}]
    with patch("rec_oncf.schedule.fetch_departures", return_value=mock_sched) as mock_fetch:
        r1 = get_schedule("14347", liaison_map, datetime(2026, 5, 9, 8, 0))
        r2 = get_schedule("14347", liaison_map, datetime(2026, 5, 9, 8, 0))
    assert r1 == mock_sched
    assert r2 == mock_sched
    assert mock_fetch.call_count == 1  # second call served from in-memory cache


# ── local schedule tests ──────────────────────────────────────────────────────

import sys as _sys
from pathlib import Path as _Path
from zoneinfo import ZoneInfo

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from rec_oncf.local_schedule import (
    build_od_index,
    get_local_schedule,
    load_schedule_index,
    parse_horaire_csv,
    save_schedule_index,
)

_SAMPLE_CSV = (
    "CASA PORT;00:00:00;06:15:00;1;1\n"
    "MOHAMMEDIA;06:36:00;06:37:00;5;1\n"
    "RABAT VILLE;07:31:00;07:33:00;12;1\n"
    "KENITRA;08:07:00;00:00:00;17;1\n"
    "KENITRA;00:00:00;08:07:00;1;10\n"
    "RABAT VILLE;08:40:00;08:42:00;5;10\n"
    "CASA PORT;09:20:00;00:00:00;12;10\n"
)

_LOCAL_LIAISON_MAP = {
    "R1": ("CASA PORT", "KENITRA"),
    "R2": ("KENITRA", "CASA PORT"),
    "R3": ("CASA PORT", "NOWHERE"),
}


@pytest.fixture
def horaire_csv_path(tmp_path):
    p = tmp_path / "horaire.csv"
    p.write_text(_SAMPLE_CSV, encoding="utf-8")
    return p


@pytest.fixture
def parsed_df(horaire_csv_path):
    return parse_horaire_csv(horaire_csv_path)


@pytest.fixture
def od_index(parsed_df):
    return build_od_index(parsed_df)


def test_parse_columns_exist(parsed_df):
    assert set(parsed_df.columns) == {"gare", "arrivee", "depart", "ordre", "num_commercial"}


def test_parse_row_count(parsed_df):
    assert len(parsed_df) == 7


def test_parse_normalizes_station_names(parsed_df):
    assert all(name == name.strip().upper() for name in parsed_df["gare"])


def test_parse_whitespace_padding(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("BENI NSAR VILLE               ;00:00:00;10:00:00;1;5\nNADOR VILLE;11:00:00;00:00:00;3;5\n", encoding="utf-8")
    df2 = parse_horaire_csv(p)
    assert df2.iloc[0]["gare"] == "BENI NSAR VILLE"


def test_od_index_is_dict(od_index):
    assert isinstance(od_index, dict)


def test_od_index_basic_pairs(od_index):
    assert ("CASA PORT", "KENITRA") in od_index
    assert ("CASA PORT", "MOHAMMEDIA") in od_index
    assert ("MOHAMMEDIA", "RABAT VILLE") in od_index


def test_od_index_no_reverse_pair(od_index):
    assert ("KENITRA", "CASA PORT") in od_index
    assert ("CASA PORT", "KENITRA") in od_index
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


def test_get_local_schedule_hit(od_index):
    result = get_local_schedule("R1", _LOCAL_LIAISON_MAP, od_index)
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"
    assert result[0]["arrive"] == "08:07"


def test_get_local_schedule_unknown_liaison(od_index):
    assert get_local_schedule("UNKNOWN", _LOCAL_LIAISON_MAP, od_index) == []


def test_get_local_schedule_unmapped_station(od_index):
    assert get_local_schedule("R3", _LOCAL_LIAISON_MAP, od_index) == []


def test_get_local_schedule_time_filter_upcoming(od_index):
    now = datetime(2026, 5, 19, 5, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", _LOCAL_LIAISON_MAP, od_index, now=now)
    assert len(result) > 0
    assert result[0]["depart"] == "06:15"


def test_get_local_schedule_time_filter_past(od_index):
    now = datetime(2026, 5, 19, 9, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R1", _LOCAL_LIAISON_MAP, od_index, now=now)
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_local_schedule_no_now_returns_all(od_index):
    result = get_local_schedule("R2", _LOCAL_LIAISON_MAP, od_index)
    assert len(result) >= 1


def test_get_local_schedule_caps_at_limit():
    od = {("A", "B"): [{"depart": f"{h:02d}:00", "arrive": f"{h+1:02d}:00"} for h in range(8, 18)]}
    lm = {"R": ("A", "B")}
    now = datetime(2026, 5, 19, 6, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R", lm, od, now=now)
    assert len(result) == 3
    assert [t["depart"] for t in result] == ["08:00", "09:00", "10:00"]


def test_get_local_schedule_custom_limit():
    od = {("A", "B"): [{"depart": f"{h:02d}:00", "arrive": f"{h+1:02d}:00"} for h in range(8, 18)]}
    lm = {"R": ("A", "B")}
    now = datetime(2026, 5, 19, 6, 0, tzinfo=ZoneInfo("Africa/Casablanca"))
    result = get_local_schedule("R", lm, od, now=now, limit=5)
    assert len(result) == 5


def test_local_schedule_save_load_roundtrip(od_index, tmp_path):
    path = tmp_path / "schedule_index.joblib"
    save_schedule_index(od_index, path)
    loaded = load_schedule_index(path)
    assert loaded == od_index


def test_load_missing_file_returns_empty(tmp_path):
    result = load_schedule_index(tmp_path / "nonexistent.joblib")
    assert result == {}
