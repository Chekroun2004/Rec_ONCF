from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
    mock.raise_for_status.return_value = None
    return mock


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
    with patch("rec_oncf.schedule.requests.get", side_effect=Exception("timeout")):
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
