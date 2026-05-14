# ONCF Live Schedule Scraping Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich `/recommend` responses with real-time ONCF train departure schedules by scraping `oncf.ma` when the caller passes `include_schedule=true`.

**Architecture:** A new `schedule.py` module holds a static station-code lookup table (24 stations, researched from the ONCF PHP reference scraper at github.com/azihassan/oncf_schedule), a `build_liaison_station_map` helper that derives origin/destination names from `oncf_clean.parquet`, an HTML scraper backed by `BeautifulSoup`, and a two-level cache (Redis → in-memory dict with 1-hour TTL). The `/recommend` endpoint gains an optional `include_schedule: bool` request field; when true it adds a `schedules` dict to the response without touching the existing `recommendations` list.

**Tech Stack:** `requests>=2.32`, `beautifulsoup4>=4.12`, `redis` (already in deps), Python stdlib `urllib.parse`, `json`, `time`.

---

## Background: ONCF station code system

The ONCF schedule search URL uses a two-part station code:
- `code_g` — 5-digit geographic/station code (e.g., `00200` for Casa Voyageurs)
- `code_r` — operator code: `0093` for standard ONCF, `0011` for Supratours-operated stations

URL format:
```
https://www.oncf.ma/fr/Horaires
  ?from[{code_g}][{code_r}]={station_name}
  &to[{code_g}][{code_r}]={station_name}
  &datedep=DD%2FMM%2FYYYY+HH%3AMM
  &dateret=&is-ar=0
```

The response is HTML. The schedule table lives in `div.container > table > tbody > tr`.
Each row: `<td>depart_time</td><td>arrive_time</td><td>train_no</td>`.

The `oncf.ma` SSL certificate has verification issues — use `verify=False` with `requests.get`.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/rec_oncf/schedule.py` | `STATION_CODES`, `normalize_station_name`, `build_liaison_station_map`, `_parse_schedule_html`, `fetch_departures`, `get_schedule`, in-memory cache |
| Create | `tests/test_schedule.py` | 13 tests for all public functions (no network — mocked HTTP) |
| Modify | `apps/api/main.py` | Add `include_schedule` to request, `schedules` to response, build liaison map + Redis at startup |
| Modify | `tests/test_api.py` | Add `liaison_map`/`redis` to fixture; add 3 new tests |
| Modify | `requirements.txt` | Add `requests>=2.32`, `beautifulsoup4>=4.12` |
| Modify | `CLAUDE.md` | Update layout, status, API docs |

---

## Task 1: Dependencies + `STATION_CODES` + `build_liaison_station_map`

**Files:**
- Modify: `requirements.txt`
- Create: `src/rec_oncf/schedule.py`
- Create: `tests/test_schedule.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_schedule.py` with this content:

```python
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
```

- [ ] **Step 2: Run to verify tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py -v
```

Expected: `ModuleNotFoundError: No module named 'rec_oncf.schedule'`

- [ ] **Step 3: Add deps to `requirements.txt`**

Add these two lines at the end of `requirements.txt`:
```
requests>=2.32
beautifulsoup4>=4.12
```

- [ ] **Step 4: Install new deps**

```powershell
.venv\Scripts\pip.exe install "requests>=2.32" "beautifulsoup4>=4.12"
```

Expected output includes: `Successfully installed ...`

- [ ] **Step 5: Create `src/rec_oncf/schedule.py`** with codes + map builder (Tasks 2 and 3 functions are stubs for now — add them in later tasks):

```python
from __future__ import annotations

import json
import time
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# (code_g, code_r, oncf_canonical_name) — oncf_canonical_name is passed in the URL
# Codes sourced from: github.com/azihassan/oncf_schedule/blob/master/codes.php
STATION_CODES: dict[str, tuple[str, str, str]] = {
    "CASA VOYAGEURS":      ("00200", "0093", "CASA VOYAGEURS"),
    "TANGER VILLE":        ("00303", "0093", "TANGER"),
    "KENITRA":             ("00250", "0093", "KENITRA"),
    "RABAT AGDAL":         ("00229", "0093", "RABAT AGDAL"),
    "FES":                 ("00380", "0093", "FES"),
    "MARRAKECH":           ("00110", "0093", "MARRAKECH"),
    "RABAT VILLE":         ("00231", "0093", "RABAT VILLE"),
    "CASA OASIS":          ("00191", "0093", "L'OASIS"),
    "MEKNES":              ("00363", "0093", "MEKNES"),
    "SETTAT":              ("00139", "0093", "SETTAT"),
    "CASA PORT":           ("00206", "0093", "CASA PORT"),
    "MOHAMMEDIA":          ("00217", "0093", "MOHAMMEDIA"),
    "SIDI SLIMANE MEDINA": ("00264", "0093", "SIDI SLIMANE MEDINA"),
    "AIN SEBAA":           ("00213", "0093", "AIN SEBAA"),
    "BERRECHID":           ("00183", "0093", "BERRECHID"),
    "SIDI KACEM":          ("00350", "0093", "SIDI KACEM"),
    "SALE":                ("00237", "0093", "SALE"),
    "MEKNES AL AMIR":      ("00362", "0093", "MEKNES AL AMIR"),
    "BENGUERIR":           ("00120", "0093", "BENGUERIR"),
    "SALE TABRIQUET":      ("00238", "0093", "SALE TABRIQUET"),
    "AEROPORT MED V":      ("00190", "0093", "AEROPORT MED V"),
    "YOUSSOUFIA":          ("00077", "0093", "YOUSSOUFIA"),
    "SAFI":                ("00057", "0093", "SAFI"),
    "MARTIL":              ("00893", "0011", "MARTIL"),
}

_CACHE_TTL = 3600  # seconds
_mem_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}


def normalize_station_name(name: str) -> str:
    return name.strip().upper()


def build_liaison_station_map(clean_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Return {LiaisonId: (origin_name, dest_name)} built from oncf_clean.parquet."""
    first = (
        clean_df.groupby("LiaisonId")[["DesignationFrGareDepart", "DesignationFrGareArrive"]]
        .first()
    )
    return {
        str(lid): (row["DesignationFrGareDepart"], row["DesignationFrGareArrive"])
        for lid, row in first.iterrows()
    }


def _parse_schedule_html(html: str) -> list[dict[str, str]]:
    raise NotImplementedError


def fetch_departures(
    origin_name: str,
    origin_code_g: str,
    origin_code_r: str,
    dest_name: str,
    dest_code_g: str,
    dest_code_r: str,
    date: datetime,
) -> list[dict[str, str]]:
    raise NotImplementedError


def get_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    date: datetime,
    *,
    redis_client=None,
) -> list[dict[str, str]]:
    raise NotImplementedError
```

- [ ] **Step 6: Run Task 1 tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py::test_station_codes_contains_casa_voyageurs tests/test_schedule.py::test_station_codes_contains_marrakech tests/test_schedule.py::test_station_codes_contains_martil_supratours tests/test_schedule.py::test_normalize_station_name tests/test_schedule.py::test_build_liaison_station_map_basic tests/test_schedule.py::test_build_liaison_station_map_deduplicates -v
```

Expected: **6 PASSED**

The Task 2 and 3 tests will fail with `NotImplementedError` — that's correct for now.

- [ ] **Step 7: Run full suite to check for regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all 56 pre-existing tests pass (6 new pass, remaining new ones fail with NotImplementedError).

- [ ] **Step 8: Commit**

```powershell
git add requirements.txt src/rec_oncf/schedule.py tests/test_schedule.py
git commit -m "feat: add schedule module skeleton with station codes and liaison map"
```

---

## Task 2: HTML Parser

**Files:**
- Modify: `src/rec_oncf/schedule.py` (replace `_parse_schedule_html` stub)

- [ ] **Step 1: Run failing HTML parser tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py::test_parse_schedule_html_returns_departures tests/test_schedule.py::test_parse_schedule_html_empty_tbody tests/test_schedule.py::test_parse_schedule_html_no_table -v
```

Expected: **3 FAILED** (`NotImplementedError`)

- [ ] **Step 2: Replace the `_parse_schedule_html` stub in `src/rec_oncf/schedule.py`**

Replace:
```python
def _parse_schedule_html(html: str) -> list[dict[str, str]]:
    raise NotImplementedError
```

With:
```python
def _parse_schedule_html(html: str) -> list[dict[str, str]]:
    """Parse ONCF schedule HTML. Returns [{depart, arrive, train}, ...] or []."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="container")
    if not container:
        return []
    table = container.find("table")
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []
    results = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        results.append({
            "depart": cells[0].get_text(strip=True),
            "arrive": cells[1].get_text(strip=True),
            "train":  cells[2].get_text(strip=True),
        })
    return results
```

- [ ] **Step 3: Run HTML parser tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py::test_parse_schedule_html_returns_departures tests/test_schedule.py::test_parse_schedule_html_empty_tbody tests/test_schedule.py::test_parse_schedule_html_no_table -v
```

Expected: **3 PASSED**

- [ ] **Step 4: Run full test_schedule.py to see progress**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py -v
```

Expected: 9 PASSED (Tasks 1+2), 4 FAILED (Task 3 — still stubs)

- [ ] **Step 5: Commit**

```powershell
git add src/rec_oncf/schedule.py
git commit -m "feat: implement ONCF schedule HTML parser with BeautifulSoup"
```

---

## Task 3: HTTP Fetcher + Caching

**Files:**
- Modify: `src/rec_oncf/schedule.py` (replace `fetch_departures` and `get_schedule` stubs)

- [ ] **Step 1: Run failing fetcher/cache tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py::test_fetch_departures_builds_oncf_url_with_codes tests/test_schedule.py::test_fetch_departures_returns_empty_on_network_error tests/test_schedule.py::test_get_schedule_unknown_liaison_returns_empty tests/test_schedule.py::test_get_schedule_unknown_station_returns_empty tests/test_schedule.py::test_get_schedule_uses_memory_cache_on_second_call -v
```

Expected: **5 FAILED** (`NotImplementedError` or `ImportError`)

- [ ] **Step 2: Replace `fetch_departures` stub in `src/rec_oncf/schedule.py`**

Replace:
```python
def fetch_departures(
    origin_name: str,
    origin_code_g: str,
    origin_code_r: str,
    dest_name: str,
    dest_code_g: str,
    dest_code_r: str,
    date: datetime,
) -> list[dict[str, str]]:
    raise NotImplementedError
```

With:
```python
def fetch_departures(
    origin_name: str,
    origin_code_g: str,
    origin_code_r: str,
    dest_name: str,
    dest_code_g: str,
    dest_code_r: str,
    date: datetime,
) -> list[dict[str, str]]:
    """Scrape oncf.ma schedule page. Returns [] on any network or parse error."""
    date_str = date.strftime("%d/%m/%Y %H:%M")
    url = (
        "https://www.oncf.ma/fr/Horaires"
        f"?from[{origin_code_g}][{origin_code_r}]={quote(origin_name)}"
        f"&to[{dest_code_g}][{dest_code_r}]={quote(dest_name)}"
        f"&datedep={quote(date_str)}"
        f"&dateret=&is-ar=0"
    )
    try:
        resp = requests.get(
            url,
            timeout=10,
            verify=False,  # oncf.ma has SSL cert verification issues
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        return _parse_schedule_html(resp.text)
    except Exception:
        return []
```

- [ ] **Step 3: Add two private cache helpers and replace `get_schedule` stub**

Add these three blocks after `fetch_departures` (replace the existing `get_schedule` stub):

```python
def _mem_cache_get(key: str) -> list[dict[str, str]] | None:
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts >= _CACHE_TTL:
        del _mem_cache[key]
        return None
    return data


def _mem_cache_set(key: str, data: list[dict[str, str]]) -> None:
    _mem_cache[key] = (time.time(), data)


def get_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    date: datetime,
    *,
    redis_client=None,
) -> list[dict[str, str]]:
    """Return departures for a liaison. Cascade: Redis → in-memory → scrape. Returns [] if unknown."""
    stations = liaison_map.get(str(liaison_id))
    if not stations:
        return []

    origin_name, dest_name = stations
    origin_codes = STATION_CODES.get(normalize_station_name(origin_name))
    dest_codes = STATION_CODES.get(normalize_station_name(dest_name))
    if not origin_codes or not dest_codes:
        return []

    origin_code_g, origin_code_r, origin_oncf_name = origin_codes
    dest_code_g, dest_code_r, dest_oncf_name = dest_codes

    cache_key = f"oncf_sched:{origin_code_g}:{dest_code_g}:{date.strftime('%Y-%m-%d')}"

    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    cached = _mem_cache_get(cache_key)
    if cached is not None:
        return cached

    result = fetch_departures(
        origin_oncf_name, origin_code_g, origin_code_r,
        dest_oncf_name, dest_code_g, dest_code_r,
        date,
    )

    _mem_cache_set(cache_key, result)
    if redis_client is not None:
        try:
            redis_client.setex(cache_key, _CACHE_TTL, json.dumps(result))
        except Exception:
            pass

    return result
```

- [ ] **Step 4: Run all `test_schedule.py` tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_schedule.py -v
```

Expected: **13 PASSED**

- [ ] **Step 5: Run full suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all 56 existing + 13 new = **69 PASSED**

- [ ] **Step 6: Commit**

```powershell
git add src/rec_oncf/schedule.py
git commit -m "feat: add ONCF schedule HTTP fetcher with Redis/memory caching"
```

---

## Task 4: API Integration

**Files:**
- Modify: `apps/api/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add 3 failing tests to `tests/test_api.py`**

Add these imports at the top of the file (after the existing imports):
```python
from unittest.mock import patch
```

Modify the `client` fixture to also set `liaison_map` and `redis` on `app.state` (needed for schedule enrichment code path):

Replace the existing `client` fixture:
```python
@pytest.fixture(scope="module")
def client():
    arts = _build_artifacts()
    clean = _build_clean_df()
    app.state.recommender = Recommender.from_data(arts, clean)
    return TestClient(app)
```

With:
```python
@pytest.fixture(scope="module")
def client():
    arts = _build_artifacts()
    clean = _build_clean_df()
    app.state.recommender = Recommender.from_data(arts, clean)
    app.state.liaison_map = {}   # empty — schedule calls return [] in unit tests
    app.state.redis = None
    return TestClient(app)
```

Add these three tests at the bottom of `tests/test_api.py`:
```python
def test_recommend_no_schedules_field_by_default(client):
    resp = client.post("/recommend", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    assert "schedules" not in resp.json()


def test_recommend_include_schedule_adds_schedules_field(client):
    mock_sched = [{"depart": "07:00", "arrive": "09:30", "train": "1234"}]
    with patch("apps.api.main.get_schedule", return_value=mock_sched):
        resp = client.post(
            "/recommend",
            json={"code_client": "1001", "k": 1, "include_schedule": True},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "schedules" in body
    assert isinstance(body["schedules"], dict)
    # Each key is a LiaisonId string; each value is a list of departure dicts
    for lid, deps in body["schedules"].items():
        assert isinstance(lid, str)
        assert isinstance(deps, list)


def test_recommend_include_schedule_unknown_user_no_schedules(client):
    resp = client.post(
        "/recommend",
        json={"code_client": "unknown_xyz", "k": 1, "include_schedule": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Unknown user → cold_start → empty recommendations → schedules key absent
    assert body["recommendations"] == []
    assert "schedules" not in body
```

- [ ] **Step 2: Run new tests to confirm they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py::test_recommend_no_schedules_field_by_default tests/test_api.py::test_recommend_include_schedule_adds_schedules_field tests/test_api.py::test_recommend_include_schedule_unknown_user_no_schedules -v
```

Expected: at least `test_recommend_include_schedule_adds_schedules_field` fails (422 — `include_schedule` not in model yet).

- [ ] **Step 3: Replace `apps/api/main.py`** with the full updated file:

```python
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.recommender import Recommender
from rec_oncf.schedule import build_liaison_station_map, get_schedule


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = default_paths()
    if not paths.xgb_model_path.exists():
        raise RuntimeError(
            f"Model not found: {paths.xgb_model_path}. Run scripts/03_train_ranker.py first."
        )
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.add(
        logs_dir / "api.log",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    app.state.recommender = Recommender.from_paths(paths)
    clean = read_parquet(paths.processed_dataset_parquet)
    app.state.liaison_map = build_liaison_station_map(clean)
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        app.state.redis = r
        logger.info("Redis schedule cache connected")
    except Exception:
        app.state.redis = None
        logger.warning("Redis unavailable — using in-memory schedule cache")
    yield


app = FastAPI(title="ONCF Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    code_client: str
    k: int = Field(default=1, ge=1, le=3)
    include_schedule: bool = False


class RecommendResponse(BaseModel):
    mode: str
    recommendations: list[str]
    schedules: dict[str, list[dict[str, str]]] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse, response_model_exclude_none=True)
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    result: dict = dict(app.state.recommender.recommend(req.code_client, req.k))

    if req.include_schedule and result["recommendations"]:
        now = datetime.now()
        result["schedules"] = {
            lid: get_schedule(lid, app.state.liaison_map, now, redis_client=app.state.redis)
            for lid in result["recommendations"]
        }

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.bind(
        mode=result["mode"],
        k=req.k,
        latency_ms=latency_ms,
        n_recommendations=len(result["recommendations"]),
    ).info("recommend")
    return result
```

- [ ] **Step 4: Run all API tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: **8 PASSED** (5 existing + 3 new)

- [ ] **Step 5: Run full test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: **72 PASSED** (56 pre-existing + 13 schedule + 3 new API)

- [ ] **Step 6: Commit**

```powershell
git add apps/api/main.py tests/test_api.py
git commit -m "feat: wire schedule enrichment into /recommend with include_schedule flag"
```

---

## Task 5: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Repository Layout** — add `schedule.py` under `src/rec_oncf/` and `test_schedule.py` under `tests/`

In the `src/rec_oncf/` section, add after `cold_start.py`:
```
│   ├── schedule.py         # ONCF live schedule scraping — STATION_CODES, build_liaison_station_map,
│   │                       #   fetch_departures, get_schedule (Redis/memory cache, HTML parsing)
```

In the `tests/` section, add:
```
│   ├── test_schedule.py    # 13 tests  ✅ passing  (station codes, HTML parser, HTTP mock, caching)
```

- [ ] **Step 2: Update `requirements.txt` mention** in the Environment section — add `requests`, `beautifulsoup4`

- [ ] **Step 3: Update API section** — add `include_schedule` to the request body and `schedules` to the response:

In `apps/api/main.py` → Request body block, change to:
```json
{"code_client": "12345", "k": 3, "include_schedule": false}
```

Add under the response section:
```
When `include_schedule=true`, response also includes:
`"schedules": {"14347": [{"depart": "07:00", "arrive": "09:30", "train": "1234"}], ...}`
Schedules are cached per (origin, destination, date) for 1 hour (Redis preferred, in-memory fallback).
Stations without a known ONCF code return an empty list silently.
```

- [ ] **Step 4: Update Current Status table** — add a new row:

```
| ONCF schedule scraping      | ✅ Done | `schedule.py` — 24 stations, Redis+memory cache, opt-in via `include_schedule` |
```

- [ ] **Step 5: Remove item 5 from What's Left to Do** — "ONCF schedule API integration" is now done.

- [ ] **Step 6: Update test count** — 56 → 72 in the test suite entry.

- [ ] **Step 7: Commit**

```powershell
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with schedule scraping integration"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| `/recommend` with `include_schedule=true` enriches each LiaisonId | Task 4 |
| Station codes (CASA VOYAGEURS → 00200/0093, etc.) | Task 1 — `STATION_CODES` dict |
| LiaisonId → station names from `oncf_clean.parquet` | Task 1 — `build_liaison_station_map` |
| Redis 1-hour TTL cache | Task 3 — `get_schedule` with `redis_client` |
| In-memory fallback cache | Task 3 — `_mem_cache`, `_mem_cache_get`, `_mem_cache_set` |
| HTML scraping with BeautifulSoup | Task 2 — `_parse_schedule_html` |
| Feature is opt-in (no latency when not requested) | Task 4 — `include_schedule: bool = False` |
| 24 known station mappings | Task 1 — all 24 entries in `STATION_CODES` |

**Placeholder scan:** No TBD, TODO, or "handle edge cases" text — all functions have complete implementations. ✅

**Type consistency:**
- `build_liaison_station_map` → `dict[str, tuple[str, str]]` ✅ matches `get_schedule`'s `liaison_map` param
- `fetch_departures` → `list[dict[str, str]]` ✅ matches `get_schedule` return type
- `get_schedule` → `list[dict[str, str]]` ✅ matches `RecommendResponse.schedules` inner type
- `STATION_CODES` values: `tuple[str, str, str]` (code_g, code_r, oncf_name) ✅ all three used in `get_schedule`
