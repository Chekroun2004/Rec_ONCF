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
_MAX_MEM_CACHE = 2048
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
        str(lid): (
            normalize_station_name(row["DesignationFrGareDepart"]),
            normalize_station_name(row["DesignationFrGareArrive"]),
        )
        for lid, row in first.iterrows()
    }


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
        return _parse_schedule_html(resp.text)
    except Exception:
        return []


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
    if len(_mem_cache) >= _MAX_MEM_CACHE:
        oldest = min(_mem_cache, key=lambda k: _mem_cache[k][0])
        del _mem_cache[oldest]
    _mem_cache[key] = (time.time(), data)


def get_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    date: datetime,
    *,
    redis_client=None,
) -> list[dict[str, str]]:
    """Return departures for a liaison. Cascade: Redis -> in-memory -> scrape. Returns [] if unknown."""
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
