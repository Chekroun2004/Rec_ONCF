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
