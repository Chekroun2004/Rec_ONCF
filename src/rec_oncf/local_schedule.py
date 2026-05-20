# src/rec_oncf/local_schedule.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

_TZ_CASABLANCA = ZoneInfo("Africa/Casablanca")


def _normalize_time(t: str) -> str:
    """Normalize H:MM:SS or HH:MM:SS to zero-padded HH:MM:SS."""
    if not isinstance(t, str) or not t.strip():
        return t
    parts = t.strip().split(":")
    if len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return t


def parse_horaire_csv(path: Path) -> pd.DataFrame:
    """Reads horaire.csv and normalizes station names and times.

    Supports two formats:
    - New format (with header): Gare,HeureArrivee,HeureDepart,Order,NumeroCommercial
    - Old format (no header): gare;arrivee;depart;ordre;num_commercial
    Separator is auto-detected. UTF-8 BOM handled via utf-8-sig encoding.
    Times are normalized to HH:MM:SS with leading zeros.
    """
    _COL_MAP = {
        "Gare": "gare",
        "HeureArrivee": "arrivee",
        "HeureDepart": "depart",
        "Order": "ordre",
        "NumeroCommercial": "num_commercial",
    }
    _REQUIRED = {"gare", "arrivee", "depart", "ordre", "num_commercial"}

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        encoding="utf-8-sig",
    )
    renamed = df.rename(columns=_COL_MAP)
    if _REQUIRED <= set(renamed.columns):
        df = renamed[list(_REQUIRED)]
    else:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            header=None,
            names=["gare", "arrivee", "depart", "ordre", "num_commercial"],
            dtype=str,
            encoding="utf-8-sig",
        )

    df["gare"] = df["gare"].str.strip().str.upper()
    for col in ("arrivee", "depart"):
        df[col] = df[col].str.strip().apply(_normalize_time)
    df["ordre"] = df["ordre"].astype(int)
    return df


def build_od_index(
    df: pd.DataFrame,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Generates all O/D pairs from horaire.csv trips.

    00:00:00 means terminus/origin station marker (not midnight).
    - arrivee=00:00:00 → this stop CANNOT be a destination
    - depart=00:00:00  → this stop CANNOT be an origin
    """
    index: dict[tuple[str, str], list[dict[str, str]]] = {}

    for _num, group in df.groupby("num_commercial"):
        stops = group.sort_values("ordre").to_dict("records")

        for i, orig in enumerate(stops):
            if orig["depart"] == "00:00:00":
                continue  # terminus — no valid departure
            dep_hhmm = orig["depart"][:5]  # "HH:MM"

            for dest in stops[i + 1 :]:
                if dest["arrivee"] == "00:00:00":
                    continue  # origin station — no valid arrival
                arr_hhmm = dest["arrivee"][:5]  # "HH:MM"

                key = (orig["gare"], dest["gare"])
                index.setdefault(key, []).append(
                    {"depart": dep_hhmm, "arrive": arr_hhmm}
                )

    for key in index:
        index[key].sort(key=lambda x: x["depart"])

    return index


def get_local_schedule(
    liaison_id: str,
    liaison_map: dict[str, tuple[str, str]],
    od_index: dict[tuple[str, str], list[dict[str, str]]],
    now: datetime | None = None,
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Returns upcoming departures for a LiaisonId.

    If now is given, filters out past departures. now must be timezone-aware
    and is normalized to Africa/Casablanca before comparison.
    Returns at most `limit` trips (default 3). If all trains have passed,
    returns the first `limit` of the day (next-day cycle).
    """
    stations = liaison_map.get(str(liaison_id))
    if not stations:
        return []

    origin = stations[0].strip().upper()
    dest = stations[1].strip().upper()
    trips = od_index.get((origin, dest), [])
    if not trips:
        return []

    if now is None:
        return trips

    if now.tzinfo is None:
        raise ValueError(
            "now must be timezone-aware; pass e.g. datetime.now(tz=ZoneInfo('Africa/Casablanca'))"
        )
    current_hhmm = now.astimezone(_TZ_CASABLANCA).strftime("%H:%M")
    upcoming = [t for t in trips if t["depart"] >= current_hhmm]
    return (upcoming or trips)[:limit]


def save_schedule_index(
    od_index: dict[tuple[str, str], list[dict[str, str]]],
    path: Path,
) -> None:
    """Serializes the O/D index with joblib."""
    joblib.dump(od_index, path)


def load_schedule_index(
    path: Path,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Loads the O/D index from disk. Returns {} if file not found."""
    if not path.exists():
        return {}
    return joblib.load(path)
