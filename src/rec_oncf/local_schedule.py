# src/rec_oncf/local_schedule.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

_TZ_CASABLANCA = ZoneInfo("Africa/Casablanca")


def parse_horaire_csv(path: Path) -> pd.DataFrame:
    """Reads horaire.csv (no header, auto-detected separator) and normalizes station names.

    Supports both comma-separated and semicolon-separated variants.
    UTF-8 BOM is handled transparently via the utf-8-sig encoding.
    """
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
) -> list[dict[str, str]]:
    """Returns upcoming departures for a LiaisonId.

    If now is given, filters out past departures. now must be timezone-aware
    and is normalized to Africa/Casablanca before comparison.
    If all trains have passed, returns first 3 (next-day cycle).
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
    return upcoming if upcoming else trips[:3]


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
