from __future__ import annotations

import csv

import pandas as pd


def _detect_sep(path, fallback: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        sample = handle.read(4096)

    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except csv.Error:
        return fallback


def read_csv(path, *, usecols: list[str] | None = None, sep: str = ";") -> pd.DataFrame:
    detected_sep = _detect_sep(path, sep)
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False, sep=detected_sep, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=usecols, low_memory=False, sep=detected_sep, encoding="cp1252")


def write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path, *, sep: str = ";") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")


def read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)
