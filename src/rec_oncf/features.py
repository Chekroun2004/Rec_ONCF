from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd


# Canonical output dtypes. build_training_rows must produce EXACTLY these,
# regardless of input-file quirks: the presence/absence of NaN flips a column
# between int64 and float64, and the datetime parse path flips ns<->us. Pinning
# the dtypes guarantees retrain data (test1.csv, future files) yields a feature
# table identical to oncf_data. These values match data/processed/features.parquet,
# so enforcing them is a no-op for the oncf pipeline.
_OUTPUT_DTYPES: dict[str, str] = {
    "DateHeureDepartVoyageSegment": "datetime64[ns]",
    "CodeClient": "int64",
    "PrixParLiaison": "float64",
    "NbrVoySegment": "float64",
    "DelaiAnticipation": "float64",
    "user_trip_index": "int64",
    "days_since_prev": "float64",
    "user_top_liaison_share": "float64",
    "depart_hour": "int32",
    "depart_dow": "int32",
    "depart_month": "int32",
    "depart_hour_sin": "float64",
    "depart_hour_cos": "float64",
    "depart_dow_sin": "float64",
    "depart_dow_cos": "float64",
    "depart_month_sin": "float64",
    "depart_month_cos": "float64",
    "is_self_purchase": "int64",
}


def _rolling_top_share(series: pd.Series) -> pd.Series:
    """For each row at position i in the series, return the share of the most
    frequent value in series[0:i] (strictly past observations). NaN for i=0.
    """
    counts: Counter[str] = Counter()
    out = np.empty(len(series), dtype=float)
    out[0] = np.nan
    for i, value in enumerate(series.tolist()):
        if i > 0:
            top_n = counts.most_common(1)[0][1]
            out[i] = top_n / float(i)
        counts[str(value)] += 1
    return pd.Series(out, index=series.index)


def build_training_rows(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()
    df["CodeClient"] = df["CodeClient"].astype("int64")
    df["LiaisonId"] = df["LiaisonId"].astype(str)

    df = df.sort_values(["CodeClient", "DateHeureDepartVoyageSegment"]).reset_index(drop=True)
    df["user_trip_index"] = df.groupby("CodeClient").cumcount()

    df["prev_liaison"] = df.groupby("CodeClient")["LiaisonId"].shift(1)

    prev_depart = df.groupby("CodeClient")["DateHeureDepartVoyageSegment"].shift(1)
    df["days_since_prev"] = (
        (df["DateHeureDepartVoyageSegment"] - prev_depart).dt.total_seconds() / 86400.0
    )

    # user_top_liaison_share: fraction of past trips on the user's most-frequent
    # liaison, computed using strictly past observations (history up to but
    # excluding the current row). Captures the "loyalty" of the user to one
    # dominant route. NaN for the first trip (no history).
    df["user_top_liaison_share"] = (
        df.groupby("CodeClient", group_keys=False)["LiaisonId"]
        .apply(_rolling_top_share)
    )

    df["is_self_purchase"] = (
        pd.to_numeric(df["AchteurId"], errors="coerce") == df["CodeClient"]
    ).fillna(False).astype(int)

    cat_cols = [
        "TypeParcoursId",
        "ClassificationId",
        "ClassePhysiqueId",
        "NiveauPrixId",
        "TrainAutocarId",
        "CarteClientId",
        "prev_liaison",
    ]
    for c in cat_cols:
        df[c] = df[c].astype("Int64").astype(str)

    if "depart_hour_sin" not in df.columns or "depart_hour_cos" not in df.columns:
        df["depart_hour_sin"] = np.sin(2.0 * np.pi * df["depart_hour"] / 24.0)
        df["depart_hour_cos"] = np.cos(2.0 * np.pi * df["depart_hour"] / 24.0)
    if "depart_dow_sin" not in df.columns or "depart_dow_cos" not in df.columns:
        df["depart_dow_sin"] = np.sin(2.0 * np.pi * df["depart_dow"] / 7.0)
        df["depart_dow_cos"] = np.cos(2.0 * np.pi * df["depart_dow"] / 7.0)
    if "depart_month_sin" not in df.columns or "depart_month_cos" not in df.columns:
        if "depart_month" not in df.columns:
            df["depart_month"] = df["DateHeureDepartVoyageSegment"].dt.month
        df["depart_month_sin"] = np.sin(2.0 * np.pi * df["depart_month"] / 12.0)
        df["depart_month_cos"] = np.cos(2.0 * np.pi * df["depart_month"] / 12.0)

    num_cols = [
        "CodeClient",
        "PrixParLiaison",
        "NbrVoySegment",
        "DelaiAnticipation",
        "user_trip_index",
        "days_since_prev",
        "user_top_liaison_share",
        "depart_hour",
        "depart_dow",
        "depart_month",
        "depart_hour_sin",
        "depart_hour_cos",
        "depart_dow_sin",
        "depart_dow_cos",
        "depart_month_sin",
        "depart_month_cos",
        "is_self_purchase",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_cols = [
        "DateHeureDepartVoyageSegment",
        "LiaisonId",
        *cat_cols,
        *num_cols,
    ]

    result = df[out_cols].copy()
    # Pin canonical dtypes so the schema never depends on input-file quirks.
    for col, dtype in _OUTPUT_DTYPES.items():
        if str(result[col].dtype) != dtype:
            result[col] = result[col].astype(dtype)
    return result


# Trip-context columns whose values come from the booking row itself (price,
# class, etc.). At inference time we don't know them for the future trip, so
# we proxy them with the last historical booking's values.
_TRIP_CONTEXT_COLS = [
    "TypeParcoursId",
    "ClassificationId",
    "ClassePhysiqueId",
    "NiveauPrixId",
    "TrainAutocarId",
    "CarteClientId",
    "PrixParLiaison",
    "NbrVoySegment",
    "DelaiAnticipation",
    "AchteurId",
]


def compute_inference_row(
    history_df: pd.DataFrame,
    *,
    asof: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a single feature row to predict the NEXT trip for a user.

    Unlike ``build_training_rows`` (which produces one row per historical
    booking using past + present information), this function produces a SINGLE
    row whose history-derived features (prev_liaison, user_trip_index,
    days_since_prev) reflect the moment ``asof`` -- i.e. the moment the user
    opens the app and we want to predict their next trip.

    Trip-context features (price, class, etc.) for the unknown future trip are
    proxied with the most recent historical booking's values.

    Args:
        history_df: All known bookings of one user, must contain at least
            CodeClient, LiaisonId, DateHeureDepartVoyageSegment and the
            trip-context columns. Order does not matter (sorted internally).
        asof: Reference datetime for the prediction (defaults to "now" in UTC
            naive). The cyclic temporal features and ``days_since_prev`` are
            computed from this value.

    Returns:
        A 1-row DataFrame in the schema produced by ``build_training_rows``.
    """
    if history_df.empty:
        raise ValueError("history_df is empty; cannot build an inference row")

    if asof is None:
        asof = pd.Timestamp(datetime.now())
    asof = pd.Timestamp(asof)

    sorted_hist = history_df.sort_values("DateHeureDepartVoyageSegment")
    last = sorted_hist.iloc[-1].copy()

    last_liaison = str(last["LiaisonId"])
    last_date = pd.Timestamp(last["DateHeureDepartVoyageSegment"])

    user_trip_index = float(len(sorted_hist))
    days_since_prev = max((asof - last_date).total_seconds() / 86400.0, 0.0)

    # user_top_liaison_share: share of the most-frequent liaison in the user's
    # FULL history (since the next trip is "in the future" of all observed rows).
    liaison_counts = sorted_hist["LiaisonId"].astype(str).value_counts()
    top_share = float(liaison_counts.iloc[0]) / float(len(sorted_hist))

    code_client = last["CodeClient"]

    row: dict = {
        "DateHeureDepartVoyageSegment": asof,
        "LiaisonId": "__unknown__",  # placeholder, dropped before predict
        "CodeClient": pd.to_numeric(code_client, errors="coerce"),
        "prev_liaison": last_liaison,
        "user_trip_index": user_trip_index,
        "days_since_prev": days_since_prev,
        "user_top_liaison_share": top_share,
        "is_self_purchase": int(
            str(last.get("AchteurId", code_client)) == str(code_client)
        ),
    }

    # Proxy trip-context features with the most recent booking's values.
    for col in _TRIP_CONTEXT_COLS:
        if col == "AchteurId":
            continue
        val = last.get(col, np.nan)
        if col in {
            "TypeParcoursId", "ClassificationId", "ClassePhysiqueId",
            "NiveauPrixId", "TrainAutocarId", "CarteClientId",
        }:
            num = pd.to_numeric(val, errors="coerce")
            row[col] = "nan" if pd.isna(num) else str(int(num))
        else:
            row[col] = pd.to_numeric(val, errors="coerce")

    # Temporal features derived from `asof`
    row["depart_hour"]  = float(asof.hour)
    row["depart_dow"]   = float(asof.dayofweek)
    row["depart_month"] = float(asof.month)
    row["depart_hour_sin"]  = float(np.sin(2.0 * np.pi * asof.hour / 24.0))
    row["depart_hour_cos"]  = float(np.cos(2.0 * np.pi * asof.hour / 24.0))
    row["depart_dow_sin"]   = float(np.sin(2.0 * np.pi * asof.dayofweek / 7.0))
    row["depart_dow_cos"]   = float(np.cos(2.0 * np.pi * asof.dayofweek / 7.0))
    row["depart_month_sin"] = float(np.sin(2.0 * np.pi * asof.month / 12.0))
    row["depart_month_cos"] = float(np.cos(2.0 * np.pi * asof.month / 12.0))

    out = pd.DataFrame([row])
    return out
