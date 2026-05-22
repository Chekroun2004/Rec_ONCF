from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ONCF_COLS = [
    "TrajetAllerRetour",
    "TypeParcoursId",
    "CodeClient",
    "ClassificationId",
    "ClassePhysiqueId",
    "NiveauPrixId",
    "TrainAutocarId",
    "LiaisonVoyageurSegmentIdSTG",
    "CarteClientId",
    "PrixParLiaison",
    "NbrVoySegment",
    "DatePaiement",
    "DateHeureDepartVoyageSegment",
    "DelaiAnticipation",
]

OPTIONAL_ONCF_COLS = [
    "PosteVenteId",
    "AchteurId",
    "TypeOperationVenteApresVenteId",
    "PrixServices",
]

REQUIRED_LIAISON_COLS = [
    "LiaisonId",
    "DesignationFrGareDepart",
    "DesignationFrGareArrive",
]

# Known column-name variants in retrain files (e.g. test1.csv) that must be
# mapped to the canonical oncf_data schema before anything else runs.
# Without this, an unrecognized buyer column is silently treated as all-NA,
# producing a wrong (always-zero) is_self_purchase feature.
COLUMN_ALIASES = {
    "Acheteurid": "AchteurId",
}

# Minimum number of trips required to be included in training data.
# Clients below this threshold will never receive a recommendation in production.
MIN_TRIPS_FOR_TRAINING = 3


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column-name variants to the canonical oncf_data schema.

    Aliases only when the canonical name is not already present, so an existing
    column is never clobbered. Returns the original frame untouched if there is
    nothing to rename.
    """
    rename = {
        src: dst
        for src, dst in COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    return df.rename(columns=rename) if rename else df


def _to_datetime(series: pd.Series) -> pd.Series:
    # First probe assumes month-first (oncf_data convention). If too many values
    # fail, the data is day-first (test1 convention) and we re-parse. The probe's
    # dayfirst warning is expected here, so we silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dt = pd.to_datetime(series, errors="coerce", utc=False)
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
    return dt


def _drop_constant_columns(df: pd.DataFrame, *, ignore: set[str]) -> tuple[pd.DataFrame, list[str]]:
    constant_cols: list[str] = []
    for column in df.columns:
        if column in ignore:
            continue
        non_null = df[column].dropna()
        if non_null.empty or non_null.nunique(dropna=True) <= 1:
            constant_cols.append(column)

    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df, constant_cols


def _cyclic_encode(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    numeric_values = pd.to_numeric(values, errors="coerce")
    angles = 2.0 * np.pi * numeric_values / float(period)
    return pd.Series(np.sin(angles), index=values.index), pd.Series(np.cos(angles), index=values.index)


def _normalize_liaison_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def make_clean_dataset(
    oncf_df: pd.DataFrame,
    liaison_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    oncf_df = _normalize_column_names(oncf_df)
    missing = [c for c in REQUIRED_ONCF_COLS if c not in oncf_df.columns]
    if missing:
        raise ValueError(f"Missing columns in oncf_data.csv: {missing}")

    missing_l = [c for c in REQUIRED_LIAISON_COLS if c not in liaison_df.columns]
    if missing_l:
        raise ValueError(f"Missing columns in Liaison.csv: {missing_l}")

    df = oncf_df.copy()
    df["_orig_row_id"] = np.arange(len(df))
    for optional_column in OPTIONAL_ONCF_COLS:
        if optional_column not in df.columns:
            df[optional_column] = pd.NA

    report: dict[str, Any] = {
        "rows_before": int(len(df)),
        "rows_after": 0,
        "dropped_constant_columns": [],
        "rows_removed_negative": 0,
        "rows_removed_duplicates": 0,
        "rows_removed_too_many_missing": 0,
        "rows_removed_missing_join": 0,
        "rows_removed_essential_missing": 0,
        "rows_removed_cold_start_clients": 0,
        "clients_removed_cold_start": 0,
        "distinct_clients": 0,
        "distinct_liaisons": 0,
        "negative_rows_detected": 0,
        "join_overlap_count": 0,
        "join_overlap_rate": 0.0,
        "join_status": "not_checked",
    }

    prov = df.copy()
    prov["_clean_action"] = "kept"
    prov["_clean_reason"] = "kept_initial"

    # --- Date parsing ---
    df["DatePaiement"] = _to_datetime(df["DatePaiement"])
    df["DateHeureDepartVoyageSegment"] = _to_datetime(df["DateHeureDepartVoyageSegment"])
    prov["DatePaiement"] = df["DatePaiement"]
    prov["DateHeureDepartVoyageSegment"] = df["DateHeureDepartVoyageSegment"]

    # --- Numeric conversion ---
    # AchteurId is intentionally excluded: it is a person identifier (label), not a
    # numeric quantity. Treating it as float would cause wrong negative detection and
    # loss of identity information.
    numeric_cols = [
        "TypeParcoursId",
        "ClassificationId",
        "ClassePhysiqueId",
        "NiveauPrixId",
        "TrainAutocarId",
        "CarteClientId",
        "PrixParLiaison",
        "NbrVoySegment",
        "DelaiAnticipation",
        "PosteVenteId",
        "TypeOperationVenteApresVenteId",
        "PrixServices",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Drop constant columns ---
    df, dropped_constant_cols = _drop_constant_columns(
        df,
        ignore={
            "CodeClient",
            "LiaisonVoyageurSegmentIdSTG",
            "TrajetAllerRetour",
            "AchteurId",
            "DatePaiement",
            "DateHeureDepartVoyageSegment",
        },
    )
    prov = prov.drop(columns=[c for c in dropped_constant_cols if c in prov.columns]) if dropped_constant_cols else prov
    report["dropped_constant_columns"] = dropped_constant_cols

    # --- Normalize and join LiaisonId ---
    df["LiaisonId"] = _normalize_liaison_id(df["LiaisonVoyageurSegmentIdSTG"])
    prov["LiaisonId"] = df["LiaisonId"]

    liaison = liaison_df[REQUIRED_LIAISON_COLS].copy()
    liaison["LiaisonId"] = _normalize_liaison_id(liaison["LiaisonId"])

    overlap = set(df["LiaisonId"].dropna()) & set(liaison["LiaisonId"].dropna())
    report["join_overlap_count"] = int(len(overlap))
    report["join_overlap_rate"] = float(len(overlap) / max(df["LiaisonId"].nunique(dropna=True), 1))

    if overlap:
        report["join_status"] = "merged"
        out = df.merge(liaison, on="LiaisonId", how="left")
    else:
        report["join_status"] = "skipped_no_overlap"
        out = df.copy()
        out["DesignationFrGareDepart"] = pd.NA
        out["DesignationFrGareArrive"] = pd.NA

    # --- Remove rows with no liaison match ---
    if "DesignationFrGareDepart" in out.columns and "DesignationFrGareArrive" in out.columns:
        join_missing_mask = out[["DesignationFrGareDepart", "DesignationFrGareArrive"]].isna().any(axis=1)
        report["rows_removed_missing_join"] = int(join_missing_mask.sum()) if overlap else 0
        missing_join_ids = out.loc[join_missing_mask, "_orig_row_id"].tolist()
        if missing_join_ids:
            prov.loc[prov["_orig_row_id"].isin(missing_join_ids), "_clean_action"] = "dropped_missing_join"
            prov.loc[prov["_orig_row_id"].isin(missing_join_ids), "_clean_reason"] = "no_liaison_match"
        out = out.loc[~join_missing_mask].copy()
    else:
        join_missing_mask = pd.Series(False, index=out.index)

    # --- Remove rows with too many missing values ---
    missing_check_columns = [
        col for col in out.columns
        if col not in {"DesignationFrGareDepart", "DesignationFrGareArrive"}
    ]
    too_many_missing_mask = out[missing_check_columns].isna().sum(axis=1) > 1
    report["rows_removed_too_many_missing"] = int(too_many_missing_mask.sum())
    too_many_missing_ids = out.loc[too_many_missing_mask, "_orig_row_id"].tolist()
    if too_many_missing_ids:
        prov.loc[prov["_orig_row_id"].isin(too_many_missing_ids), "_clean_action"] = "dropped_too_many_missing"
        prov.loc[prov["_orig_row_id"].isin(too_many_missing_ids), "_clean_reason"] = ">1_missing_excluding_join"
    out = out.loc[~too_many_missing_mask].copy()

    # --- Cancellation detection and removal ---
    # A cancellation row signals that the immediately preceding booking for the same
    # (CodeClient, TrajetAllerRetour) pair should be voided.
    #
    # Cancellation indicators:
    #   NbrVoySegment <= 0  → explicit cancellation (0 = no passengers = cancelled trip)
    #   PrixParLiaison < 0  → refund/reversal (0 is valid = 100% discount)
    #   Other numeric IDs < 0 → anomalous rows that indicate a reversal entry
    ordered = out.sort_values(
        ["CodeClient", "DateHeureDepartVoyageSegment"]
    ).reset_index(drop=True)

    cancel_strict_neg_cols = [
        col for col in [
            "TypeParcoursId",
            "ClassificationId",
            "ClassePhysiqueId",
            "NiveauPrixId",
            "TrainAutocarId",
            "CarteClientId",
            "DelaiAnticipation",
            "PosteVenteId",
            "TypeOperationVenteApresVenteId",
        ]
        if col in ordered.columns
    ]

    is_cancellation = pd.Series(False, index=ordered.index)
    if "NbrVoySegment" in ordered.columns:
        is_cancellation |= ordered["NbrVoySegment"].le(0)
    if "PrixParLiaison" in ordered.columns:
        is_cancellation |= ordered["PrixParLiaison"].lt(0)
    if cancel_strict_neg_cols:
        is_cancellation |= ordered[cancel_strict_neg_cols].lt(0).any(axis=1)

    report["negative_rows_detected"] = int(is_cancellation.sum())

    # Per-(CodeClient, TrajetAllerRetour) previous row.
    # This ensures a cancellation only voids the preceding booking for the same
    # trip identifier, and never crosses into another client's records.
    group_cols = ["CodeClient"]
    if "TrajetAllerRetour" in ordered.columns:
        group_cols.append("TrajetAllerRetour")

    group_prev_orig = ordered.groupby(group_cols, sort=False)["_orig_row_id"].shift(1)

    neg_orig_ids = set(ordered.loc[is_cancellation, "_orig_row_id"].tolist())
    prev_orig_ids = set(group_prev_orig.loc[is_cancellation].dropna().astype(int).tolist())
    indices_to_drop = neg_orig_ids | prev_orig_ids

    if neg_orig_ids:
        prov.loc[prov["_orig_row_id"].isin(list(neg_orig_ids)), "_clean_action"] = "dropped_cancellation"
        prov.loc[prov["_orig_row_id"].isin(list(neg_orig_ids)), "_clean_reason"] = "cancellation_row"
    if prev_orig_ids:
        prov.loc[prov["_orig_row_id"].isin(list(prev_orig_ids)), "_clean_action"] = "dropped_cancelled_booking"
        prov.loc[prov["_orig_row_id"].isin(list(prev_orig_ids)), "_clean_reason"] = "voided_by_following_cancellation"

    cancellation_mask = ordered["_orig_row_id"].isin(indices_to_drop)
    report["rows_removed_negative"] = int(cancellation_mask.sum())
    out = ordered.loc[~cancellation_mask].copy()

    # --- Remove rows with missing essential fields ---
    essential_missing_mask = out[["CodeClient", "LiaisonId", "DateHeureDepartVoyageSegment"]].isna().any(axis=1)
    essential_missing_ids = out.loc[essential_missing_mask, "_orig_row_id"].tolist()
    if essential_missing_ids:
        prov.loc[prov["_orig_row_id"].isin(essential_missing_ids), "_clean_action"] = "dropped_essential_missing"
        prov.loc[prov["_orig_row_id"].isin(essential_missing_ids), "_clean_reason"] = "missing_essential_fields"
    report["rows_removed_essential_missing"] = int(essential_missing_mask.sum())
    out = out.loc[~essential_missing_mask].copy()

    # --- Remove exact duplicates ---
    out = out.sort_values(["CodeClient", "DateHeureDepartVoyageSegment"]).reset_index(drop=True)
    dup_mask = out.duplicated(keep="first")
    dup_ids = out.loc[dup_mask, "_orig_row_id"].tolist()
    if dup_ids:
        prov.loc[prov["_orig_row_id"].isin(dup_ids), "_clean_action"] = "dropped_duplicate"
        prov.loc[prov["_orig_row_id"].isin(dup_ids), "_clean_reason"] = "duplicate_row"
    before_dedup = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["rows_removed_duplicates"] = int(before_dedup - len(out))

    # --- Temporal features (cyclic encoding) ---
    # Hour of day (period 24), day of week (period 7), month of year (period 12).
    # Cyclic encoding preserves the circular nature of time: hour 23 is close to hour 0,
    # December is close to January, etc. Raw integers would create artificial discontinuities.
    out["depart_hour"] = out["DateHeureDepartVoyageSegment"].dt.hour
    out["depart_dow"] = out["DateHeureDepartVoyageSegment"].dt.dayofweek
    out["depart_month"] = out["DateHeureDepartVoyageSegment"].dt.month

    out["depart_hour_sin"], out["depart_hour_cos"] = _cyclic_encode(out["depart_hour"], 24)
    out["depart_dow_sin"], out["depart_dow_cos"] = _cyclic_encode(out["depart_dow"], 7)
    out["depart_month_sin"], out["depart_month_cos"] = _cyclic_encode(out["depart_month"], 12)

    # --- Exclude cold-start clients ---
    # Clients with fewer than MIN_TRIPS_FOR_TRAINING total bookings will never receive
    # a recommendation in production (cold-start rule). Including them in the training
    # set only adds noise without contributing learnable signal.
    trip_counts = out.groupby("CodeClient").size()
    cold_start_clients = trip_counts[trip_counts < MIN_TRIPS_FOR_TRAINING].index
    cold_start_mask = out["CodeClient"].isin(cold_start_clients)

    cold_start_ids = out.loc[cold_start_mask, "_orig_row_id"].tolist()
    if cold_start_ids:
        prov.loc[prov["_orig_row_id"].isin(cold_start_ids), "_clean_action"] = "dropped_cold_start_client"
        prov.loc[prov["_orig_row_id"].isin(cold_start_ids), "_clean_reason"] = f"client_has_fewer_than_{MIN_TRIPS_FOR_TRAINING}_trips"

    report["rows_removed_cold_start_clients"] = int(cold_start_mask.sum())
    report["clients_removed_cold_start"] = int(len(cold_start_clients))
    out = out[~cold_start_mask].reset_index(drop=True)

    # --- Final report ---
    report["rows_after"] = int(len(out))
    report["distinct_clients"] = int(out["CodeClient"].nunique())
    report["distinct_liaisons"] = int(out["LiaisonId"].nunique())

    prov.loc[prov["_clean_action"] == "kept", "_clean_reason"] = "kept_in_final_dataset"

    if "_orig_row_id" in out.columns:
        out = out.drop(columns=["_orig_row_id"])

    return out, report, prov
