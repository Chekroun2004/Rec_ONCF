"""Compute simple baselines on the same temporal split as the XGBoost model.

Baselines:
    1. most_frequent : predict the user's most frequent liaison in their training history.
    2. prev_liaison  : predict the user's previous (chronologically last) liaison.
    3. global_top    : predict the globally most frequent liaisons (popularity).

Output:
    reports/baseline_metrics.json
    Stdout : table HR@1 / HR@3 / MRR@3 for each baseline + comparison vs XGBoost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.training import temporal_split


def _hit_rate_at_k(y_true: np.ndarray, ranked_preds: np.ndarray, k: int) -> float:
    topk = ranked_preds[:, :k]
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def _mrr_at_k(y_true: np.ndarray, ranked_preds: np.ndarray, k: int) -> float:
    topk = ranked_preds[:, :k]
    rr = np.zeros(len(y_true))
    for rank in range(k):
        hit = (topk[:, rank] == y_true) & (rr == 0)
        rr[hit] = 1.0 / (rank + 1)
    return float(rr.mean())


def _pad_to_k(items: list[str], k: int, pool: list[str]) -> list[str]:
    out = list(items)[:k]
    seen = set(out)
    for p in pool:
        if len(out) >= k:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    while len(out) < k:
        out.append("__missing__")
    return out[:k]


def baseline_most_frequent(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    user_col: str,
    label_col: str,
    time_col: str,
    k: int,
    global_pool: list[str],
) -> np.ndarray:
    """For each test row, predict user's most frequent liaisons in train history,
    sorted by frequency desc then recency desc, padded with global popularity.
    """
    grp = (
        train_df.sort_values(time_col)
        .groupby(user_col, sort=False)
    )
    user_pred: dict[str, list[str]] = {}
    for uid, sub in grp:
        freq = sub[label_col].value_counts()
        last = sub.groupby(label_col)[time_col].max()
        score = pd.DataFrame({"freq": freq, "last": last}).fillna(0)
        score = score.sort_values(["freq", "last"], ascending=[False, False])
        user_pred[str(uid)] = score.index.astype(str).tolist()

    preds = np.empty((len(test_df), k), dtype=object)
    test_users = test_df[user_col].astype(str).to_numpy()
    for i, uid in enumerate(test_users):
        items = user_pred.get(uid, [])
        preds[i] = _pad_to_k(items, k, global_pool)
    return preds


def baseline_prev_liaison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    user_col: str,
    label_col: str,
    time_col: str,
    k: int,
    global_pool: list[str],
) -> np.ndarray:
    """Predict the user's last seen liaison in train history (single-class baseline).
    Padded with global popularity to reach k slots.
    """
    last_per_user = (
        train_df.sort_values(time_col)
        .groupby(user_col, sort=False)[label_col]
        .last()
        .astype(str)
        .to_dict()
    )
    preds = np.empty((len(test_df), k), dtype=object)
    test_users = test_df[user_col].astype(str).to_numpy()
    for i, uid in enumerate(test_users):
        last = last_per_user.get(uid)
        items = [last] if last is not None else []
        preds[i] = _pad_to_k(items, k, global_pool)
    return preds


def baseline_global_top(
    test_df: pd.DataFrame,
    *,
    k: int,
    global_pool: list[str],
) -> np.ndarray:
    top_k = global_pool[:k]
    preds = np.empty((len(test_df), k), dtype=object)
    for i in range(len(test_df)):
        preds[i] = top_k
    return preds


def main() -> None:
    paths = default_paths()
    if not paths.features_parquet.exists():
        raise FileNotFoundError(
            f"Missing: {paths.features_parquet}. Run scripts/02_build_features.py"
        )

    df = read_parquet(paths.features_parquet)

    label_col = "LiaisonId"
    time_col = "DateHeureDepartVoyageSegment"
    user_col = "CodeClient"

    df[label_col] = df[label_col].astype(str)
    df[user_col] = df[user_col].astype(str)

    train_df, test_df = temporal_split(df, time_col=time_col, train_frac=0.8)

    # Restrict test set to labels seen in train (same protocol as XGBoost eval).
    known = set(train_df[label_col].unique())
    n_test_before = len(test_df)
    test_df = test_df[test_df[label_col].isin(known)].reset_index(drop=True)
    n_test_dropped = n_test_before - len(test_df)

    # Global popularity (computed on train only, no leakage).
    global_pool = (
        train_df[label_col]
        .value_counts()
        .index.astype(str)
        .tolist()
    )

    y_true = test_df[label_col].astype(str).to_numpy()
    K = 3

    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}  "
          f"(dropped {n_test_dropped} rows with unseen labels)")
    print(f"Distinct liaisons in pool: {len(global_pool):,}")
    print()

    results: dict[str, dict] = {}

    # --- Baseline 1: most_frequent -----------------------------------
    preds = baseline_most_frequent(
        train_df, test_df,
        user_col=user_col, label_col=label_col, time_col=time_col,
        k=K, global_pool=global_pool,
    )
    results["most_frequent"] = {
        "hit_rate@1": _hit_rate_at_k(y_true, preds, 1),
        "hit_rate@3": _hit_rate_at_k(y_true, preds, 3),
        "mrr@3":      _mrr_at_k(y_true, preds, 3),
    }

    # --- Baseline 2: prev_liaison ------------------------------------
    preds = baseline_prev_liaison(
        train_df, test_df,
        user_col=user_col, label_col=label_col, time_col=time_col,
        k=K, global_pool=global_pool,
    )
    results["prev_liaison"] = {
        "hit_rate@1": _hit_rate_at_k(y_true, preds, 1),
        "hit_rate@3": _hit_rate_at_k(y_true, preds, 3),
        "mrr@3":      _mrr_at_k(y_true, preds, 3),
    }

    # --- Baseline 3: global_top --------------------------------------
    preds = baseline_global_top(test_df, k=K, global_pool=global_pool)
    results["global_top"] = {
        "hit_rate@1": _hit_rate_at_k(y_true, preds, 1),
        "hit_rate@3": _hit_rate_at_k(y_true, preds, 3),
        "mrr@3":      _mrr_at_k(y_true, preds, 3),
    }

    # --- Reference: XGBoost (loaded from previous run) ---------------
    xgb_path = paths.project_root / "reports" / "offline_metrics.json"
    if xgb_path.exists():
        xgb_metrics = json.loads(xgb_path.read_text(encoding="utf-8"))
        results["xgboost_multiclass"] = {
            "hit_rate@1": xgb_metrics.get("hit_rate@1"),
            "hit_rate@3": xgb_metrics.get("hit_rate@3"),
            "mrr@3":      xgb_metrics.get("mrr@3"),
        }

    # --- Output ------------------------------------------------------
    summary = {
        "rows_train": int(len(train_df)),
        "rows_test":  int(len(test_df)),
        "rows_test_unseen_dropped": int(n_test_dropped),
        "classes": int(len(global_pool)),
        "models": results,
    }

    out_path = paths.project_root / "reports" / "baseline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'Model':<22} {'HR@1':>8} {'HR@3':>8} {'MRR@3':>8}")
    print("-" * 50)
    for name, m in results.items():
        hr1 = m.get("hit_rate@1") or 0.0
        hr3 = m.get("hit_rate@3") or 0.0
        mrr = m.get("mrr@3") or 0.0
        print(f"{name:<22} {hr1:>8.4f} {hr3:>8.4f} {mrr:>8.4f}")

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
