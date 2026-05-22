# scripts/12_simulate_daily_retrain.py
"""Daily-retrain simulation with two datasets.

Phase A (baseline) trains on test1 only (data/processed/test1_features.parquet).
Phase B (daily) trains on the combined oncf+test1 universe
(data/processed/oncf_full_features.parquet). NOTHING here touches the prod model
served by the API; all artifacts go to models/sim/.

    --baseline   train on test1 minus the 7 simulation days  (Phase A)
    --day N      365-day sliding window ending at sim-day N, eval on day N+1 (Phase B)

The actual XGBoost training is the heavy step; run these last.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.popularity import load_popularity
from rec_oncf.recommender import Recommender
from rec_oncf.retrain import check_guardrail, evaluate_model, load_current_metrics
from rec_oncf.simulation import (
    baseline_frame,
    day_frame,
    eval_on_next_day,
    filter_sliding_window,
    history_through,
    last_n_dates,
    log_simulation_entry,
)
from rec_oncf.training import (
    build_metadata,
    fingerprint_dataframe,
    save_artifacts,
    temporal_split,
    train_xgb_multiclass,
)

WINDOW_DAYS = 365
N_SIM_DAYS = 7
# Minimum bookings/day for a date to count as a simulation day. test1 trails off
# into single-booking advance reservations after ~Dec 2021; this keeps the 7
# simulation days on real high-volume dates (2021-12-20 .. 2021-12-31).
MIN_DAY_BOOKINGS = 200
LABEL_COL = "LiaisonId"
TIME_COL = "DateHeureDepartVoyageSegment"

# Challenger hyperparameters = current prod reference (2026-05-16).
CHALLENGER = dict(
    n_estimators=250,
    learning_rate=0.06,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=1.5,
)

TEST1_CLEAN = PROJECT_ROOT / "data" / "processed" / "test1_clean.parquet"
TEST1_FEATURES = PROJECT_ROOT / "data" / "processed" / "test1_features.parquet"
FULL_CLEAN = PROJECT_ROOT / "data" / "processed" / "oncf_full_clean.parquet"
FULL_FEATURES = PROJECT_ROOT / "data" / "processed" / "oncf_full_features.parquet"
LOG_PATH = PROJECT_ROOT / "reports" / "simulation_daily.json"
SIM_ROOT = PROJECT_ROOT / "models" / "sim"
BASELINE_META = SIM_ROOT / "baseline" / "xgb_ranker.meta.json"


def _require_baseline_universe() -> None:
    if not TEST1_FEATURES.exists():
        raise FileNotFoundError(
            f"{TEST1_FEATURES} missing. Build test1 features first:\n"
            f"  python scripts/01_make_dataset.py --input <test1.csv> --output {TEST1_CLEAN}\n"
            f"  python scripts/02_build_features.py --input {TEST1_CLEAN} --output {TEST1_FEATURES}"
        )


def _require_full_universe() -> None:
    if not FULL_FEATURES.exists() or not FULL_CLEAN.exists():
        raise FileNotFoundError(
            f"Missing full-universe data. Expected:\n  {FULL_CLEAN}\n  {FULL_FEATURES}\n"
            "Build the universe first:\n"
            "  python scripts/01_make_dataset.py --input <oncf_data.csv> "
            "--extra-csv <test1.csv> --output data/processed/oncf_full_clean.parquet\n"
            "  python scripts/02_build_features.py --input data/processed/oncf_full_clean.parquet "
            "--output data/processed/oncf_full_features.parquet"
        )


def _train_and_eval(train_window):
    """Train on the 80% temporal split, evaluate on the 20% test split."""
    df_train, _ = temporal_split(train_window, time_col=TIME_COL)
    arts = train_xgb_multiclass(df_train, label_col=LABEL_COL, time_col=TIME_COL, **CHALLENGER)
    metrics = evaluate_model(arts, train_window)
    return arts, df_train, metrics


def _save(arts, df_train, metrics, train_window, out_dir: Path) -> None:
    paths = default_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(
        arts,
        train_rows=len(df_train),
        test_rows=metrics["test_rows"],
        metrics={k: v for k, v in metrics.items() if k != "test_rows"},
        dataset_fingerprint=fingerprint_dataframe(train_window),
    )
    save_artifacts(
        arts,
        model_path=out_dir / paths.xgb_model_path.name,
        label_encoder_path=out_dir / paths.label_encoder_path.name,
        metadata=metadata,
    )


def run_baseline() -> None:
    _require_baseline_universe()
    features = read_parquet(TEST1_FEATURES)
    sim_dates = last_n_dates(features, n=N_SIM_DAYS, min_count=MIN_DAY_BOOKINGS)
    train_window = baseline_frame(features, sim_dates=sim_dates)

    print("=== BASELINE (Phase A) — test1 minus 7 jours ===")
    print(f"  Jours de simulation exclus : {sim_dates[0]} .. {sim_dates[-1]}")
    print(f"  Lignes d'entraînement : {len(train_window):,}")
    t0 = time.time()
    arts, df_train, metrics = _train_and_eval(train_window)
    _save(arts, df_train, metrics, train_window, SIM_ROOT / "baseline")
    dur = time.time() - t0
    print(f"  HR@1={metrics['hit_rate@1']:.4f} HR@3={metrics['hit_rate@3']:.4f} "
          f"MRR@3={metrics['mrr@3']:.4f} (test_rows={metrics['test_rows']:,})")
    print(f"=== DONE baseline — {dur:.0f}s — modèle: {SIM_ROOT / 'baseline'} ===")


def run_day(n: int) -> None:
    _require_full_universe()
    if n < 1 or n > N_SIM_DAYS:
        raise ValueError(f"--day must be in [1, {N_SIM_DAYS}], got {n}")

    features = read_parquet(FULL_FEATURES)
    sim_dates = last_n_dates(features, n=N_SIM_DAYS, min_count=MIN_DAY_BOOKINGS)
    day_date = sim_dates[n - 1]
    next_date = sim_dates[n] if n < N_SIM_DAYS else None

    print(f"=== JOUR {n} ({day_date}) — fenêtre glissante {WINDOW_DAYS}j ===")
    window = filter_sliding_window(features, end_date=day_date, window_days=WINDOW_DAYS)
    print(f"  Fenêtre [{day_date} - {WINDOW_DAYS}j ; {day_date}] : {len(window):,} lignes")
    if len(window) < 10_000:
        print(f"  ATTENTION : seulement {len(window)} lignes dans la fenêtre.")

    t0 = time.time()
    arts, df_train, metrics = _train_and_eval(window)
    _save(arts, df_train, metrics, window, SIM_ROOT / f"day_{n}")
    print(f"  Split interne : HR@1={metrics['hit_rate@1']:.4f} "
          f"HR@3={metrics['hit_rate@3']:.4f} MRR@3={metrics['mrr@3']:.4f}")

    next_day_metrics = None
    if next_date is not None:
        clean = read_parquet(FULL_CLEAN)
        history = history_through(clean, date=day_date)  # <= D : pas de fuite du futur
        paths = default_paths()
        popularity = load_popularity(paths.popularity_path) if paths.popularity_path.exists() else []
        rec = Recommender.from_data(arts, history, popularity=popularity)
        next_clean = day_frame(clean, date=next_date)
        next_day_metrics = eval_on_next_day(rec, next_clean)
        print(f"  Éval J+1 ({next_date}) : HR@1={next_day_metrics['hr@1']:.4f} "
              f"HR@3={next_day_metrics['hr@3']:.4f} MRR@3={next_day_metrics['mrr@3']:.4f} "
              f"(n_eval={next_day_metrics['n_eval']})")
    else:
        print("  Pas de J+1 (dernier jour) — éval next-day ignorée.")

    baseline_metrics = load_current_metrics(BASELINE_META)
    passes, reason = check_guardrail(baseline_metrics, metrics, threshold=0.05)
    dur = time.time() - t0

    log_simulation_entry(LOG_PATH, {
        "day": n,
        "date": day_date,
        "window_days": WINDOW_DAYS,
        "train_rows": len(df_train),
        "internal_split_metrics": {
            "hr@1": metrics["hit_rate@1"],
            "hr@3": metrics["hit_rate@3"],
            "mrr@3": metrics["mrr@3"],
            "test_rows": metrics["test_rows"],
        },
        "next_day_metrics": next_day_metrics,
        "duration_seconds": round(dur, 1),
        "guardrail_passes": passes,
        "guardrail_reason": reason,
        "model_dir": str((SIM_ROOT / f"day_{n}").relative_to(PROJECT_ROOT)),
    })
    print(f"=== DONE jour {n} — {dur:.0f}s — guardrail: {'OK' if passes else 'FLAG'} — {reason} ===")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true", help="Phase A: train 2018->2022 minus 7 days")
    group.add_argument("--day", type=int, help=f"Phase B: simulate day N in [1, {N_SIM_DAYS}]")
    args = parser.parse_args(argv)

    if args.baseline:
        run_baseline()
    else:
        run_day(args.day)


if __name__ == "__main__":
    main()
