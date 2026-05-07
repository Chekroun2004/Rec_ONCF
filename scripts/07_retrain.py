from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.retrain import retrain_pipeline


def _print_report(report: dict) -> None:
    cur = report["current_metrics"]
    new = report["new_metrics"]
    print()
    print("=" * 60)
    print("RETRAINING REPORT")
    print("=" * 60)
    print()
    print("Current model:")
    if cur:
        print(f"  hit_rate@1 = {cur.get('hit_rate@1', 0):.4f}")
        print(f"  hit_rate@3 = {cur.get('hit_rate@3', 0):.4f}")
        print(f"  mrr@3      = {cur.get('mrr@3', 0):.4f}")
    else:
        print("  (no existing model — first run)")
    print()
    print("New model:")
    print(f"  hit_rate@1 = {new.get('hit_rate@1', 0):.4f}")
    print(f"  hit_rate@3 = {new.get('hit_rate@3', 0):.4f}")
    print(f"  mrr@3      = {new.get('mrr@3', 0):.4f}")
    print(f"  test_rows  = {new.get('test_rows', 0)}")
    print()
    symbol = "OK" if report["guardrail_passes"] else "BLOCKED"
    print(f"Guardrail : {symbol}")
    print(f"Reason    : {report['guardrail_reason']}")
    print()
    if report["dry_run"]:
        print("DRY RUN — models/ not changed.")
    elif report["promoted"]:
        print("New model promoted to models/")
    else:
        print("Promotion blocked. models/ unchanged.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the ONCF recommender with a KPI guardrail."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrain and evaluate but do not overwrite models/.",
    )
    args = parser.parse_args()

    paths = default_paths()
    if not paths.features_parquet.exists():
        raise FileNotFoundError(
            f"Missing features: {paths.features_parquet}. Run scripts/02_build_features.py first."
        )
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Starting retrain pipeline (this will take ~43 min on CPU)...")
    report = retrain_pipeline(paths, dry_run=args.dry_run)
    _print_report(report)
    sys.exit(0 if report["guardrail_passes"] else 1)


if __name__ == "__main__":
    main()
