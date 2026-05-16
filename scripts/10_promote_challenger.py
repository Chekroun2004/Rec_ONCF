"""Promote the challenger model to production.

Steps:
  1. Verify challenger files exist and metrics are available.
  2. Archive current prod files to models/archive/<timestamp>/.
  3. Copy challenger files -> prod files.
  4. Print a promotion summary.

Run after 09_train_challenger.py has completed and the A/B test outcome
is favourable (challenger HR@1 >= prod HR@1).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths


def main() -> None:
    paths = default_paths()

    challenger_model = paths.models_dir / "xgb_ranker_challenger.json"
    challenger_le    = paths.models_dir / "label_encoder_challenger.joblib"
    challenger_onnx  = paths.models_dir / "xgb_ranker_challenger.onnx"
    challenger_meta  = paths.models_dir / "xgb_ranker_challenger.meta.json"

    # -----------------------------------------------------------------------
    # Verify challenger exists
    # -----------------------------------------------------------------------
    missing = [f for f in [challenger_model, challenger_le, challenger_onnx] if not f.exists()]
    if missing:
        print("ERROR: challenger files missing — run scripts/09_train_challenger.py first.")
        for f in missing:
            print(f"  Missing: {f}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load and compare metrics
    # -----------------------------------------------------------------------
    report_path = PROJECT_ROOT / "reports" / "challenger_metrics.json"
    prod_path   = PROJECT_ROOT / "reports" / "offline_metrics.json"

    challenger_hr1 = prod_hr1 = None
    if report_path.exists():
        r = json.loads(report_path.read_text(encoding="utf-8"))
        challenger_hr1 = r.get("hit_rate@1")
        prod_hr1       = r.get("prod_hit_rate@1")

    print("=== Promotion Summary ===")
    if challenger_hr1 is not None and prod_hr1 is not None:
        delta = (challenger_hr1 - prod_hr1) * 100
        verdict = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "EQUAL"
        print(f"Challenger HR@1 : {challenger_hr1:.4f}")
        print(f"Prod HR@1       : {prod_hr1:.4f}")
        print(f"Delta           : {delta:+.2f} pp  -> {verdict}")
        if delta < 0:
            print("\nWARNING: challenger is worse than prod on HR@1.")
            answer = input("Promote anyway? [y/N] ").strip().lower()
            if answer != "y":
                print("Promotion cancelled.")
                sys.exit(0)
    else:
        print("(metrics not available — proceeding with promotion)")

    # -----------------------------------------------------------------------
    # Archive current prod files
    # -----------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = paths.models_dir / "archive" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)

    prod_files = [
        paths.xgb_model_path,
        paths.label_encoder_path,
        paths.onnx_model_path,
        paths.xgb_model_path.with_suffix(".meta.json"),
    ]
    archived = []
    for f in prod_files:
        if f.exists():
            dest = archive_dir / f.name
            shutil.copy2(f, dest)
            archived.append(f.name)

    print(f"\nArchived {len(archived)} prod file(s) -> models/archive/{ts}/")
    for name in archived:
        print(f"  {name}")

    # -----------------------------------------------------------------------
    # Promote challenger -> prod
    # -----------------------------------------------------------------------
    promotions = [
        (challenger_model, paths.xgb_model_path),
        (challenger_le,    paths.label_encoder_path),
        (challenger_onnx,  paths.onnx_model_path),
    ]
    if challenger_meta.exists():
        promotions.append((challenger_meta, paths.xgb_model_path.with_suffix(".meta.json")))

    print("\nPromoting challenger -> prod ...")
    for src, dst in promotions:
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {src.name} -> {dst.name}  ({size_mb:.1f} MB)")

    # Update prod metrics report
    if report_path.exists() and prod_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prod   = json.loads(prod_path.read_text(encoding="utf-8"))
        prod["hit_rate@1"] = report["hit_rate@1"]
        prod["hit_rate@3"] = report["hit_rate@3"]
        prod["mrr@3"]      = report["mrr@3"]
        prod["promoted_from_challenger_at_utc"] = ts
        prod_path.write_text(json.dumps(prod, indent=2), encoding="utf-8")
        print(f"\nUpdated {prod_path.name} with challenger metrics.")

    print("\nDone. Restart the API to load the promoted model:")
    print("  .venv\\Scripts\\python.exe -m uvicorn apps.api.main:app --reload")


if __name__ == "__main__":
    main()
