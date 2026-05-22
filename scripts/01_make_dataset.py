from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cleaning import make_clean_dataset
from rec_oncf.config import default_paths
from rec_oncf.io import read_csv, write_csv, write_parquet


def main(argv: list[str] | None = None) -> None:
    paths = default_paths()

    parser = argparse.ArgumentParser(
        description="Clean a raw bookings CSV into a model-ready parquet. "
        "Defaults reproduce the oncf_data pipeline; pass --input/--output to "
        "run the SAME pipeline on retrain data (e.g. test1.csv)."
    )
    parser.add_argument(
        "--input", type=Path, default=paths.raw_oncf_data,
        help="Raw bookings CSV (default: Desktop/oncf_data.csv)",
    )
    parser.add_argument(
        "--liaison", type=Path, default=paths.raw_liaison,
        help="Liaison lookup CSV (default: Desktop/Liaison.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=paths.processed_dataset_parquet,
        help="Cleaned parquet output (default: data/processed/oncf_clean.parquet)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")
    if not args.liaison.exists():
        raise FileNotFoundError(f"Missing liaison CSV: {args.liaison}")

    out_parquet = args.output
    is_default = out_parquet == paths.processed_dataset_parquet
    reports_dir = paths.project_root / "reports"

    # Preserve exact default filenames; derive sibling names from the output
    # stem for any non-default (retrain) run so nothing clobbers oncf artifacts.
    if is_default:
        out_csv = paths.processed_dataset_csv
        report_path = reports_dir / "cleaning_report.json"
        prov_path = reports_dir / "cleaning_provenance.parquet"
    else:
        out_csv = out_parquet.with_suffix(".csv")
        report_path = reports_dir / f"{out_parquet.stem}_cleaning_report.json"
        prov_path = reports_dir / f"{out_parquet.stem}_cleaning_provenance.parquet"

    oncf = read_csv(args.input)
    liaison = read_csv(args.liaison)

    clean, report, provenance = make_clean_dataset(oncf, liaison)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(clean, out_parquet)
    write_csv(clean, out_csv)

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(provenance, prov_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input:  {args.input}")
    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(clean):,}")
    print(f"Distinct liaisons: {clean['LiaisonId'].nunique():,}")
    print(f"Report: {report_path}")
    print(f"Provenance: {prov_path}")


if __name__ == "__main__":
    main()
