from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cleaning import _normalize_column_names, _to_datetime, make_clean_dataset
from rec_oncf.config import default_paths
from rec_oncf.io import read_csv, write_csv, write_parquet

# Date columns must be parsed per-file before concatenation: oncf_data is M/D/Y
# but test1 is D/M/Y. Parsing the concatenated raw strings with one heuristic
# silently corrupts whichever convention loses the dayfirst vote.
_DATE_COLS = ("DatePaiement", "DateHeureDepartVoyageSegment")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column aliases and parse date columns with this file's own
    convention (so each file keeps its M/D/Y or D/M/Y interpretation)."""
    df = _normalize_column_names(df)
    for col in _DATE_COLS:
        if col in df.columns:
            df = df.copy()
            df[col] = _to_datetime(df[col])
    return df


def load_and_concat(main_path: Path, extra_path: Path | None) -> pd.DataFrame:
    """Load the main CSV and optionally append an extra CSV before cleaning.

    Each file's columns are alias-normalized and its date columns parsed with the
    file's own convention BEFORE concatenation, so the oncf (M/D/Y) and test1
    (D/M/Y) date formats never clobber each other. A real schema mismatch raises
    ValueError. extra_path=None returns the (prepared) main CSV.
    """
    main = _prepare(read_csv(main_path))
    if extra_path is None:
        return main
    extra = _prepare(read_csv(extra_path))
    if set(extra.columns) != set(main.columns):
        only_main = sorted(set(main.columns) - set(extra.columns))
        only_extra = sorted(set(extra.columns) - set(main.columns))
        raise ValueError(
            f"schema mismatch between {main_path.name} and {extra_path.name}: "
            f"only_in_main={only_main}, only_in_extra={only_extra}"
        )
    extra = extra[main.columns.tolist()]
    return pd.concat([main, extra], ignore_index=True)


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
    parser.add_argument(
        "--extra-csv", type=Path, default=None,
        help="Optional CSV (same schema) concatenated to --input before cleaning, "
             "e.g. test1.csv to build the combined 2018->2022 universe.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input}")
    if not args.liaison.exists():
        raise FileNotFoundError(f"Missing liaison CSV: {args.liaison}")
    if args.extra_csv is not None and not args.extra_csv.exists():
        raise FileNotFoundError(f"Missing extra CSV: {args.extra_csv}")

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

    oncf = load_and_concat(args.input, args.extra_csv)
    liaison = read_csv(args.liaison)
    print(f"Loaded {len(oncf):,} input rows ({len(oncf.columns)} columns)")

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
