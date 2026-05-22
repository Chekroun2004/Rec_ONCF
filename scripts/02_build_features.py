from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.features import build_training_rows
from rec_oncf.io import read_parquet, write_parquet


def main(argv: list[str] | None = None) -> None:
    paths = default_paths()

    parser = argparse.ArgumentParser(
        description="Build the model-ready feature table from a cleaned parquet. "
        "Defaults reproduce the oncf_data pipeline; pass --input/--output to run "
        "the SAME feature engineering on retrain data (e.g. test1_clean.parquet)."
    )
    parser.add_argument(
        "--input", type=Path, default=paths.processed_dataset_parquet,
        help="Cleaned parquet (default: data/processed/oncf_clean.parquet)",
    )
    parser.add_argument(
        "--output", type=Path, default=paths.features_parquet,
        help="Features parquet output (default: data/processed/features.parquet)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {args.input}. Run scripts/01_make_dataset.py first."
        )

    clean = read_parquet(args.input)
    feats = build_training_rows(clean)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(feats, args.output)

    print(f"Input:  {args.input}")
    print(f"Wrote: {args.output}")
    print(f"Rows: {len(feats):,}")
    print(f"Users: {feats['CodeClient'].nunique():,}")
    print(f"Classes (liaisons): {feats['LiaisonId'].nunique():,}")


if __name__ == "__main__":
    main()
