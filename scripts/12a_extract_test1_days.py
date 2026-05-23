from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.config import default_paths
from rec_oncf.extract_days import extract_last_n_days
from rec_oncf.io import read_csv, write_csv


def main() -> None:
    paths = default_paths()
    test1_path = paths.raw_dir / "test1.csv"
    if not test1_path.exists():
        raise FileNotFoundError(f"Missing: {test1_path}")

    print(f"Loading {test1_path}...")
    df = read_csv(test1_path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    base, days = extract_last_n_days(df, n=7, date_col="DateHeureDepartVoyageSegment")

    out_dir = PROJECT_ROOT / "data" / "raw" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = PROJECT_ROOT / "data" / "raw" / "test1_base.csv"

    write_csv(base, base_path, sep=",")
    print(f"  base: {len(base):,} rows -> {base_path.relative_to(PROJECT_ROOT)}")

    for day, day_df in days.items():
        day_path = out_dir / f"test1_day_{day}.csv"
        write_csv(day_df, day_path, sep=",")
        print(f"  {day}: {len(day_df):,} rows -> {day_path.relative_to(PROJECT_ROOT)}")

    total_check = len(base) + sum(len(d) for d in days.values())
    print(f"\nTotal preserved: {total_check:,} == {len(df):,} ? {total_check == len(df)}")


if __name__ == "__main__":
    main()
