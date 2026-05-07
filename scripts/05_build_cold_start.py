from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cold_start import build_cold_start_recommender, save_cold_start
from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet


def main() -> None:
    paths = default_paths()

    if not paths.processed_dataset_parquet.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {paths.processed_dataset_parquet}\n"
            "Run scripts/01_make_dataset.py first."
        )

    print("Loading oncf_clean.parquet ...")
    clean = read_parquet(paths.processed_dataset_parquet)
    print(f"  {len(clean):,} rows, {clean['LiaisonId'].nunique()} unique routes")

    print("Building co-occurrence lookup ...")
    rec = build_cold_start_recommender(clean)
    print(f"  {len(rec.cooccurrence)} routes with co-occurrence data")
    print(f"  Global top-3: {rec.global_top[:3]}")

    save_cold_start(rec, paths.cold_start_path)
    print(f"Saved -> {paths.cold_start_path}")


if __name__ == "__main__":
    main()
