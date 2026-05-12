from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.popularity import build_popularity_list, save_popularity


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

    popularity = build_popularity_list(clean)
    print(f"  Top-10 by frequency: {popularity[:10]}")

    save_popularity(popularity, paths.popularity_path)
    print(f"Saved {len(popularity)} liaisons -> {paths.popularity_path}")


if __name__ == "__main__":
    main()
