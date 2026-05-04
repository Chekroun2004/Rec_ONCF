from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.features import build_training_rows
from rec_oncf.io import read_parquet, write_parquet


def main() -> None:
    paths = default_paths()

    if not paths.processed_dataset_parquet.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {paths.processed_dataset_parquet}. Run scripts/01_make_dataset.py"
        )

    clean = read_parquet(paths.processed_dataset_parquet)
    feats = build_training_rows(clean)
    write_parquet(feats, paths.features_parquet)

    print(f"Wrote: {paths.features_parquet}")
    print(f"Rows: {len(feats):,}")
    print(f"Users: {feats['CodeClient'].nunique():,}")
    print(f"Classes (liaisons): {feats['LiaisonId'].nunique():,}")


if __name__ == "__main__":
    main()
