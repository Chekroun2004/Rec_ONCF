# scripts/11_build_schedule_index.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.local_schedule import build_od_index, parse_horaire_csv, save_schedule_index
from rec_oncf.schedule import build_liaison_station_map


def main() -> None:
    paths = default_paths()

    if not paths.horaire_csv_path.exists():
        raise FileNotFoundError(
            f"horaire.csv not found: {paths.horaire_csv_path}\n"
            "Place the file on the Desktop."
        )

    print(f"Loading {paths.horaire_csv_path}...")
    df = parse_horaire_csv(paths.horaire_csv_path)
    n_trains = df["num_commercial"].nunique()
    n_gares = df["gare"].nunique()
    print(f"  {len(df)} stops  |  {n_trains} trains  |  {n_gares} distinct stations")

    print("Building O/D index...")
    od_index = build_od_index(df)
    print(f"  {len(od_index)} O/D pairs generated")

    paths.models_dir.mkdir(exist_ok=True)
    save_schedule_index(od_index, paths.schedule_index_path)
    print(f"Index saved -> {paths.schedule_index_path}")

    if paths.processed_dataset_parquet.exists():
        clean = read_parquet(paths.processed_dataset_parquet)
        liaison_map = build_liaison_station_map(clean)
        covered = sum(
            1
            for (o, d) in liaison_map.values()
            if (o.strip().upper(), d.strip().upper()) in od_index
        )
        print(f"Coverage: {covered}/{len(liaison_map)} LiaisonIds covered by index")
    else:
        print("(oncf_clean.parquet not found — coverage not computed)")


if __name__ == "__main__":
    main()
