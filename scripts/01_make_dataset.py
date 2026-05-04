from __future__ import annotations

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


def main() -> None:
    paths = default_paths()

    if not paths.raw_oncf_data.exists():
        raise FileNotFoundError(f"Missing file on Desktop: {paths.raw_oncf_data}")
    if not paths.raw_liaison.exists():
        raise FileNotFoundError(f"Missing file on Desktop: {paths.raw_liaison}")

    oncf = read_csv(paths.raw_oncf_data)
    liaison = read_csv(paths.raw_liaison)

    clean, report, provenance = make_clean_dataset(oncf, liaison)

    write_parquet(clean, paths.processed_dataset_parquet)
    write_csv(clean, paths.processed_dataset_csv)
    # write provenance report (parquet) for per-row cleaning decisions
    prov_path = paths.project_root / "reports" / "cleaning_provenance.parquet"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(provenance, prov_path)
    report_path = paths.project_root / "reports" / "cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {paths.processed_dataset_parquet}")
    print(f"Wrote: {paths.processed_dataset_csv}")
    print(f"Rows: {len(clean):,}")
    print(f"Distinct liaisons: {clean['LiaisonId'].nunique():,}")
    print(f"Report: {report_path}")
    print(f"Provenance: {prov_path}")


if __name__ == "__main__":
    main()
