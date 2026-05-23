from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    # Raw source CSVs (data/raw/)
    raw_dir: Path
    raw_oncf_data: Path
    raw_liaison: Path
    horaire_csv_path: Path

    # Cleaned parquets (data/clean/)
    clean_dir: Path
    processed_dataset_parquet: Path   # data/clean/oncf_clean.parquet

    # Feature parquets (data/features/)
    features_dir: Path
    features_parquet: Path            # data/features/oncf_features.parquet

    # Models
    models_dir: Path
    xgb_model_path: Path
    label_encoder_path: Path
    cold_start_path: Path
    onnx_model_path: Path
    popularity_path: Path
    schedule_index_path: Path


def default_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw"
    clean_dir = project_root / "data" / "clean"
    features_dir = project_root / "data" / "features"

    models_dir = project_root / "models"

    return Paths(
        project_root=project_root,
        raw_dir=raw_dir,
        raw_oncf_data=raw_dir / "oncf_data.csv",
        raw_liaison=raw_dir / "Liaison.csv",
        horaire_csv_path=raw_dir / "horaire.csv",
        clean_dir=clean_dir,
        processed_dataset_parquet=clean_dir / "parquet" / "oncf_clean.parquet",
        features_dir=features_dir,
        features_parquet=features_dir / "parquet" / "oncf_features.parquet",
        models_dir=models_dir,
        xgb_model_path=models_dir / "xgb_ranker.json",
        label_encoder_path=models_dir / "label_encoder.joblib",
        cold_start_path=models_dir / "cold_start.joblib",
        onnx_model_path=models_dir / "xgb_ranker.onnx",
        popularity_path=models_dir / "popularity.joblib",
        schedule_index_path=models_dir / "schedule_index.joblib",
    )
