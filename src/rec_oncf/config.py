from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    desktop: Path
    raw_oncf_data: Path
    raw_liaison: Path

    processed_dir: Path
    processed_dataset_parquet: Path
    processed_dataset_csv: Path
    features_parquet: Path

    models_dir: Path
    xgb_model_path: Path
    label_encoder_path: Path
    cold_start_path: Path
    onnx_model_path: Path
    popularity_path: Path
    horaire_csv_path: Path
    schedule_index_path: Path


def default_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[2]
    desktop = Path.home() / "Desktop"

    raw_oncf_data = desktop / "oncf_data.csv"
    raw_liaison = desktop / "Liaison.csv"

    processed_dir = project_root / "data" / "processed"
    processed_dataset_parquet = processed_dir / "oncf_clean.parquet"
    processed_dataset_csv = processed_dir / "oncf_clean.csv"
    features_parquet = processed_dir / "features.parquet"

    models_dir = project_root / "models"
    xgb_model_path = models_dir / "xgb_ranker.json"
    label_encoder_path = models_dir / "label_encoder.joblib"
    cold_start_path = models_dir / "cold_start.joblib"
    onnx_model_path = models_dir / "xgb_ranker.onnx"
    popularity_path = models_dir / "popularity.joblib"
    horaire_csv_path    = desktop / "horaire.csv"
    schedule_index_path = models_dir / "schedule_index.joblib"

    return Paths(
        project_root=project_root,
        desktop=desktop,
        raw_oncf_data=raw_oncf_data,
        raw_liaison=raw_liaison,
        processed_dir=processed_dir,
        processed_dataset_parquet=processed_dataset_parquet,
        processed_dataset_csv=processed_dataset_csv,
        features_parquet=features_parquet,
        models_dir=models_dir,
        xgb_model_path=xgb_model_path,
        label_encoder_path=label_encoder_path,
        cold_start_path=cold_start_path,
        onnx_model_path=onnx_model_path,
        popularity_path=popularity_path,
        horaire_csv_path=horaire_csv_path,
        schedule_index_path=schedule_index_path,
    )
