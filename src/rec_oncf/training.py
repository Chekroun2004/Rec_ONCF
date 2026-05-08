from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    label_encoder: LabelEncoder


def temporal_split(df: pd.DataFrame, *, time_col: str, train_frac: float = 0.8):
    df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_xgb_multiclass(df_train: pd.DataFrame, *, label_col: str, time_col: str) -> TrainArtifacts:
    df_train = df_train.sort_values(time_col)

    y_raw = df_train[label_col].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    _drop = [c for c in [label_col, time_col, "CodeClient"] if c in df_train.columns]
    X = df_train.drop(columns=_drop)

    # Identify columns — robust to pandas 2 (object) and pandas 3 (StringDtype)
    cat_cols = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # OrdinalEncoder keeps feature matrix at ~23 cols; OHE would explode to 5000+
    # due to high-cardinality prev_liaison (1067 unique values)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    return TrainArtifacts(pipeline=pipe, label_encoder=le)


def predict_proba(artifacts: TrainArtifacts, df: pd.DataFrame, *, label_col: str):
    drop = [c for c in [label_col, "CodeClient"] if c in df.columns]
    drop += [c for c in df.columns if df[c].dtype.kind == "M"]  # drop all datetime cols
    X = df.drop(columns=drop)
    return artifacts.pipeline.predict_proba(X)


def _safe_pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _model_hyperparams(pipeline: Pipeline) -> dict:
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return {}
    keys = (
        "objective", "eval_metric", "tree_method", "device",
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "reg_lambda",
        "n_jobs", "random_state",
    )
    return {k: getattr(clf, k, None) for k in keys}


def build_metadata(
    artifacts: TrainArtifacts,
    *,
    train_rows: int,
    test_rows: int,
    metrics: dict | None = None,
    dataset_fingerprint: str | None = None,
) -> dict:
    """Build a metadata dict to be saved alongside the model."""
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": {
            "xgboost": _safe_pkg_version("xgboost"),
            "scikit-learn": _safe_pkg_version("scikit-learn"),
            "pandas": _safe_pkg_version("pandas"),
            "numpy": _safe_pkg_version("numpy"),
        },
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "n_classes": int(len(artifacts.label_encoder.classes_)),
        "hyperparameters": _model_hyperparams(artifacts.pipeline),
        "metrics": metrics or {},
        "dataset_fingerprint": dataset_fingerprint,
    }


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Stable short fingerprint of a dataframe (shape + column names + sampled hash)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    if len(df):
        sample = df.head(min(100, len(df))).to_csv(index=False).encode()
        h.update(sample)
    return h.hexdigest()[:16]


def save_artifacts(
    artifacts: TrainArtifacts,
    *,
    model_path,
    label_encoder_path,
    metadata: dict | None = None,
) -> None:
    """Save the model, the encoder and (optionally) a metadata sidecar JSON.

    Metadata is written next to the model with the same stem and ``.meta.json``
    suffix (e.g. ``models/xgb_ranker.json`` -> ``models/xgb_ranker.meta.json``).
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.pipeline, model_path)
    joblib.dump(artifacts.label_encoder, label_encoder_path)
    if metadata is not None:
        meta_path = model_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(*, model_path, label_encoder_path) -> TrainArtifacts:
    pipe = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    return TrainArtifacts(pipeline=pipe, label_encoder=le)


def top_k_labels(proba: np.ndarray, label_encoder: LabelEncoder, *, k: int) -> list[list[str]]:
    topk = np.argsort(-proba, axis=1)[:, :k]
    labels = label_encoder.inverse_transform(topk.reshape(-1))
    return labels.reshape(topk.shape).tolist()


def export_onnx(pipeline: Pipeline, path: Path) -> None:
    """Export the XGBoost step of the pipeline to ONNX format.

    Only the XGBClassifier is exported — the sklearn preprocessor runs as
    usual at inference and is not the bottleneck.
    Requires onnxmltools and skl2onnx packages.
    """
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clf = pipeline.named_steps["clf"]
    n_features = clf.n_features_in_
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(clf, initial_types=initial_types)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def predict_proba_onnx(
    session,  # onnxruntime.InferenceSession
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    *,
    label_col: str,
) -> np.ndarray:
    """Run inference via ONNX Runtime — same contract as predict_proba().

    Drops label_col, CodeClient, and datetime columns, runs the sklearn
    preprocessor, then passes the result to the ONNX session.
    Returns array of shape (1, n_classes).
    """
    drop = [c for c in [label_col, "CodeClient"] if c in df.columns]
    drop += [c for c in df.columns if df[c].dtype.kind == "M"]
    X = df.drop(columns=drop)
    X_pre = preprocessor.transform(X).astype(np.float32)
    return session.run(["probabilities"], {"input": X_pre})[0]


class FastPreprocessor:
    """Drop-in for ColumnTransformer.transform — ~200x faster on single rows.

    Built once at startup from a fitted ColumnTransformer. Encodes a feature
    dict directly to a float32 numpy array via pre-extracted ordinal lookup
    tables and numeric passthrough, bypassing all pandas overhead.

    Column order matches ColumnTransformer output: cat-encoded columns first
    (in transformer order), then passthrough numeric columns.
    """

    def __init__(self, ct: ColumnTransformer) -> None:
        transformers_by_name = {name: (enc, cols) for name, enc, cols in ct.transformers_}
        cat_enc, cat_cols = transformers_by_name["cat"]
        _, num_cols = transformers_by_name["num"]

        self._cat_cols: list[str] = list(cat_cols)
        self._num_cols: list[str] = list(num_cols)
        # {col: {str(value): float(ordinal_index)}}
        self._cat_maps: dict[str, dict[str, float]] = {
            col: {str(v): float(i) for i, v in enumerate(cat_enc.categories_[j])}
            for j, col in enumerate(self._cat_cols)
        }
        self._n_features: int = len(self._cat_cols) + len(self._num_cols)

    def encode(self, row: dict) -> np.ndarray:
        """Encode one feature dict → float32 array of shape (1, n_features).

        Unknown category values → -1.0 (matching OrdinalEncoder unknown_value=-1).
        NaN / None numeric values pass through as np.nan.
        """
        out = np.empty(self._n_features, dtype=np.float32)
        for i, col in enumerate(self._cat_cols):
            val = str(row.get(col, "nan"))
            out[i] = self._cat_maps[col].get(val, -1.0)
        offset = len(self._cat_cols)
        for i, col in enumerate(self._num_cols):
            v = row.get(col)
            out[offset + i] = np.nan if pd.isna(v) else float(v)
        return out[np.newaxis, :]
