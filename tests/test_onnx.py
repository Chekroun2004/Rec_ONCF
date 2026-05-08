from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.training import export_onnx, predict_proba_onnx, FastPreprocessor


def _make_pipeline_and_row():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "prev_liaison":     rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId":   rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":     rng.choice(["1", "2"], n),
        "TrainAutocarId":   rng.choice(["10", "20"], n),
        "CarteClientId":    rng.choice(["0", "1"], n),
        "PrixParLiaison":   rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":    np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index":  np.arange(n, dtype=float),
        "days_since_prev":  rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour":      rng.integers(0, 24, n).astype(float),
        "depart_dow":       rng.integers(0, 7, n).astype(float),
        "depart_month":     rng.integers(1, 13, n).astype(float),
        "depart_hour_sin":  rng.standard_normal(n),
        "depart_hour_cos":  rng.standard_normal(n),
        "depart_dow_sin":   rng.standard_normal(n),
        "depart_dow_cos":   rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n).astype(float),
    })
    y_raw = rng.choice(["R1", "R2", "R3"], n)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    cat_cols = [
        "prev_liaison", "TypeParcoursId", "ClassificationId",
        "ClassePhysiqueId", "NiveauPrixId", "TrainAutocarId", "CarteClientId",
    ]
    num_cols = [c for c in df.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=5, device="cpu",
        eval_metric="mlogloss", random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(df, y)

    row = df.iloc[[0]].copy()
    row["LiaisonId"] = "__unknown__"
    row["CodeClient"] = 12345
    row["DateHeureDepartVoyageSegment"] = pd.Timestamp("2026-01-01")

    return pipe, row, len(le.classes_)


def test_export_creates_onnx_file(tmp_path):
    pipe, _, _ = _make_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_onnx_probas_match_sklearn(tmp_path):
    pipe, row, _ = _make_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))

    extra = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra if c in row.columns])
    proba_sklearn = pipe.predict_proba(row_clean)

    proba_onnx = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")

    np.testing.assert_allclose(proba_onnx, proba_sklearn, atol=1e-4)


def test_onnx_output_shape(tmp_path):
    pipe, row, n_classes = _make_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))

    proba = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")
    assert proba.shape == (1, n_classes)


def test_fast_preprocessor_matches_sklearn():
    """FastPreprocessor.encode must produce the same float32 array as ColumnTransformer.transform."""
    pipe, row, _ = _make_pipeline_and_row()
    ct = pipe["pre"]
    fp = FastPreprocessor(ct)

    extra = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra if c in row.columns])
    X_sklearn = ct.transform(row_clean).astype(np.float32)

    row_dict = row.iloc[0].to_dict()
    X_fast = fp.encode(row_dict)

    np.testing.assert_array_equal(X_sklearn, X_fast)


def test_fast_preprocessor_unknown_category():
    """An unseen category value must encode to -1.0 (matching OrdinalEncoder unknown_value=-1)."""
    pipe, row, _ = _make_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])

    row_dict = row.iloc[0].to_dict()
    row_dict["TypeParcoursId"] = "UNSEEN_VALUE_XYZ"

    X_fast = fp.encode(row_dict)
    type_idx = cat_cols.index("TypeParcoursId")
    assert X_fast[0, type_idx] == -1.0


def test_fast_preprocessor_nan_numeric_passthrough():
    """NaN in a numeric column must survive as np.nan in the output array."""
    pipe, row, _ = _make_pipeline_and_row()
    fp = FastPreprocessor(pipe["pre"])
    cat_cols = list(pipe["pre"].transformers_[0][2])
    num_cols = list(pipe["pre"].transformers_[1][2])

    row_dict = row.iloc[0].to_dict()
    row_dict["PrixParLiaison"] = float("nan")

    X_fast = fp.encode(row_dict)
    prix_idx = len(cat_cols) + num_cols.index("PrixParLiaison")
    assert np.isnan(X_fast[0, prix_idx])
