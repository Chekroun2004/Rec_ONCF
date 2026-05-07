# ONNX Runtime Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace XGBoost sklearn inference with ONNX Runtime to reduce `/recommend` latency from ~100ms to <50ms without retraining or changing model accuracy.

**Architecture:** Export only the `XGBClassifier` step from the sklearn Pipeline to ONNX (XGBoost 2.x native export). At inference, run the sklearn `ColumnTransformer` preprocessor as before (~2ms), then pass the result to an `InferenceSession` (~10ms) instead of the full `pipeline.predict_proba()` (~98ms). The `Recommender` dataclass gets an optional `onnx_session` field; `from_data()` (used in all unit tests) leaves it `None` and falls back to sklearn — no test changes needed.

**Tech Stack:** `onnxruntime>=1.18`, XGBoost 2.x (already installed), scikit-learn (already installed), pytest

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add `onnxruntime>=1.18` |
| Modify | `src/rec_oncf/config.py` | Add `onnx_model_path: Path` → `models/xgb_ranker.onnx` |
| Modify | `src/rec_oncf/training.py` | Add `export_onnx()` + `predict_proba_onnx()` |
| Create | `scripts/06_export_onnx.py` | Load joblib → export ONNX → benchmark sklearn vs ONNX |
| Modify | `src/rec_oncf/recommender.py` | Add `onnx_session` field; `from_paths` loads it; `recommend()` uses ONNX when available |
| Create | `tests/test_onnx.py` | 3 TDD tests: file creation, proba parity, output shape |

---

## Task 1: Dependencies + config

**Files:**
- Modify: `requirements.txt`
- Modify: `src/rec_oncf/config.py`

- [ ] **Step 1: Add `onnxruntime` to `requirements.txt`**

Open `requirements.txt` and add after `joblib>=1.4`:

```
onnxruntime>=1.18
```

- [ ] **Step 2: Install it**

```powershell
.venv\Scripts\python.exe -m pip install onnxruntime --quiet
```

Expected: no error.

- [ ] **Step 3: Add `onnx_model_path` to `src/rec_oncf/config.py`**

In the `Paths` dataclass, add after `label_encoder_path`:

```python
    cold_start_path: Path
    onnx_model_path: Path
```

In `default_paths()`, add after `cold_start_path = models_dir / "cold_start.joblib"`:

```python
    onnx_model_path = models_dir / "xgb_ranker.onnx"
```

In the `return Paths(...)` call, add after `cold_start_path=cold_start_path,`:

```python
        onnx_model_path=onnx_model_path,
```

- [ ] **Step 4: Verify full test suite still passes**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: `53 passed`.

- [ ] **Step 5: Commit**

```powershell
git add requirements.txt src/rec_oncf/config.py
git commit -m "feat: add onnxruntime dependency and onnx_model_path to config"
```

---

## Task 2: `export_onnx` and `predict_proba_onnx` in `training.py` (TDD)

**Files:**
- Create: `tests/test_onnx.py`
- Modify: `src/rec_oncf/training.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_onnx.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.training import TrainArtifacts, export_onnx, predict_proba_onnx


def _make_pipeline_and_row():
    """Build a tiny but realistic pipeline + one sample inference row."""
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "prev_liaison":    rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId":  rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":    rng.choice(["1", "2"], n),
        "TrainAutocarId":  rng.choice(["10", "20"], n),
        "CarteClientId":   rng.choice(["0", "1"], n),
        "PrixParLiaison":  rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":   np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index": np.arange(n, dtype=float),
        "days_since_prev": rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour":     rng.integers(0, 24, n).astype(float),
        "depart_dow":      rng.integers(0, 7, n).astype(float),
        "depart_month":    rng.integers(1, 13, n).astype(float),
        "depart_hour_sin": rng.standard_normal(n),
        "depart_hour_cos": rng.standard_normal(n),
        "depart_dow_sin":  rng.standard_normal(n),
        "depart_dow_cos":  rng.standard_normal(n),
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

    # Build a realistic inference row (includes cols that get dropped)
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

    proba_sklearn = predict_proba_onnx.__wrapped_sklearn__(pipe, row)
    proba_onnx = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")

    np.testing.assert_allclose(proba_onnx, proba_sklearn, atol=1e-4)


def test_onnx_output_shape(tmp_path):
    pipe, row, n_classes = _make_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))

    proba = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")
    assert proba.shape == (1, n_classes)
```

- [ ] **Step 2: Run tests — expect ImportError**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_onnx.py -v
```

Expected: `ImportError: cannot import name 'export_onnx' from 'rec_oncf.training'`

- [ ] **Step 3: Add `export_onnx` and `predict_proba_onnx` to `src/rec_oncf/training.py`**

Add these two functions at the end of `training.py`, after `top_k_labels`. Also add `from pathlib import Path` at the top if not already present (it isn't — add it to the imports):

```python
from pathlib import Path
```

Then add at the end of the file:

```python
def export_onnx(pipeline: Pipeline, path: Path) -> None:
    """Export the XGBoost step of the pipeline to ONNX format.

    The sklearn preprocessor is NOT exported — it runs as usual at inference.
    Only the XGBClassifier is exported, since it is the inference bottleneck.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.named_steps["clf"].save_model(str(path))


def predict_proba_onnx(
    session,  # onnxruntime.InferenceSession
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    *,
    label_col: str,
) -> np.ndarray:
    """Run inference using an ONNX Runtime session.

    Drops label_col, CodeClient, and datetime columns exactly like
    predict_proba() does, then runs the sklearn preprocessor and passes
    the result to the ONNX session.

    Returns array of shape (1, n_classes) — same contract as predict_proba().
    """
    drop = [c for c in [label_col, "CodeClient"] if c in df.columns]
    drop += [c for c in df.columns if df[c].dtype.kind == "M"]
    X = df.drop(columns=drop)
    X_pre = preprocessor.transform(X).astype(np.float32)
    return session.run(["probabilities"], {"input": X_pre})[0]
```

Now fix the test — `__wrapped_sklearn__` doesn't exist. Replace the `test_onnx_probas_match_sklearn` test with a version that computes the sklearn baseline inline:

```python
def test_onnx_probas_match_sklearn(tmp_path):
    pipe, row, _ = _make_pipeline_and_row()
    path = tmp_path / "model.onnx"
    export_onnx(pipe, path)
    session = InferenceSession(str(path))

    # Sklearn baseline: drop extra cols then pipeline.predict_proba
    extra_cols = ["LiaisonId", "CodeClient", "DateHeureDepartVoyageSegment"]
    row_clean = row.drop(columns=[c for c in extra_cols if c in row.columns])
    proba_sklearn = pipe.predict_proba(row_clean)

    proba_onnx = predict_proba_onnx(session, pipe["pre"], row, label_col="LiaisonId")

    np.testing.assert_allclose(proba_onnx, proba_sklearn, atol=1e-4)
```

Update `tests/test_onnx.py` to use this corrected version (remove the `__wrapped_sklearn__` line from `test_onnx_probas_match_sklearn`).

The final `tests/test_onnx.py` content (replacing Step 1's version):

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from rec_oncf.training import export_onnx, predict_proba_onnx


def _make_pipeline_and_row():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "prev_liaison":    rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId":  rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":    rng.choice(["1", "2"], n),
        "TrainAutocarId":  rng.choice(["10", "20"], n),
        "CarteClientId":   rng.choice(["0", "1"], n),
        "PrixParLiaison":  rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":   np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index": np.arange(n, dtype=float),
        "days_since_prev": rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour":     rng.integers(0, 24, n).astype(float),
        "depart_dow":      rng.integers(0, 7, n).astype(float),
        "depart_month":    rng.integers(1, 13, n).astype(float),
        "depart_hour_sin": rng.standard_normal(n),
        "depart_hour_cos": rng.standard_normal(n),
        "depart_dow_sin":  rng.standard_normal(n),
        "depart_dow_cos":  rng.standard_normal(n),
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
```

- [ ] **Step 4: Run tests — expect all 3 to pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_onnx.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Run full suite to confirm no regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: `56 passed`.

- [ ] **Step 6: Commit**

```powershell
git add tests/test_onnx.py src/rec_oncf/training.py
git commit -m "feat: add export_onnx and predict_proba_onnx to training.py"
```

---

## Task 3: Script `06_export_onnx.py`

**Files:**
- Create: `scripts/06_export_onnx.py`

- [ ] **Step 1: Create the script**

```python
# scripts/06_export_onnx.py
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import export_onnx, load_artifacts, predict_proba, predict_proba_onnx


def _benchmark(fn, n: int = 20) -> float:
    """Return median latency in ms over n calls."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main() -> None:
    paths = default_paths()

    if not paths.xgb_model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {paths.xgb_model_path}. Run scripts/03_train_ranker.py first."
        )

    print("Loading artifacts ...")
    arts = load_artifacts(
        model_path=paths.xgb_model_path,
        label_encoder_path=paths.label_encoder_path,
    )

    print(f"Exporting ONNX -> {paths.onnx_model_path} ...")
    export_onnx(arts.pipeline, paths.onnx_model_path)
    size_mb = paths.onnx_model_path.stat().st_size / 1e6
    print(f"  Done ({size_mb:.1f} MB)")

    print("Loading ONNX session ...")
    from onnxruntime import InferenceSession
    session = InferenceSession(str(paths.onnx_model_path))

    print("Building benchmark row ...")
    clean = read_parquet(paths.processed_dataset_parquet)
    history_lookup = {
        str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
        for cid, grp in clean.groupby("CodeClient")
    }
    warm_users = [(uid, len(df)) for uid, df in history_lookup.items() if len(df) >= 10]
    warm_users.sort(key=lambda x: -x[1])
    uid, n_trips = warm_users[0]
    history = history_lookup[uid]
    feat_row = compute_inference_row(history)
    print(f"  User {uid} ({n_trips} trips)")

    preprocessor = arts.pipeline["pre"]

    p50_sklearn = _benchmark(lambda: predict_proba(arts, feat_row, label_col="LiaisonId"))
    p50_onnx = _benchmark(
        lambda: predict_proba_onnx(session, preprocessor, feat_row, label_col="LiaisonId")
    )

    speedup = p50_sklearn / p50_onnx
    print()
    print(f"XGBoost sklearn p50 : {p50_sklearn:.1f} ms")
    print(f"XGBoost ONNX    p50 : {p50_onnx:.1f} ms")
    print(f"Speedup             : {speedup:.1f}x")

    if p50_onnx >= 30:
        raise RuntimeError(
            f"ONNX p50 = {p50_onnx:.1f} ms >= 30 ms target. "
            "Check onnxruntime installation or model export."
        )
    print(f"\nTarget met: ONNX p50 {p50_onnx:.1f} ms < 30 ms")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```powershell
.venv\Scripts\python.exe scripts/06_export_onnx.py
```

Expected output (numbers will vary):
```
Loading artifacts ...
Exporting ONNX -> ...\models\xgb_ranker.onnx ...
  Done (XX.X MB)
Loading ONNX session ...
Building benchmark row ...
  User XXXXXXXX (NNN trips)

XGBoost sklearn p50 : ~98.0 ms
XGBoost ONNX    p50 : ~10-25 ms
Speedup             : ~4-10x

Target met: ONNX p50 X.X ms < 30 ms
```

If `RuntimeError` is raised (p50 >= 30ms), check onnxruntime version: `pip show onnxruntime`.

- [ ] **Step 3: Commit**

```powershell
git add scripts/06_export_onnx.py
git commit -m "feat: add script 06 to export XGBoost to ONNX and benchmark"
```

---

## Task 4: Wire ONNX into `Recommender`

**Files:**
- Modify: `src/rec_oncf/recommender.py`

- [ ] **Step 1: Replace the full content of `src/rec_oncf/recommender.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from rec_oncf.candidates import generate_candidates
from rec_oncf.cold_start import ColdStartRecommender, build_cold_start_recommender, load_cold_start
from rec_oncf.config import Paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx, export_onnx

_COLD_START = {"mode": "cold_start", "recommendations": []}


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    For users with 1-2 trips, falls back to co-occurrence collaborative
    filtering. For warm users, features are computed ON THE FLY from live
    history and scored via ONNX Runtime (fast path) or sklearn (fallback).
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None  # onnxruntime.InferenceSession

    @classmethod
    def from_paths(cls, paths: Paths) -> Recommender:
        from onnxruntime import InferenceSession
        artifacts = load_artifacts(
            model_path=paths.xgb_model_path,
            label_encoder_path=paths.label_encoder_path,
        )
        clean = read_parquet(paths.processed_dataset_parquet)
        cold_start_rec = load_cold_start(paths.cold_start_path)
        if not paths.onnx_model_path.exists():
            raise RuntimeError(
                f"ONNX model not found: {paths.onnx_model_path}. "
                "Run scripts/06_export_onnx.py first."
            )
        onnx_session = InferenceSession(str(paths.onnx_model_path))
        return cls._build(artifacts, clean, cold_start_rec, onnx_session)

    @classmethod
    def from_data(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,  # kept for backward compat, unused
    ) -> Recommender:
        cold_start_rec = build_cold_start_recommender(clean_df)
        return cls._build(artifacts, clean_df, cold_start_rec)

    @classmethod
    def _build(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        cold_start_rec: ColdStartRecommender,
        onnx_session: object | None = None,
    ) -> Recommender:
        history_lookup = {
            str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
            for cid, grp in clean_df.groupby("CodeClient")
        }
        return cls(
            artifacts=artifacts,
            history_lookup=history_lookup,
            cold_start_rec=cold_start_rec,
            onnx_session=onnx_session,
        )

    def recommend(
        self,
        code_client: str,
        k: int = 1,
        *,
        asof: datetime | pd.Timestamp | None = None,
    ) -> dict:
        code_client = str(code_client)
        k = min(max(k, 1), 3)

        history = self.history_lookup.get(code_client)

        if history is None:
            return _COLD_START

        if len(history) < 3:
            recs = self.cold_start_rec.recommend(history, k)
            if recs:
                return {"mode": "cold_start_cf", "recommendations": recs}
            return _COLD_START

        candidates = generate_candidates(
            history, user_id=code_client, max_candidates=10
        )
        if not candidates:
            return _COLD_START

        feat_row = compute_inference_row(history, asof=asof)

        le = self.artifacts.label_encoder
        known = set(le.classes_)
        valid_candidates = [c for c in candidates if c in known]

        if not valid_candidates:
            return {"mode": "model", "recommendations": candidates[:k]}

        if self.onnx_session is not None:
            proba = predict_proba_onnx(
                self.onnx_session,
                self.artifacts.pipeline["pre"],
                feat_row,
                label_col="LiaisonId",
            )[0]
        else:
            proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]

        cand_idx = le.transform(np.asarray(valid_candidates))
        cand_scores = proba[cand_idx]

        order = np.argsort(-cand_scores)[:k]
        recs = [valid_candidates[i] for i in order]
        return {"mode": "model", "recommendations": recs}
```

- [ ] **Step 2: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: `56 passed` (all existing tests use `from_data()` → `onnx_session=None` → sklearn fallback).

- [ ] **Step 3: Run ruff**

```powershell
.venv\Scripts\python.exe -m ruff check src/ apps/ scripts/ tests/ --output-format=concise
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```powershell
git add src/rec_oncf/recommender.py
git commit -m "feat: wire ONNX Runtime into Recommender for fast inference"
```

---

## Task 5: Final validation + push

- [ ] **Step 1: Run the full test suite one final time**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: `56 passed`.

- [ ] **Step 2: Re-run the benchmark to confirm improvement**

```powershell
.venv\Scripts\python.exe scripts/06_export_onnx.py
```

Expected: ONNX p50 < 30ms, speedup printed.

- [ ] **Step 3: Push**

```powershell
git push origin main
```

---

## Self-Review

**Spec coverage:**
- [x] `export_onnx()` — Task 2
- [x] `predict_proba_onnx()` — Task 2
- [x] `onnx_model_path` in config — Task 1
- [x] `scripts/06_export_onnx.py` with benchmark — Task 3
- [x] `Recommender.from_paths` loads ONNX, raises RuntimeError if missing — Task 4
- [x] `Recommender.from_data` leaves `onnx_session=None` (backward compat) — Task 4
- [x] `recommend()` uses ONNX when available, sklearn otherwise — Task 4
- [x] 3 TDD tests — Task 2
- [x] Success criterion: ONNX p50 < 30ms asserted in script — Task 3

**Placeholder scan:** None found.

**Type consistency:** `predict_proba_onnx(session, preprocessor, df, *, label_col)` used identically in Task 2 (tests + implementation) and Task 4 (recommender).
