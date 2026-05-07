# Retraining Pipeline with KPI Guardrail — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A script that retrains the full ONCF recommender pipeline on fresh data, compares HR@1 to the currently deployed model, and only overwrites `models/` if the new model's HR@1 does not drop by more than 5pp.

**Architecture:** A new `src/rec_oncf/retrain.py` module holds all reusable functions (`load_current_metrics`, `check_guardrail`, `evaluate_model`, `promote_artifacts`, `retrain_pipeline`). A thin CLI `scripts/07_retrain.py` calls `retrain_pipeline()` and exits 0 on success / 1 when blocked. New models are always trained into `models/staging/` first; `promote_artifacts()` copies them to `models/` only after the guardrail passes. `retrain_pipeline` accepts optional `features_df` / `clean_df` overrides so tests never hit disk reads.

**Tech Stack:** pandas, xgboost, scikit-learn, joblib, onnxmltools, onnxruntime, shutil (stdlib), pytest

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `src/rec_oncf/retrain.py` | `load_current_metrics`, `check_guardrail`, `evaluate_model`, `promote_artifacts`, `retrain_pipeline` |
| Create | `scripts/07_retrain.py` | Thin CLI: argparse `--dry-run`, call `retrain_pipeline`, print report, exit code |
| Create | `tests/test_retrain.py` | TDD tests for every function (13 tests) |

---

## Task 1: `load_current_metrics`, `check_guardrail`, `evaluate_model`

**Files:**
- Create: `tests/test_retrain.py`
- Create: `src/rec_oncf/retrain.py` (partial — three functions)

### Context

`models/xgb_ranker.meta.json` is written by `save_artifacts()` in `training.py` and has this shape:
```json
{
  "trained_at_utc": "...",
  "metrics": {"hit_rate@1": 0.7628, "hit_rate@3": 0.9055, "mrr@3": 0.8277},
  ...
}
```

`metrics.py` signatures (take encoded integer arrays, not strings):
```python
hit_rate_at_k(y_true: np.ndarray, y_proba: np.ndarray, *, k: int) -> float
mrr_at_k(y_true: np.ndarray, y_proba: np.ndarray, *, k: int) -> float
```
`y_true` must be the integer class indices from `label_encoder.transform(...)`, and `y_proba` is the raw probability matrix from `predict_proba(artifacts, df, label_col=...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrain.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rec_oncf.config import Paths
from rec_oncf.retrain import (
    check_guardrail,
    evaluate_model,
    load_current_metrics,
    promote_artifacts,
    retrain_pipeline,
)
from rec_oncf.training import temporal_split, train_xgb_multiclass


# ---------- shared helpers ----------

def _make_mini_features(n: int = 200) -> pd.DataFrame:
    """Mini features DataFrame with the same schema as features.parquet."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "LiaisonId": rng.choice(["R1", "R2", "R3"], n),
        "DateHeureDepartVoyageSegment": pd.date_range("2024-01-01", periods=n, freq="h"),
        "prev_liaison": rng.choice(["A", "B", "nan"], n),
        "TypeParcoursId": rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId": rng.choice(["1", "2"], n),
        "TrainAutocarId": rng.choice(["10", "20"], n),
        "CarteClientId": rng.choice(["0", "1"], n),
        "PrixParLiaison": rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment": np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index": np.arange(n, dtype=float),
        "days_since_prev": rng.choice([np.nan, 7.0, 14.0], n),
        "user_top_liaison_share": rng.uniform(0, 1, n),
        "depart_hour": rng.integers(0, 24, n).astype(float),
        "depart_dow": rng.integers(0, 7, n).astype(float),
        "depart_month": rng.integers(1, 13, n).astype(float),
        "depart_hour_sin": rng.standard_normal(n),
        "depart_hour_cos": rng.standard_normal(n),
        "depart_dow_sin": rng.standard_normal(n),
        "depart_dow_cos": rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n).astype(float),
    })


def _make_mini_clean(n: int = 100) -> pd.DataFrame:
    """Minimal oncf_clean DataFrame for cold-start CF building."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "CodeClient": rng.choice(["C1", "C2", "C3"], n).astype(str),
        "LiaisonId": rng.choice(["R1", "R2", "R3"], n).astype(str),
        "DateHeureDepartVoyageSegment": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _make_test_paths(base: Path) -> Paths:
    """Build a Paths dataclass pointing entirely to tmp_path subdirs."""
    models_dir = base / "models"
    models_dir.mkdir()
    processed_dir = base / "data" / "processed"
    processed_dir.mkdir(parents=True)
    return Paths(
        project_root=base,
        desktop=base,
        raw_oncf_data=base / "oncf_data.csv",
        raw_liaison=base / "Liaison.csv",
        processed_dir=processed_dir,
        processed_dataset_parquet=processed_dir / "oncf_clean.parquet",
        processed_dataset_csv=processed_dir / "oncf_clean.csv",
        features_parquet=processed_dir / "features.parquet",
        models_dir=models_dir,
        xgb_model_path=models_dir / "xgb_ranker.json",
        label_encoder_path=models_dir / "label_encoder.joblib",
        cold_start_path=models_dir / "cold_start.joblib",
        onnx_model_path=models_dir / "xgb_ranker.onnx",
    )


# ---------- load_current_metrics ----------

def test_load_current_metrics_returns_empty_when_file_missing(tmp_path):
    metrics = load_current_metrics(tmp_path / "xgb_ranker.meta.json")
    assert metrics == {}


def test_load_current_metrics_reads_metrics_key(tmp_path):
    meta = {"metrics": {"hit_rate@1": 0.76, "hit_rate@3": 0.90}, "other": "ignored"}
    (tmp_path / "xgb_ranker.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    metrics = load_current_metrics(tmp_path / "xgb_ranker.meta.json")
    assert metrics["hit_rate@1"] == pytest.approx(0.76)
    assert metrics["hit_rate@3"] == pytest.approx(0.90)
    assert "other" not in metrics


# ---------- check_guardrail ----------

def test_check_guardrail_passes_when_no_current():
    passes, reason = check_guardrail({}, {"hit_rate@1": 0.5})
    assert passes
    assert "waived" in reason


def test_check_guardrail_passes_small_drop():
    passes, _ = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.73})
    assert passes  # 0.02 drop < 0.05


def test_check_guardrail_passes_improvement():
    passes, _ = check_guardrail({"hit_rate@1": 0.70}, {"hit_rate@1": 0.76})
    assert passes


def test_check_guardrail_passes_at_exact_threshold():
    # Drop exactly equal to threshold passes (condition is strictly greater than)
    passes, _ = check_guardrail({"hit_rate@1": 0.80}, {"hit_rate@1": 0.75})
    assert passes  # 0.05 drop == 0.05 threshold → not strictly greater → passes


def test_check_guardrail_fails_above_threshold():
    passes, reason = check_guardrail({"hit_rate@1": 0.75}, {"hit_rate@1": 0.69})
    assert not passes  # 0.06 drop > 0.05
    assert "BLOCKED" in reason


def test_check_guardrail_custom_threshold():
    passes, _ = check_guardrail(
        {"hit_rate@1": 0.80}, {"hit_rate@1": 0.75}, threshold=0.10
    )
    assert passes  # 0.05 drop < 0.10 custom threshold


# ---------- evaluate_model ----------

def test_evaluate_model_returns_correct_keys():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    metrics = evaluate_model(arts, df)
    assert set(metrics.keys()) == {"hit_rate@1", "hit_rate@3", "mrr@3", "test_rows"}


def test_evaluate_model_metrics_in_range():
    df = _make_mini_features()
    df_train, _ = temporal_split(df, time_col="DateHeureDepartVoyageSegment")
    arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    metrics = evaluate_model(arts, df)
    assert 0.0 <= metrics["hit_rate@1"] <= 1.0
    assert 0.0 <= metrics["hit_rate@3"] <= 1.0
    assert 0.0 <= metrics["mrr@3"] <= 1.0
    assert metrics["test_rows"] > 0
```

- [ ] **Step 2: Run tests — expect ImportError**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrain.py -v
```

Expected: `ImportError: cannot import name 'check_guardrail' from 'rec_oncf.retrain'` (module doesn't exist yet).

- [ ] **Step 3: Create `src/rec_oncf/retrain.py` with the three functions**

```python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from rec_oncf.cold_start import build_cold_start_recommender, save_cold_start
from rec_oncf.config import Paths
from rec_oncf.io import read_parquet
from rec_oncf.metrics import hit_rate_at_k, mrr_at_k
from rec_oncf.training import (
    TrainArtifacts,
    build_metadata,
    export_onnx,
    fingerprint_dataframe,
    predict_proba,
    save_artifacts,
    temporal_split,
    train_xgb_multiclass,
)


def load_current_metrics(meta_path: Path) -> dict:
    """Load metrics from a model sidecar JSON. Returns {} when file is missing (first run)."""
    if not meta_path.exists():
        return {}
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return data.get("metrics", {})


def check_guardrail(
    current: dict,
    new: dict,
    *,
    threshold: float = 0.05,
) -> tuple[bool, str]:
    """Return (passes, reason). Fails only if new HR@1 drops strictly more than threshold."""
    if not current:
        return True, "No current metrics — guardrail waived (first run)"
    current_hr1 = current.get("hit_rate@1", 0.0)
    new_hr1 = new.get("hit_rate@1", 0.0)
    drop = current_hr1 - new_hr1
    if drop > threshold:
        return (
            False,
            f"BLOCKED: HR@1 dropped {drop:.4f} "
            f"(current={current_hr1:.4f}, new={new_hr1:.4f}, threshold={threshold:.4f})",
        )
    return (
        True,
        f"OK: HR@1 current={current_hr1:.4f}, new={new_hr1:.4f}, drop={drop:.4f}",
    )


def evaluate_model(
    artifacts: TrainArtifacts,
    features_df: pd.DataFrame,
    *,
    label_col: str = "LiaisonId",
    time_col: str = "DateHeureDepartVoyageSegment",
    train_frac: float = 0.8,
) -> dict:
    """Evaluate artifacts on the temporal test split of features_df.

    Returns {"hit_rate@1": float, "hit_rate@3": float, "mrr@3": float, "test_rows": int}.
    """
    _, df_test = temporal_split(features_df, time_col=time_col, train_frac=train_frac)
    known = set(artifacts.label_encoder.classes_)
    df_test = df_test[df_test[label_col].astype(str).isin(known)]
    if df_test.empty:
        return {"hit_rate@1": 0.0, "hit_rate@3": 0.0, "mrr@3": 0.0, "test_rows": 0}
    proba = predict_proba(artifacts, df_test, label_col=label_col)
    y_true = artifacts.label_encoder.transform(df_test[label_col].astype(str).to_numpy())
    return {
        "hit_rate@1": hit_rate_at_k(y_true, proba, k=1),
        "hit_rate@3": hit_rate_at_k(y_true, proba, k=3),
        "mrr@3": mrr_at_k(y_true, proba, k=3),
        "test_rows": len(df_test),
    }


def promote_artifacts(staging_dir: Path, models_dir: Path) -> None:
    """Copy all files from staging_dir into models_dir, overwriting existing files."""
    for src in staging_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, models_dir / src.name)


def retrain_pipeline(
    paths: Paths,
    *,
    dry_run: bool = False,
    features_df: pd.DataFrame | None = None,
    clean_df: pd.DataFrame | None = None,
) -> dict:
    """Full pipeline: retrain → evaluate → guardrail → promote.

    Returns a report dict. Pass features_df / clean_df to skip disk reads (useful in tests).
    """
    if features_df is None:
        features_df = read_parquet(paths.features_parquet)
    if clean_df is None:
        clean_df = read_parquet(paths.processed_dataset_parquet)

    current_metrics = load_current_metrics(paths.models_dir / "xgb_ranker.meta.json")

    df_train, _ = temporal_split(features_df, time_col="DateHeureDepartVoyageSegment")

    staging_dir = paths.models_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    new_arts = train_xgb_multiclass(
        df_train, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    new_metrics = evaluate_model(new_arts, features_df)

    metadata = build_metadata(
        new_arts,
        train_rows=len(df_train),
        test_rows=new_metrics["test_rows"],
        metrics={k: v for k, v in new_metrics.items() if k != "test_rows"},
        dataset_fingerprint=fingerprint_dataframe(features_df),
    )
    save_artifacts(
        new_arts,
        model_path=staging_dir / paths.xgb_model_path.name,
        label_encoder_path=staging_dir / paths.label_encoder_path.name,
        metadata=metadata,
    )

    cs = build_cold_start_recommender(clean_df)
    save_cold_start(cs, staging_dir / paths.cold_start_path.name)

    export_onnx(new_arts.pipeline, staging_dir / paths.onnx_model_path.name)

    passes, reason = check_guardrail(current_metrics, new_metrics)

    report = {
        "current_metrics": current_metrics,
        "new_metrics": new_metrics,
        "guardrail_passes": passes,
        "guardrail_reason": reason,
        "promoted": False,
        "dry_run": dry_run,
    }

    if passes and not dry_run:
        promote_artifacts(staging_dir, paths.models_dir)
        report["promoted"] = True

    return report
```

Note: this file contains all five functions. Tasks 1, 2, and 3 in this plan work on different parts of the same file.

- [ ] **Step 4: Run the 9 Task 1 tests — expect all to pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrain.py::test_load_current_metrics_returns_empty_when_file_missing tests/test_retrain.py::test_load_current_metrics_reads_metrics_key tests/test_retrain.py::test_check_guardrail_passes_when_no_current tests/test_retrain.py::test_check_guardrail_passes_small_drop tests/test_retrain.py::test_check_guardrail_passes_improvement tests/test_retrain.py::test_check_guardrail_passes_at_exact_threshold tests/test_retrain.py::test_check_guardrail_fails_above_threshold tests/test_retrain.py::test_check_guardrail_custom_threshold tests/test_retrain.py::test_evaluate_model_returns_correct_keys tests/test_retrain.py::test_evaluate_model_metrics_in_range -v
```

Expected: `10 passed` (the `promote_artifacts` and `retrain_pipeline` tests will error — that's OK for now, we run only the 10 Task 1 tests here).

- [ ] **Step 5: Commit**

```powershell
git add tests/test_retrain.py src/rec_oncf/retrain.py
git commit -m "feat: add load_current_metrics, check_guardrail, evaluate_model to retrain.py"
```

---

## Task 2: `promote_artifacts` + `retrain_pipeline` tests

**Files:**
- Modify: `tests/test_retrain.py` (add 5 more tests — they were already written in the file created in Task 1, so they just need to pass now)
- `src/rec_oncf/retrain.py` already contains both functions from Task 1 Step 3

The tests to pass in this task are the last 5 in the file created in Task 1:

```python
# ---------- promote_artifacts ----------

def test_promote_artifacts_copies_files(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (staging / "xgb_ranker.json").write_bytes(b"model-data")
    (staging / "label_encoder.joblib").write_bytes(b"le-data")
    (staging / "xgb_ranker.meta.json").write_text('{"ok": true}', encoding="utf-8")

    promote_artifacts(staging, models)

    assert (models / "xgb_ranker.json").read_bytes() == b"model-data"
    assert (models / "label_encoder.joblib").read_bytes() == b"le-data"
    assert (models / "xgb_ranker.meta.json").exists()


def test_promote_artifacts_overwrites_existing(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (models / "xgb_ranker.json").write_bytes(b"old")
    (staging / "xgb_ranker.json").write_bytes(b"new")

    promote_artifacts(staging, models)

    assert (models / "xgb_ranker.json").read_bytes() == b"new"


# ---------- retrain_pipeline ----------

def test_retrain_pipeline_promotes_on_first_run(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(
        paths,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["guardrail_passes"] is True
    assert report["promoted"] is True
    assert report["dry_run"] is False
    assert (paths.models_dir / "xgb_ranker.json").exists()
    assert (paths.models_dir / "label_encoder.joblib").exists()
    assert (paths.models_dir / "xgb_ranker.onnx").exists()
    assert (paths.models_dir / "cold_start.joblib").exists()


def test_retrain_pipeline_dry_run_does_not_promote(tmp_path):
    paths = _make_test_paths(tmp_path)
    report = retrain_pipeline(
        paths,
        dry_run=True,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["dry_run"] is True
    assert report["promoted"] is False
    assert not (paths.models_dir / "xgb_ranker.json").exists()


def test_retrain_pipeline_blocked_by_guardrail(tmp_path):
    paths = _make_test_paths(tmp_path)
    meta = {"metrics": {"hit_rate@1": 0.999, "hit_rate@3": 0.999, "mrr@3": 0.999}}
    (paths.models_dir / "xgb_ranker.meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    report = retrain_pipeline(
        paths,
        features_df=_make_mini_features(),
        clean_df=_make_mini_clean(),
    )
    assert report["guardrail_passes"] is False
    assert report["promoted"] is False
    assert not (paths.models_dir / "xgb_ranker.json").exists()
```

These tests call `train_xgb_multiclass` on 160 rows — expect 5–15 seconds total for the 3 integration tests.

- [ ] **Step 1: Run only the 5 Task 2 tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrain.py::test_promote_artifacts_copies_files tests/test_retrain.py::test_promote_artifacts_overwrites_existing tests/test_retrain.py::test_retrain_pipeline_promotes_on_first_run tests/test_retrain.py::test_retrain_pipeline_dry_run_does_not_promote tests/test_retrain.py::test_retrain_pipeline_blocked_by_guardrail -v
```

Expected: `5 passed`. If any retrain_pipeline test fails with an import error from `retrain.py`, the full `retrain.py` was not saved correctly in Task 1 — re-save it.

- [ ] **Step 2: Run the full test_retrain.py to confirm all 15 tests pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrain.py -v
```

Expected: `15 passed`.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: `71 passed` (56 existing + 15 new).

- [ ] **Step 4: Commit**

```powershell
git add tests/test_retrain.py
git commit -m "test: add retrain_pipeline integration tests"
```

---

## Task 3: `scripts/07_retrain.py` CLI + final validation

**Files:**
- Create: `scripts/07_retrain.py`

- [ ] **Step 1: Create the script**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.retrain import retrain_pipeline


def _print_report(report: dict) -> None:
    cur = report["current_metrics"]
    new = report["new_metrics"]
    print()
    print("=" * 60)
    print("RETRAINING REPORT")
    print("=" * 60)
    print()
    print("Current model:")
    if cur:
        print(f"  hit_rate@1 = {cur.get('hit_rate@1', 0):.4f}")
        print(f"  hit_rate@3 = {cur.get('hit_rate@3', 0):.4f}")
        print(f"  mrr@3      = {cur.get('mrr@3', 0):.4f}")
    else:
        print("  (no existing model — first run)")
    print()
    print("New model:")
    print(f"  hit_rate@1 = {new.get('hit_rate@1', 0):.4f}")
    print(f"  hit_rate@3 = {new.get('hit_rate@3', 0):.4f}")
    print(f"  mrr@3      = {new.get('mrr@3', 0):.4f}")
    print(f"  test_rows  = {new.get('test_rows', 0)}")
    print()
    symbol = "OK" if report["guardrail_passes"] else "BLOCKED"
    print(f"Guardrail : {symbol}")
    print(f"Reason    : {report['guardrail_reason']}")
    print()
    if report["dry_run"]:
        print("DRY RUN — models/ not changed.")
    elif report["promoted"]:
        print("New model promoted to models/")
    else:
        print("Promotion blocked. models/ unchanged.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the ONCF recommender with a KPI guardrail."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrain and evaluate but do not overwrite models/.",
    )
    args = parser.parse_args()

    paths = default_paths()
    report = retrain_pipeline(paths, dry_run=args.dry_run)
    _print_report(report)
    sys.exit(0 if report["guardrail_passes"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ruff to ensure no lint errors**

```powershell
.venv\Scripts\python.exe -m ruff check src/ apps/ scripts/ tests/ --output-format=concise
```

Expected: `All checks passed!`

- [ ] **Step 3: Run the full test suite one final time**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: `71 passed`.

- [ ] **Step 4: Smoke-test the script with `--dry-run`**

This will retrain from the real `features.parquet` (~43 min) and compare against the current model. Use `--dry-run` so nothing is promoted. Run in a real terminal (not Claude's shell) because it takes time:

```powershell
.venv\Scripts\python.exe scripts/07_retrain.py --dry-run
```

Expected output (example):
```
============================================================
RETRAINING REPORT
============================================================

Current model:
  hit_rate@1 = 0.7628
  hit_rate@3 = 0.9055
  mrr@3      = 0.8277

New model:
  hit_rate@1 = 0.76XX
  hit_rate@3 = 0.90XX
  mrr@3      = 0.82XX
  test_rows  = 98261

Guardrail : OK
Reason    : OK: HR@1 current=0.7628, new=0.76XX, drop=...

DRY RUN — models/ not changed.
```

If you want to actually promote: `python scripts/07_retrain.py` (no `--dry-run`).

- [ ] **Step 5: Commit and push**

```powershell
git add scripts/07_retrain.py
git commit -m "feat: add script 07 — retraining pipeline with KPI guardrail"
git push origin main
```

- [ ] **Step 6: Update CLAUDE.md**

In the `## Current Status` table, add:
```markdown
| Retraining pipeline (script 07) | ✅ Done | `scripts/07_retrain.py --dry-run`; guardrail blocks if HR@1 drops >5pp |
```

In the scripts section of the repository layout, add:
```
│   ├── 07_retrain.py       # full retrain + KPI guardrail → promote models/  ✅ done
```

In the "How to Run Everything" section, add:
```powershell
# 9. Retrain with guardrail (optional — ~43 min on CPU)
.venv\Scripts\python.exe scripts/07_retrain.py --dry-run   # evaluate only
.venv\Scripts\python.exe scripts/07_retrain.py              # evaluate + promote
```

```powershell
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for retraining pipeline"
git push origin main
```

---

## Self-Review

**Spec coverage:**
- [x] Retrains full pipeline from `features.parquet` — `retrain_pipeline` → `train_xgb_multiclass` + `evaluate_model` + cold-start + ONNX export
- [x] Compares against currently deployed model — `load_current_metrics` reads `models/xgb_ranker.meta.json`
- [x] Guardrail blocks if HR@1 drops >5pp — `check_guardrail` with `drop > threshold`
- [x] Only overwrites `models/` on success — `promote_artifacts` called only when `passes and not dry_run`
- [x] Windows-compatible, no daemon — plain script runnable from PowerShell or Task Scheduler
- [x] `--dry-run` flag — evaluate without promoting
- [x] Exit code 1 on blocked — `sys.exit(0 if report["guardrail_passes"] else 1)`
- [x] First-run safe — `load_current_metrics` returns `{}` when no meta.json; guardrail waived

**Placeholder scan:** None.

**Type consistency:**
- `check_guardrail(current: dict, new: dict, *, threshold: float = 0.05) -> tuple[bool, str]` — used identically in tests and in `retrain_pipeline`
- `evaluate_model(artifacts, features_df, *, label_col, time_col, train_frac) -> dict` — keys `{"hit_rate@1", "hit_rate@3", "mrr@3", "test_rows"}` used identically in tests and in `retrain_pipeline` (which strips `test_rows` when building metadata)
- `promote_artifacts(staging_dir, models_dir)` — used identically in tests and in `retrain_pipeline`
- `retrain_pipeline(paths, *, dry_run, features_df, clean_df) -> dict` — keys `{current_metrics, new_metrics, guardrail_passes, guardrail_reason, promoted, dry_run}` used identically in all 3 integration tests
