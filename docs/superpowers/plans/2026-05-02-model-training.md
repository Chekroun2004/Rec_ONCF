# Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the feature set, add `mrr_at_k`, switch to `tree_method="hist"`, and wire everything into `03_train_ranker.py` to produce trained model artifacts and offline metrics.

**Architecture:** Four focused changes to existing library files (`metrics.py`, `features.py`, `training.py`, `03_train_ranker.py`), each tested in isolation before the full pipeline is re-run. Tests live in a new `tests/` directory with pytest.

**Tech Stack:** Python 3.12, XGBoost 3.2, scikit-learn 1.8, pandas 3.0, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `pythonpath = ["src"]` to pytest config |
| `tests/__init__.py` | Create | Makes `tests/` a package |
| `tests/test_metrics.py` | Create | Tests for `mrr_at_k` |
| `tests/test_features.py` | Create | Tests for new feature columns |
| `tests/test_training.py` | Create | Smoke test for `tree_method="hist"` with NaN inputs |
| `src/rec_oncf/metrics.py` | Modify | Add `mrr_at_k(y_true, y_proba, k)` |
| `src/rec_oncf/features.py` | Modify | Add `is_self_purchase` + `TrajetAllerRetour` |
| `src/rec_oncf/training.py` | Modify | Add `tree_method="hist"` to `XGBClassifier` |
| `scripts/03_train_ranker.py` | Modify | Import and compute `mrr@3`, add to report |

---

## Task 1: Test infrastructure + `mrr_at_k`

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/test_metrics.py`
- Modify: `src/rec_oncf/metrics.py`

- [ ] **Step 1.1: Add pytest pythonpath to `pyproject.toml`**

Open `pyproject.toml` and replace the existing `[tool.pytest.ini_options]` section with:

```toml
[tool.pytest.ini_options]
addopts = "-q"
pythonpath = ["src"]
```

- [ ] **Step 1.2: Create `tests/__init__.py`**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 1.3: Write failing tests for `mrr_at_k`**

Create `tests/test_metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from rec_oncf.metrics import mrr_at_k


def test_mrr_at_k_perfect():
    # true class ranked #1 for all samples → MRR = 1.0
    y_true = np.array([0, 1, 2])
    proba = np.eye(3)
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(1.0)


def test_mrr_at_k_second_place():
    # true class always ranked #2 → MRR = 0.5
    y_true = np.array([0])
    proba = np.array([[0.4, 0.6]])  # class 1 first, class 0 second
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.5)


def test_mrr_at_k_not_in_topk():
    # true class ranked #3 but k=2 → not retrieved → MRR = 0.0
    y_true = np.array([2])
    proba = np.array([[0.6, 0.3, 0.1]])  # class 2 is last
    assert mrr_at_k(y_true, proba, k=2) == pytest.approx(0.0)


def test_mrr_at_k_mixed():
    # sample 0: true=0, ranked #1 → rr=1.0
    # sample 1: true=2, ranked #2 → rr=0.5
    # MRR = (1.0 + 0.5) / 2 = 0.75
    y_true = np.array([0, 2])
    proba = np.array([
        [0.9, 0.05, 0.05],
        [0.5, 0.1, 0.4],
    ])
    assert mrr_at_k(y_true, proba, k=3) == pytest.approx(0.75)
```

- [ ] **Step 1.4: Run tests — verify they FAIL**

```
pytest tests/test_metrics.py -v
```

Expected: `ImportError` or `AttributeError: module 'rec_oncf.metrics' has no attribute 'mrr_at_k'`

- [ ] **Step 1.5: Implement `mrr_at_k` in `src/rec_oncf/metrics.py`**

Append to the existing file (keep `hit_rate_at_k` unchanged):

```python
def mrr_at_k(y_true: np.ndarray, y_proba: np.ndarray, *, k: int) -> float:
    """Mean Reciprocal Rank at k for multi-class probabilities.

    y_true: shape (n,)
    y_proba: shape (n, n_classes)
    """
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    rr = np.zeros(len(y_true))
    for rank in range(k):
        hit = (topk[:, rank] == y_true) & (rr == 0)
        rr[hit] = 1.0 / (rank + 1)
    return float(np.mean(rr))
```

- [ ] **Step 1.6: Run tests — verify they PASS**

```
pytest tests/test_metrics.py -v
```

Expected: `4 passed`

- [ ] **Step 1.7: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/test_metrics.py src/rec_oncf/metrics.py
git commit -m "feat: add mrr_at_k metric and test infrastructure"
```

---

## Task 2: Feature enrichment — `is_self_purchase` + `TrajetAllerRetour`

**Files:**
- Create: `tests/test_features.py`
- Modify: `src/rec_oncf/features.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_features.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rec_oncf.features import build_training_rows


def _minimal_clean_df(n: int = 6) -> pd.DataFrame:
    """Minimal DataFrame that mimics oncf_clean.parquet schema."""
    return pd.DataFrame({
        "CodeClient": ["C1", "C1", "C1", "C2", "C2", "C2"],
        "AchteurId":  ["C1", "C1", "X9", "C2", "C2", "C2"],   # C1 row2 is not self-purchase
        "LiaisonId": ["10", "20", "10", "30", "30", "10"],
        "TrajetAllerRetour": ["AR", "AS", "AR", "AR", "AR", "AS"],
        "DateHeureDepartVoyageSegment": pd.date_range("2020-01-01", periods=n, freq="30D"),
        "TypeParcoursId": [1, 1, 1, 2, 2, 2],
        "ClassificationId": [1, 1, 1, 1, 1, 1],
        "ClassePhysiqueId": [2, 2, 2, 1, 1, 1],
        "NiveauPrixId": [1, 1, 1, 2, 2, 2],
        "TrainAutocarId": [10, 10, 10, 20, 20, 20],
        "CarteClientId": [0, 0, 0, 1, 1, 1],
        "PrixParLiaison": [100.0, 200.0, np.nan, 150.0, 150.0, 100.0],
        "NbrVoySegment": [1, 1, 1, 2, 2, 1],
        "DelaiAnticipation": [5, 10, 3, 7, 2, 14],
        "depart_hour": [8, 18, 8, 7, 7, 9],
        "depart_dow": [0, 4, 0, 1, 1, 3],
        "depart_month": [1, 2, 3, 1, 2, 3],
        "DatePaiement": pd.date_range("2019-12-25", periods=n, freq="30D"),
    })


def test_is_self_purchase_column_exists():
    df = build_training_rows(_minimal_clean_df())
    assert "is_self_purchase" in df.columns


def test_is_self_purchase_values():
    df = build_training_rows(_minimal_clean_df())
    # C1 rows: first two are self-purchase (AchteurId==CodeClient), third is not
    c1 = df[df["CodeClient"] == "C1"].sort_values("DateHeureDepartVoyageSegment")
    vals = c1["is_self_purchase"].tolist()
    assert vals[0] == 1
    assert vals[1] == 1
    assert vals[2] == 0


def test_trajet_aller_retour_column_exists():
    df = build_training_rows(_minimal_clean_df())
    assert "TrajetAllerRetour" in df.columns


def test_trajet_aller_retour_is_string():
    df = build_training_rows(_minimal_clean_df())
    assert df["TrajetAllerRetour"].dtype == object


def test_output_has_26_columns():
    df = build_training_rows(_minimal_clean_df())
    # 3 identifiers + 8 cat + 15 num = 26
    assert len(df.columns) == 26
```

- [ ] **Step 2.2: Run tests — verify they FAIL**

```
pytest tests/test_features.py -v
```

Expected: `AssertionError` on `is_self_purchase` and `TrajetAllerRetour` not in columns; column count is 24 not 26.

- [ ] **Step 2.3: Update `src/rec_oncf/features.py`**

Replace the full file content:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def build_training_rows(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()
    df["CodeClient"] = df["CodeClient"].astype(str)
    df["LiaisonId"] = df["LiaisonId"].astype(str)

    df = df.sort_values(["CodeClient", "DateHeureDepartVoyageSegment"]).reset_index(drop=True)
    df["user_trip_index"] = df.groupby("CodeClient").cumcount()

    df["prev_liaison"] = df.groupby("CodeClient")["LiaisonId"].shift(1)

    prev_depart = df.groupby("CodeClient")["DateHeureDepartVoyageSegment"].shift(1)
    df["days_since_prev"] = (
        (df["DateHeureDepartVoyageSegment"] - prev_depart).dt.total_seconds() / 86400.0
    )

    # New: whether the booker is the traveler themselves
    df["is_self_purchase"] = (
        df["AchteurId"].astype(str) == df["CodeClient"].astype(str)
    ).astype(int)

    cat_cols = [
        "TypeParcoursId",
        "ClassificationId",
        "ClassePhysiqueId",
        "NiveauPrixId",
        "TrainAutocarId",
        "CarteClientId",
        "prev_liaison",
        "TrajetAllerRetour",
    ]
    for c in cat_cols:
        if c == "TrajetAllerRetour":
            df[c] = df[c].astype(str)
        else:
            df[c] = df[c].astype("Int64").astype(str)

    num_cols = [
        "PrixParLiaison",
        "NbrVoySegment",
        "DelaiAnticipation",
        "user_trip_index",
        "days_since_prev",
        "depart_hour",
        "depart_dow",
        "depart_month",
        "depart_hour_sin",
        "depart_hour_cos",
        "depart_dow_sin",
        "depart_dow_cos",
        "depart_month_sin",
        "depart_month_cos",
        "is_self_purchase",
    ]

    if "depart_hour_sin" not in df.columns or "depart_hour_cos" not in df.columns:
        df["depart_hour_sin"] = np.sin(2.0 * np.pi * df["depart_hour"] / 24.0)
        df["depart_hour_cos"] = np.cos(2.0 * np.pi * df["depart_hour"] / 24.0)
    if "depart_dow_sin" not in df.columns or "depart_dow_cos" not in df.columns:
        df["depart_dow_sin"] = np.sin(2.0 * np.pi * df["depart_dow"] / 7.0)
        df["depart_dow_cos"] = np.cos(2.0 * np.pi * df["depart_dow"] / 7.0)
    if "depart_month_sin" not in df.columns or "depart_month_cos" not in df.columns:
        if "depart_month" not in df.columns:
            df["depart_month"] = df["DateHeureDepartVoyageSegment"].dt.month
        df["depart_month_sin"] = np.sin(2.0 * np.pi * df["depart_month"] / 12.0)
        df["depart_month_cos"] = np.cos(2.0 * np.pi * df["depart_month"] / 12.0)

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_cols = [
        "CodeClient",
        "DateHeureDepartVoyageSegment",
        "LiaisonId",
        *cat_cols,
        *num_cols,
    ]

    return df[out_cols].copy()
```

- [ ] **Step 2.4: Run tests — verify they PASS**

```
pytest tests/test_features.py -v
```

Expected: `5 passed`

- [ ] **Step 2.5: Commit**

```bash
git add tests/test_features.py src/rec_oncf/features.py
git commit -m "feat: add is_self_purchase and TrajetAllerRetour features"
```

---

## Task 3: Switch to `tree_method="hist"` + smoke test

**Files:**
- Create: `tests/test_training.py`
- Modify: `src/rec_oncf/training.py`

- [ ] **Step 3.1: Write failing smoke test**

Create `tests/test_training.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rec_oncf.training import predict_proba, train_xgb_multiclass


def _tiny_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "LiaisonId": rng.choice(["A", "B", "C"], n),
        "DateHeureDepartVoyageSegment": pd.date_range("2020-01-01", periods=n, freq="D"),
        "TypeParcoursId":   rng.choice(["1", "2"], n),
        "ClassificationId": rng.choice(["1", "2"], n),
        "ClassePhysiqueId": rng.choice(["1", "2"], n),
        "NiveauPrixId":     rng.choice(["1", "2"], n),
        "TrainAutocarId":   rng.choice(["10", "20"], n),
        "CarteClientId":    rng.choice(["0", "1"], n),
        "prev_liaison":     rng.choice(["A", "B", "nan"], n),
        "TrajetAllerRetour": rng.choice(["AR", "AS"], n),
        "PrixParLiaison":   rng.choice([np.nan, 100.0, 200.0], n),
        "NbrVoySegment":    np.ones(n),
        "DelaiAnticipation": rng.integers(0, 30, n).astype(float),
        "user_trip_index":  np.arange(n),
        "days_since_prev":  rng.choice([np.nan, 7.0, 14.0], n),
        "depart_hour":      rng.integers(0, 24, n),
        "depart_dow":       rng.integers(0, 7, n),
        "depart_month":     rng.integers(1, 13, n),
        "depart_hour_sin":  rng.standard_normal(n),
        "depart_hour_cos":  rng.standard_normal(n),
        "depart_dow_sin":   rng.standard_normal(n),
        "depart_dow_cos":   rng.standard_normal(n),
        "depart_month_sin": rng.standard_normal(n),
        "depart_month_cos": rng.standard_normal(n),
        "is_self_purchase": rng.integers(0, 2, n),
    })


def test_train_handles_nulls_and_returns_proba():
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    proba = predict_proba(arts, df, label_col="LiaisonId")
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_artifacts_have_correct_classes():
    df = _tiny_df()
    arts = train_xgb_multiclass(
        df, label_col="LiaisonId", time_col="DateHeureDepartVoyageSegment"
    )
    assert set(arts.label_encoder.classes_) == {"A", "B", "C"}
```

- [ ] **Step 3.2: Run tests — verify they PASS (baseline, before the hist change)**

```
pytest tests/test_training.py -v
```

Expected: `2 passed` (confirms the test itself is valid before the change).

- [ ] **Step 3.3: Add `tree_method="hist"` to `src/rec_oncf/training.py`**

In `train_xgb_multiclass`, replace the `XGBClassifier` block:

```python
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.08,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
    )
```

- [ ] **Step 3.4: Run tests again — verify they still PASS**

```
pytest tests/test_training.py -v
```

Expected: `2 passed`

- [ ] **Step 3.5: Run all tests**

```
pytest tests/ -v
```

Expected: all tests pass (metrics + features + training).

- [ ] **Step 3.6: Commit**

```bash
git add tests/test_training.py src/rec_oncf/training.py
git commit -m "feat: switch XGBoost to tree_method=hist for native NaN handling"
```

---

## Task 4: Rebuild `features.parquet` + add `mrr@3` to training script

**Files:**
- Modify: `scripts/03_train_ranker.py`

- [ ] **Step 4.1: Rebuild `features.parquet` with new columns**

```
python scripts/02_build_features.py
```

Expected output:
```
Wrote: ...\data\processed\features.parquet
Rows: 491,680
Users: 69,449
Classes (liaisons): 1067
```

- [ ] **Step 4.2: Verify the new columns are in the rebuilt parquet**

```
python -c "import pandas as pd; df = pd.read_parquet('data/processed/features.parquet'); print(df.columns.tolist()); print('cols:', len(df.columns))"
```

Expected: `is_self_purchase` and `TrajetAllerRetour` appear in the list, `cols: 26`.

- [ ] **Step 4.3: Add `mrr@3` to `scripts/03_train_ranker.py`**

Replace the import line and the report block:

```python
from rec_oncf.metrics import hit_rate_at_k, mrr_at_k
```

```python
    hr1 = hit_rate_at_k(y_true, proba, k=1)
    hr3 = hit_rate_at_k(y_true, proba, k=3)
    mrr3 = mrr_at_k(y_true, proba, k=3)

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "classes": int(len(artifacts.label_encoder.classes_)),
        "hit_rate@1": float(hr1),
        "hit_rate@3": float(hr3),
        "mrr@3": float(mrr3),
    }
```

- [ ] **Step 4.4: Commit**

```bash
git add scripts/03_train_ranker.py
git commit -m "feat: add mrr@3 to training script report"
```

---

## Task 5: Run the full training pipeline

- [ ] **Step 5.1: Run training**

```
python scripts/03_train_ranker.py
```

This will take several minutes (491 680 rows, 1 067 classes, 300 trees). Expected output:

```json
{
  "rows_train": 393344,
  "rows_test": 98336,
  "classes": 1067,
  "hit_rate@1": <value>,
  "hit_rate@3": <value>,
  "mrr@3": <value>
}
Saved model: ...\models\xgb_ranker.json
Saved label encoder: ...\models\label_encoder.joblib
Saved report: ...\reports\offline_metrics.json
```

- [ ] **Step 5.2: Verify artifacts exist**

```
python -c "
from pathlib import Path
for p in ['models/xgb_ranker.json', 'models/label_encoder.joblib', 'reports/offline_metrics.json']:
    f = Path(p)
    print(f'{p}: {\"OK\" if f.exists() else \"MISSING\"} ({f.stat().st_size/1024:.0f} KB)' if f.exists() else f'{p}: MISSING')
"
```

- [ ] **Step 5.3: Review metrics**

Open `reports/offline_metrics.json`. Sanity targets:

| Metric | Minimum acceptable | Notes |
|---|---|---|
| `hit_rate@1` | > 0.30 | Common routes dominate — 30% is a real baseline |
| `hit_rate@3` | > 0.50 | Users typically book one of their top-3 routes |
| `mrr@3` | > 0.35 | Should sit between hit@1 and hit@3 |

If all three are above these thresholds, the model is working. If any are below, do not proceed — check the feature matrix shape first (`df.shape` in the script) and ensure `features.parquet` was rebuilt.

- [ ] **Step 5.4: Commit final state**

```bash
git add reports/offline_metrics.json
git commit -m "chore: add offline metrics report from baseline training run"
```

---

## Checklist: Spec Coverage

| Spec requirement | Covered by |
|---|---|
| `tree_method="hist"` for NaN handling | Task 3 |
| `is_self_purchase` feature | Task 2 |
| `TrajetAllerRetour` feature | Task 2 |
| `mrr_at_k` in metrics.py | Task 1 |
| `mrr@3` in training report | Task 4 |
| Rebuild features.parquet (26 cols) | Task 4 |
| `models/pipeline.joblib` + `label_encoder.joblib` | Task 5 (existing save logic) |
| `reports/offline_metrics.json` | Task 5 |
| Temporal split 80/20 | Already in `03_train_ranker.py` — no change needed |
| `CodeClient` excluded from X | Already in `03_train_ranker.py` — no change needed |
