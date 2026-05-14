# Cold-Start Collaborative Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the empty cold-start response for users with 1–2 bookings with route co-occurrence collaborative filtering, improving HR@1 on the 44,397-row cold-start segment (currently 73.93%).

**Architecture:** Precompute a route co-occurrence lookup offline (users who traveled route A also traveled route B) and save it to disk. At inference, users with 1–2 trips get recommendations from that lookup instead of an empty response. Users with 0 trips (truly unknown) still return the existing `cold_start` mode. The XGBoost model is not involved in cold-start CF predictions.

**Tech Stack:** pandas, joblib, FastAPI (existing), pytest

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `src/rec_oncf/cold_start.py` | Build co-occurrence lookup + `ColdStartRecommender` dataclass |
| Modify | `src/rec_oncf/config.py` | Add `cold_start_path` to `Paths` and `default_paths()` |
| Create | `scripts/05_build_cold_start.py` | CLI script: `oncf_clean.parquet` → `models/cold_start.joblib` |
| Modify | `src/rec_oncf/recommender.py` | Wire `ColdStartRecommender` into `recommend()` for 1–2 trip users |
| Create | `tests/test_cold_start.py` | Unit tests for co-occurrence logic and `ColdStartRecommender` |
| Modify | `tests/test_recommender.py` | Add test for 1–2 trip user returning `cold_start_cf` mode |

---

## Task 1: `cold_start.py` — co-occurrence logic (TDD)

**Files:**
- Create: `tests/test_cold_start.py`
- Create: `src/rec_oncf/cold_start.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cold_start.py
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from rec_oncf.cold_start import (
    ColdStartRecommender,
    build_cold_start_recommender,
    load_cold_start,
    save_cold_start,
)


def _make_clean() -> pd.DataFrame:
    # 3 users:
    #   user A: routes [X, Y, Z]
    #   user B: routes [X, Y]
    #   user C: routes [Z, W]
    rows = [
        ("A", "X"), ("A", "Y"), ("A", "Z"),
        ("B", "X"), ("B", "Y"),
        ("C", "Z"), ("C", "W"),
    ]
    df = pd.DataFrame(rows, columns=["CodeClient", "LiaisonId"])
    df["DateHeureDepartVoyageSegment"] = pd.date_range("2020-01-01", periods=len(df))
    return df


def test_global_top_order():
    rec = build_cold_start_recommender(_make_clean())
    # X appears 2x, Y 2x, Z 2x, W 1x — top should be length >= 1
    assert len(rec.global_top) >= 1
    assert rec.global_top[0] in {"X", "Y", "Z"}  # all tied at 2, W is last


def test_global_top_excludes_nothing():
    rec = build_cold_start_recommender(_make_clean())
    assert "W" in rec.global_top  # low freq routes still included


def test_cooccurrence_x_contains_y():
    rec = build_cold_start_recommender(_make_clean())
    # X co-occurs with Y in 2 users (A and B) — Y must be top co-route for X
    assert "Y" in rec.cooccurrence.get("X", [])


def test_cooccurrence_z_contains_w():
    rec = build_cold_start_recommender(_make_clean())
    # Z co-occurs with W only via user C
    assert "W" in rec.cooccurrence.get("Z", [])


def test_recommend_no_history_returns_global_top():
    rec = build_cold_start_recommender(_make_clean())
    result = rec.recommend(history_df=None, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_recommend_with_history_uses_cooccurrence():
    rec = build_cold_start_recommender(_make_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=3)
    # X co-occurs with Y and Z — both should appear
    assert "Y" in result or "Z" in result


def test_recommend_k_respected():
    rec = build_cold_start_recommender(_make_clean())
    history = pd.DataFrame({"LiaisonId": ["X"]})
    result = rec.recommend(history_df=history, k=1)
    assert len(result) == 1


def test_recommend_unknown_route_falls_back_to_global():
    rec = build_cold_start_recommender(_make_clean())
    # Route "UNKNOWN" has no co-occurrence data
    history = pd.DataFrame({"LiaisonId": ["UNKNOWN"]})
    result = rec.recommend(history_df=history, k=2)
    assert len(result) <= 2
    assert all(r in rec.global_top for r in result)


def test_save_load_roundtrip(tmp_path):
    rec = build_cold_start_recommender(_make_clean())
    path = tmp_path / "cold_start.joblib"
    save_cold_start(rec, path)
    loaded = load_cold_start(path)
    assert loaded.global_top == rec.global_top
    assert loaded.cooccurrence == rec.cooccurrence
```

- [ ] **Step 2: Run tests — expect all to fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cold_start.py -v
```
Expected: `ImportError` — `cold_start` module does not exist yet.

- [ ] **Step 3: Implement `cold_start.py`**

```python
# src/rec_oncf/cold_start.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class ColdStartRecommender:
    cooccurrence: dict[str, list[str]]  # route -> top co-occurring routes
    global_top: list[str]               # routes ranked by global popularity

    def recommend(self, history_df: pd.DataFrame | None, k: int) -> list[str]:
        if history_df is None or history_df.empty:
            return self.global_top[:k]

        known_routes = history_df["LiaisonId"].astype(str).unique().tolist()

        scores: Counter[str] = Counter()
        for route in known_routes:
            for co_route in self.cooccurrence.get(route, []):
                scores[co_route] += 1

        if not scores:
            return self.global_top[:k]

        return [r for r, _ in scores.most_common(k)]


def build_route_cooccurrence(clean_df: pd.DataFrame, top_n: int = 20) -> dict[str, list[str]]:
    user_routes = (
        clean_df.groupby("CodeClient")["LiaisonId"]
        .apply(lambda s: s.astype(str).unique().tolist())
    )
    cooc: dict[str, Counter[str]] = {}
    for routes in user_routes:
        for route in routes:
            if route not in cooc:
                cooc[route] = Counter()
            for other in routes:
                if other != route:
                    cooc[route][other] += 1

    return {
        route: [r for r, _ in counter.most_common(top_n)]
        for route, counter in cooc.items()
    }


def build_global_top(clean_df: pd.DataFrame, n: int = 20) -> list[str]:
    return (
        clean_df["LiaisonId"]
        .astype(str)
        .value_counts()
        .head(n)
        .index.tolist()
    )


def build_cold_start_recommender(clean_df: pd.DataFrame) -> ColdStartRecommender:
    return ColdStartRecommender(
        cooccurrence=build_route_cooccurrence(clean_df),
        global_top=build_global_top(clean_df),
    )


def save_cold_start(rec: ColdStartRecommender, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rec, path)


def load_cold_start(path: Path) -> ColdStartRecommender:
    return joblib.load(Path(path))
```

- [ ] **Step 4: Run tests — expect all to pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cold_start.py -v
```
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/rec_oncf/cold_start.py tests/test_cold_start.py
git commit -m "feat: add ColdStartRecommender with route co-occurrence CF"
```

---

## Task 2: Add `cold_start_path` to config

**Files:**
- Modify: `src/rec_oncf/config.py`

- [ ] **Step 1: Add `cold_start_path` to the `Paths` dataclass and `default_paths()`**

In `src/rec_oncf/config.py`, update `Paths` (add the field after `label_encoder_path`):

```python
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
```

And in `default_paths()` (add before the `return` statement):

```python
    cold_start_path = models_dir / "cold_start.joblib"

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
    )
```

- [ ] **Step 2: Verify the full test suite still passes**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all existing tests + 9 cold_start tests pass.

- [ ] **Step 3: Commit**

```powershell
git add src/rec_oncf/config.py
git commit -m "feat: add cold_start_path to Paths config"
```

---

## Task 3: Script `05_build_cold_start.py`

**Files:**
- Create: `scripts/05_build_cold_start.py`

- [ ] **Step 1: Create the script**

```python
# scripts/05_build_cold_start.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.cold_start import build_cold_start_recommender, save_cold_start
from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet


def main() -> None:
    paths = default_paths()

    if not paths.processed_dataset_parquet.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {paths.processed_dataset_parquet}\n"
            "Run scripts/01_make_dataset.py first."
        )

    print("Loading oncf_clean.parquet …")
    clean = read_parquet(paths.processed_dataset_parquet)

    print(f"  {len(clean):,} rows, {clean['LiaisonId'].nunique()} unique routes")
    print("Building co-occurrence lookup …")
    rec = build_cold_start_recommender(clean)

    n_routes = len(rec.cooccurrence)
    print(f"  {n_routes} routes with co-occurrence data")
    print(f"  Global top-3: {rec.global_top[:3]}")

    save_cold_start(rec, paths.cold_start_path)
    print(f"Saved → {paths.cold_start_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```powershell
.venv\Scripts\python.exe scripts/05_build_cold_start.py
```
Expected output (exact numbers may vary):
```
Loading oncf_clean.parquet …
  491,680 rows, 1011 unique routes
Building co-occurrence lookup …
  1011 routes with co-occurrence data
  Global top-3: ['...', '...', '...']
Saved → ...\models\cold_start.joblib
```

- [ ] **Step 3: Commit**

```powershell
git add scripts/05_build_cold_start.py
git commit -m "feat: add script 05 to build and save cold-start CF lookup"
```

---

## Task 4: Wire `ColdStartRecommender` into `Recommender`

**Files:**
- Modify: `src/rec_oncf/recommender.py`
- Modify: `tests/test_recommender.py`

- [ ] **Step 1: Write the new failing test in `test_recommender.py`**

Add this test at the bottom of `tests/test_recommender.py` (after the existing tests):

```python
def test_cold_start_few_trips_returns_cf_mode():
    """A user with 1-2 trips should get cold_start_cf, not an empty cold_start."""
    arts = _make_artifacts()
    # user 2001 has only 2 trips
    dates = pd.date_range("2020-01-01", periods=2, freq="30D")
    clean = _make_clean_df()
    extra = pd.DataFrame({
        "CodeClient": [2001, 2001],
        "AchteurId":  [2001, 2001],
        "LiaisonId":  ["A", "B"],
        "DateHeureDepartVoyageSegment": list(dates),
    })
    for col in ["TypeParcoursId", "ClassificationId", "ClassePhysiqueId",
                "NiveauPrixId", "TrainAutocarId", "CarteClientId"]:
        extra[col] = 1
    extra["PrixParLiaison"] = 100.0
    extra["NbrVoySegment"] = 1.0
    extra["DelaiAnticipation"] = 5.0
    full_clean = pd.concat([clean, extra], ignore_index=True)

    rec = Recommender.from_data(arts, full_clean)
    result = rec.recommend("2001", k=1)
    assert result["mode"] == "cold_start_cf"
    assert len(result["recommendations"]) >= 1
```

- [ ] **Step 2: Run the new test — expect it to fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_recommender.py::test_cold_start_few_trips_returns_cf_mode -v
```
Expected: `FAILED` — mode is `cold_start`, not `cold_start_cf`.

- [ ] **Step 3: Update `recommender.py`**

Replace the full content of `src/rec_oncf/recommender.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from rec_oncf.candidates import generate_candidates
from rec_oncf.cold_start import ColdStartRecommender, build_cold_start_recommender, load_cold_start
from rec_oncf.config import Paths
from rec_oncf.features import compute_inference_row
from rec_oncf.io import read_parquet
from rec_oncf.training import TrainArtifacts, load_artifacts, predict_proba

_COLD_START = {"mode": "cold_start", "recommendations": []}


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    For users with 1-2 trips, falls back to co-occurrence collaborative
    filtering instead of returning an empty cold_start response.
    Features for warm users are computed ON THE FLY from live history.
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender

    @classmethod
    def from_paths(cls, paths: Paths) -> Recommender:
        artifacts = load_artifacts(
            model_path=paths.xgb_model_path,
            label_encoder_path=paths.label_encoder_path,
        )
        clean = read_parquet(paths.processed_dataset_parquet)
        cold_start_rec = load_cold_start(paths.cold_start_path)
        return cls._build(artifacts, clean, cold_start_rec)

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
    ) -> Recommender:
        history_lookup = {
            str(cid): grp.sort_values("DateHeureDepartVoyageSegment")
            for cid, grp in clean_df.groupby("CodeClient")
        }
        return cls(
            artifacts=artifacts,
            history_lookup=history_lookup,
            cold_start_rec=cold_start_rec,
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

        proba = predict_proba(self.artifacts, feat_row, label_col="LiaisonId")[0]
        cand_idx = le.transform(np.asarray(valid_candidates))
        cand_scores = proba[cand_idx]

        order = np.argsort(-cand_scores)[:k]
        recs = [valid_candidates[i] for i in order]
        return {"mode": "model", "recommendations": recs}
```

- [ ] **Step 4: Run the full test suite — expect all to pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all tests pass (previous cold_start tests still pass — unknown user still returns `cold_start` mode).

- [ ] **Step 5: Commit**

```powershell
git add src/rec_oncf/recommender.py tests/test_recommender.py
git commit -m "feat: wire ColdStartRecommender into Recommender for 1-2 trip users"
```

---

## Task 5: Offline evaluation of the improvement

**Goal:** Quantify HR@1 improvement on the cold-start segment (users with 1–2 trips in train).

- [ ] **Step 1: Run the existing test suite one final time**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Expected: all tests green.

- [ ] **Step 2: Quick sanity check on the live API**

Start the API in one terminal:
```powershell
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

In another terminal, call with a cold-start user (adapt `code_client` to a real user with 1–2 trips in `oncf_clean.parquet`):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" -Method Post -ContentType "application/json" -Body '{"code_client": "<id_with_few_trips>", "k": 3}'
```
Expected: response with `"mode": "cold_start_cf"` and non-empty `recommendations`.

- [ ] **Step 3: Final commit + push**

```powershell
git add .
git commit -m "feat: cold-start CF complete — 1-2 trip users now get route co-occurrence recommendations"
git push origin main
```

---

## Self-Review Checklist

- [x] All 6 files mapped with exact paths
- [x] No TBD / placeholder steps
- [x] Types consistent: `ColdStartRecommender`, `build_cold_start_recommender`, `save_cold_start`, `load_cold_start` used identically across tasks
- [x] `Paths.cold_start_path` added in Task 2 and used in Task 4 (`from_paths`)
- [x] Existing test `test_cold_start_unknown_user` still passes — unknown users (not in `history_lookup`) still return `cold_start` mode
- [x] TDD respected: failing test written before implementation in every task
- [x] Frequent commits: one per task
