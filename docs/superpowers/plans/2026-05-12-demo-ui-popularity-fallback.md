# Popularity Fallback + ONCF Demo UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The recommender never returns an empty list (global-popularity fallback), enriches responses with human-readable route labels, and ships a polished ONCF-styled web page served by the API for live demos.

**Architecture:** A new tiny `popularity` module + pipeline script produces `models/popularity.joblib` (a list of `LiaisonId` ordered by global frequency). `Recommender` loads it and routes every "cannot serve via model" branch to a `popularity` mode instead of an empty `cold_start`. `Recommender` also builds a `LiaisonId → "GARE → GARE"` lookup from `oncf_clean.parquet` and adds a `labels` dict to every `recommend()` result. The FastAPI app mounts `apps/api/static/` and serves `index.html` at `GET /`; the page is vanilla HTML/CSS/JS (no build step) and calls `POST /recommend`.

**Tech Stack:** Python 3.12, pandas 3.x, joblib, FastAPI/Starlette `StaticFiles`/`FileResponse`, pytest. Front-end: plain HTML5 + CSS + ES module JS, no dependencies.

**Reference docs to skim before starting:** `CLAUDE.md` (project conventions, Pandas-3 dtype warning, Loi 09-08 rule that `code_client` is never logged or put in URLs), the spec at `docs/superpowers/specs/2026-05-12-demo-ui-popularity-fallback-design.md`, and the existing modules `src/rec_oncf/cold_start.py`, `src/rec_oncf/recommender.py`, `apps/api/main.py`, `scripts/05_build_cold_start.py` for patterns.

**Run tests:** `.venv\Scripts\python.exe -m pytest tests/ -v`
**Run one test:** `.venv\Scripts\python.exe -m pytest tests/test_x.py::test_y -v`
**Run a script:** `.venv\Scripts\python.exe scripts/<name>.py`

---

## File Structure

**New files**
- `src/rec_oncf/popularity.py` — build/save/load the global-popularity liaison list (parallels `cold_start.py`).
- `scripts/08_build_popularity.py` — thin pipeline script: `oncf_clean.parquet` → `models/popularity.joblib`.
- `tests/test_popularity.py` — unit tests for the `popularity` module.
- `apps/api/static/index.html` — the demo page markup.
- `apps/api/static/styles.css` — ONCF-styled CSS.
- `apps/api/static/app.js` — fetch + render logic.

**Modified files**
- `src/rec_oncf/config.py` — add `popularity_path` to `Paths` and `default_paths()`.
- `src/rec_oncf/recommender.py` — `popularity` + `liaison_label_lookup` fields; `_fallback`, `_labels_for`, `_recommend_core` helpers; `from_paths`/`from_data`/`_build` load/accept them; drop the module-level `_COLD_START` constant.
- `apps/api/main.py` — mount `StaticFiles`, add `GET /` → `index.html`, add `labels` to `RecommendResponse`.
- `tests/test_recommender.py` — popularity-fallback and `labels` tests.
- `tests/test_api.py` — `GET /` serves HTML; `/recommend` returns a `labels` dict.
- `CLAUDE.md` — document new mode, script 08, artifact, `labels` field, demo UI, bump test count, update "How to run everything".

---

## Task 1: Add `popularity_path` to config

**Files:**
- Modify: `src/rec_oncf/config.py`

- [ ] **Step 1: Add the field and the value**

In `src/rec_oncf/config.py`, add `popularity_path: Path` to the `Paths` dataclass (right after `onnx_model_path: Path`), and in `default_paths()` add `popularity_path = models_dir / "popularity.joblib"` next to the other model paths, and pass `popularity_path=popularity_path` into the `Paths(...)` constructor call.

Resulting dataclass field block:

```python
    models_dir: Path
    xgb_model_path: Path
    label_encoder_path: Path
    cold_start_path: Path
    onnx_model_path: Path
    popularity_path: Path
```

Resulting additions in `default_paths()`:

```python
    onnx_model_path = models_dir / "xgb_ranker.onnx"
    popularity_path = models_dir / "popularity.joblib"
```

```python
        onnx_model_path=onnx_model_path,
        popularity_path=popularity_path,
    )
```

- [ ] **Step 2: Verify it imports**

Run: `.venv\Scripts\python.exe -c "from rec_oncf.config import default_paths; print(default_paths().popularity_path)"`
Expected: prints a path ending in `models\popularity.joblib`, no error.

- [ ] **Step 3: Commit**

```bash
git add src/rec_oncf/config.py
git commit -m "feat: add popularity_path to config"
```

---

## Task 2: `popularity` module (build / save / load)

**Files:**
- Create: `src/rec_oncf/popularity.py`
- Test: `tests/test_popularity.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_popularity.py`:

```python
from __future__ import annotations

import pandas as pd

from rec_oncf.popularity import build_popularity_list, load_popularity, save_popularity


def test_build_popularity_orders_by_descending_frequency():
    df = pd.DataFrame({"LiaisonId": ["A", "A", "A", "B", "B", "C"]})
    assert build_popularity_list(df) == ["A", "B", "C"]


def test_build_popularity_includes_every_liaison_as_str():
    df = pd.DataFrame({"LiaisonId": [10, 20, 30, 20]})
    result = build_popularity_list(df)
    assert set(result) == {"10", "20", "30"}
    assert all(isinstance(x, str) for x in result)


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "sub" / "popularity.joblib"
    save_popularity(["A", "B", "C"], p)
    assert load_popularity(p) == ["A", "B", "C"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_popularity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rec_oncf.popularity'`.

- [ ] **Step 3: Write the module**

Create `src/rec_oncf/popularity.py`:

```python
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def build_popularity_list(clean_df: pd.DataFrame) -> list[str]:
    """LiaisonId values ordered by descending global frequency.

    Cancellations are already removed upstream by cleaning.py, so a plain
    value_counts on the clean frame is the global popularity ranking.
    """
    counts = clean_df["LiaisonId"].astype(str).value_counts()
    return [str(x) for x in counts.index]


def save_popularity(popularity: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump([str(x) for x in popularity], path)


def load_popularity(path: Path) -> list[str]:
    return [str(x) for x in joblib.load(path)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_popularity.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rec_oncf/popularity.py tests/test_popularity.py
git commit -m "feat: popularity module — global liaison frequency list"
```

---

## Task 3: `scripts/08_build_popularity.py` and produce the artifact

**Files:**
- Create: `scripts/08_build_popularity.py`

- [ ] **Step 1: Write the script**

Create `scripts/08_build_popularity.py` (mirrors `scripts/05_build_cold_start.py`):

```python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.popularity import build_popularity_list, save_popularity


def main() -> None:
    paths = default_paths()

    if not paths.processed_dataset_parquet.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {paths.processed_dataset_parquet}\n"
            "Run scripts/01_make_dataset.py first."
        )

    print("Loading oncf_clean.parquet ...")
    clean = read_parquet(paths.processed_dataset_parquet)
    print(f"  {len(clean):,} rows, {clean['LiaisonId'].nunique()} unique routes")

    popularity = build_popularity_list(clean)
    print(f"  Top-10 by frequency: {popularity[:10]}")

    save_popularity(popularity, paths.popularity_path)
    print(f"Saved {len(popularity)} liaisons -> {paths.popularity_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `.venv\Scripts\python.exe scripts/08_build_popularity.py`
Expected: prints row count, a top-10 list, and `Saved <N> liaisons -> ...\models\popularity.joblib`. Confirm the file exists:
Run: `.venv\Scripts\python.exe -c "from rec_oncf.popularity import load_popularity; from rec_oncf.config import default_paths; p=load_popularity(default_paths().popularity_path); print(len(p), p[:3])"`
Expected: prints a count (~1000) and the 3 most popular liaison ids.

- [ ] **Step 3: Commit**

```bash
git add scripts/08_build_popularity.py
git commit -m "feat: script 08 — build models/popularity.joblib"
```

(The `.joblib` artifact itself is gitignored under `models/` — do not commit it.)

---

## Task 4: `Recommender` — popularity fallback + `labels`

**Files:**
- Modify: `src/rec_oncf/recommender.py`
- Test: `tests/test_recommender.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_recommender.py` (the file already has `_make_artifacts()` and `_make_clean_df()` helpers — reuse them):

```python
def test_unknown_user_falls_back_to_popularity():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean, popularity=["C", "A", "B"])
    result = rec.recommend("9999", k=2)
    assert result["mode"] == "popularity"
    assert result["recommendations"] == ["C", "A"]
    assert result["recommendations"]  # never empty


def test_unknown_user_without_popularity_is_cold_start():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean)  # no popularity
    result = rec.recommend("9999", k=2)
    assert result["mode"] == "cold_start"
    assert result["recommendations"] == []


def test_recommend_result_always_has_labels_key():
    arts = _make_artifacts()
    clean = _make_clean_df()
    rec = Recommender.from_data(arts, clean, popularity=["A", "B"])
    for cc in ("1001", "9999"):
        result = rec.recommend(cc, k=2)
        assert "labels" in result
        assert isinstance(result["labels"], dict)


def test_recommend_labels_map_liaison_to_station_pair():
    arts = _make_artifacts()
    clean = _make_clean_df()
    clean["DesignationFrGareDepart"] = "GARE X"
    clean["DesignationFrGareArrive"] = "GARE Y"
    rec = Recommender.from_data(arts, clean, popularity=["A"])
    result = rec.recommend("1001", k=3)
    for lid in result["recommendations"]:
        assert result["labels"][lid] == "GARE X → GARE Y"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recommender.py -v`
Expected: the 4 new tests FAIL (`from_data() got an unexpected keyword argument 'popularity'` / `KeyError: 'labels'`); the 5 pre-existing tests still PASS.

- [ ] **Step 3: Rewrite `src/rec_oncf/recommender.py`**

Replace the whole file with:

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
from rec_oncf.popularity import load_popularity
from rec_oncf.training import FastPreprocessor, TrainArtifacts, load_artifacts, predict_proba, predict_proba_onnx

_LABEL_COLS = ("LiaisonId", "DesignationFrGareDepart", "DesignationFrGareArrive")


def _build_label_lookup(clean_df: pd.DataFrame) -> dict[str, str]:
    """LiaisonId -> "GARE DEPART → GARE ARRIVEE". Empty if columns absent."""
    if not set(_LABEL_COLS) <= set(clean_df.columns):
        return {}
    sub = clean_df.loc[:, list(_LABEL_COLS)].drop_duplicates("LiaisonId")
    return {
        str(dep_arr.LiaisonId): f"{dep_arr.DesignationFrGareDepart} → {dep_arr.DesignationFrGareArrive}"
        for dep_arr in sub.itertuples(index=False)
    }


@dataclass
class Recommender:
    """Two-stage recommender (Candidate Generation + Ranking).

    For users with 1-2 trips, falls back to co-occurrence collaborative
    filtering. For warm users, features are computed ON THE FLY from live
    history and scored via ONNX Runtime (fast path) or sklearn (fallback).
    When the model cannot serve a user at all, falls back to global popularity
    (so the recommendation list is never empty), unless the popularity list is
    itself empty (then the legacy ``cold_start`` / empty-list result).
    Every result carries a ``labels`` dict mapping each recommended LiaisonId to
    a human-readable "GARE → GARE" string (missing ids are simply omitted).
    """
    artifacts: TrainArtifacts
    history_lookup: dict[str, pd.DataFrame]
    cold_start_rec: ColdStartRecommender
    onnx_session: object | None = None        # onnxruntime.InferenceSession
    fast_preprocessor: FastPreprocessor | None = None
    popularity: list[str] = field(default_factory=list)
    liaison_label_lookup: dict[str, str] = field(default_factory=dict)

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
        fast_preprocessor = FastPreprocessor(artifacts.pipeline["pre"])
        popularity = (
            load_popularity(paths.popularity_path) if paths.popularity_path.exists() else []
        )
        return cls._build(
            artifacts, clean, cold_start_rec, onnx_session, fast_preprocessor, popularity
        )

    @classmethod
    def from_data(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,  # kept for backward compat, unused
        *,
        popularity: list[str] | None = None,
    ) -> Recommender:
        cold_start_rec = build_cold_start_recommender(clean_df)
        return cls._build(artifacts, clean_df, cold_start_rec, popularity=popularity)

    @classmethod
    def _build(
        cls,
        artifacts: TrainArtifacts,
        clean_df: pd.DataFrame,
        cold_start_rec: ColdStartRecommender,
        onnx_session: object | None = None,
        fast_preprocessor: FastPreprocessor | None = None,
        popularity: list[str] | None = None,
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
            fast_preprocessor=fast_preprocessor,
            popularity=[str(x) for x in (popularity or [])],
            liaison_label_lookup=_build_label_lookup(clean_df),
        )

    # --- internals -------------------------------------------------------

    def _fallback(self, k: int) -> dict:
        if self.popularity:
            return {"mode": "popularity", "recommendations": list(self.popularity[:k])}
        return {"mode": "cold_start", "recommendations": []}

    def _labels_for(self, recs: list[str]) -> dict[str, str]:
        return {r: self.liaison_label_lookup[r] for r in recs if r in self.liaison_label_lookup}

    def _recommend_core(
        self,
        code_client: str,
        k: int,
        asof: datetime | pd.Timestamp | None,
    ) -> dict:
        history = self.history_lookup.get(code_client)
        if history is None:
            return self._fallback(k)

        if len(history) < 3:
            recs = self.cold_start_rec.recommend(history, k)
            if recs:
                return {"mode": "cold_start_cf", "recommendations": recs}
            return self._fallback(k)

        candidates = generate_candidates(history, user_id=code_client, max_candidates=10)
        if not candidates:
            return self._fallback(k)

        feat_row = compute_inference_row(history, asof=asof)

        le = self.artifacts.label_encoder
        known = set(le.classes_)
        valid_candidates = [c for c in candidates if c in known]
        if not valid_candidates:
            return {"mode": "model", "recommendations": candidates[:k]}

        if self.onnx_session is not None:
            if self.fast_preprocessor is not None:
                row_dict = feat_row.iloc[0].to_dict()
                X_pre = self.fast_preprocessor.encode(row_dict)
                proba = self.onnx_session.run(["probabilities"], {"input": X_pre})[0][0]
            else:
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

    # --- public API ------------------------------------------------------

    def recommend(
        self,
        code_client: str,
        k: int = 1,
        *,
        asof: datetime | pd.Timestamp | None = None,
    ) -> dict:
        code_client = str(code_client)
        k = min(max(k, 1), 3)
        result = self._recommend_core(code_client, k, asof)
        return {**result, "labels": self._labels_for(result["recommendations"])}
```

Note for the implementer: the only behavioural changes are (a) the four old `return _COLD_START` sites now go through `_fallback(k)`, (b) `recommend()` adds a `labels` key, (c) `from_data` and `_build` accept `popularity`. Everything else is verbatim from the previous file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recommender.py -v`
Expected: all 9 tests pass (5 old + 4 new).

- [ ] **Step 5: Run the broader suite to catch fallout**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recommender.py tests/test_api.py tests/test_cold_start.py -v`
Expected: all pass. (`test_api.py` currently checks `body["mode"] == "cold_start"` and `body["recommendations"] == []` for an unknown user — still true because the test fixture builds the recommender without popularity.)

- [ ] **Step 6: Commit**

```bash
git add src/rec_oncf/recommender.py tests/test_recommender.py
git commit -m "feat: popularity fallback + labels in Recommender.recommend()"
```

---

## Task 5: ONCF demo UI (static files)

**Files:**
- Create: `apps/api/static/index.html`
- Create: `apps/api/static/styles.css`
- Create: `apps/api/static/app.js`

> **Use the `frontend-design` skill for this task.** Goal: a polished, production-grade single page in the visual language of the ONCF mobile app — not a generic AI layout.

**Hard constraints (do not deviate):**
- Plain HTML5 + CSS + a single ES-module `app.js`. **No build step, no npm, no CDN, no external fonts/scripts.** The page must work opened as a static file served from `/`.
- The page calls **only** `POST /recommend` (same origin). Request body: `{"code_client": <string>, "k": <1|2|3>, "include_schedule": <bool>}`. `code_client` goes **only in the POST body** — never in the URL/query string, never written to `localStorage`/`sessionStorage`/cookies (Loi 09-08 / CNDP).
- Palette anchored on ONCF brand: orange accent `#F37021`, dark slate `#1B2A4A` (headers/primary text), near-white background `#F7F8FA`, card surface `#FFFFFF`, success green `#1E8E5A`, amber `#C9821A`, neutral grey `#6B7280`. Expose these as CSS custom properties so they're trivially retunable.

**Page contents (spec §B.2/B.3):**
- **Header bar** (slate background, orange accent): wordmark `ONCF` + subtitle `Recommandation de trajet — démo`.
- **Form card**: text input `Code client`; a small segmented control or `<select>` for `k` (options 1 / 2 / 3, default 3); a checkbox `Afficher les horaires`; a primary button `Recommander`.
- **Results region**:
  - One **route card** per item in `recommendations` (in order): a large `GARE DEP → GARE ARR` headline (from `labels[liaisonId]`; if absent, show `Liaison <id>`), a rank pill (`#1`, `#2`, `#3`).
  - If `schedules` is present in the response and `schedules[liaisonId]` is a non-empty list, render under that card a compact list of upcoming departures. Each schedule item is an object whose keys may include `depart`, `arrive`, `train` — render whatever string values are present, gracefully (don't assume all keys exist). If the list is empty, render a muted `Horaires indisponibles` line.
  - A **status footer** under the cards: a colored **mode badge** — `model` → green, `cold_start_cf` → amber, `cold_start` → grey, `popularity` → grey (label it e.g. `Modèle`, `Démarrage à froid (CF)`, `Aucune reco`, `Populaire`) — plus `request_id` (monospace, small) and the **client-side round-trip latency in ms** (measured with `performance.now()` around the fetch).
- **States**:
  - Empty `code_client` → inline validation message ("Saisissez un code client"), no request sent.
  - In-flight → button disabled + a spinner / "Recherche…" text.
  - Network or non-2xx response → an error banner showing the failure (status text / message), form stays usable.
  - 2xx with `recommendations == []` (only possible if `popularity.joblib` is absent) → friendly "Aucune recommandation disponible pour ce client." message instead of cards.
- Responsive: usable from ~360px wide up to desktop. Accessible: labels tied to inputs, button has an accessible name, sufficient contrast.

- [ ] **Step 1: Create `apps/api/static/` with `index.html`, `styles.css`, `app.js`** per the constraints above (use the `frontend-design` skill).

- [ ] **Step 2: Smoke-check the markup is valid standalone**

Run: `.venv\Scripts\python.exe -c "import pathlib,html.parser; p=pathlib.Path('apps/api/static/index.html').read_text(encoding='utf-8'); print('len', len(p)); html.parser.HTMLParser().feed(p); print('parsed ok')"`
Expected: prints a length and `parsed ok`, no exception. (Functional verification happens in Task 6 once the route is wired.)

- [ ] **Step 3: Commit**

```bash
git add apps/api/static/index.html apps/api/static/styles.css apps/api/static/app.js
git commit -m "feat: ONCF-styled demo UI (static page)"
```

---

## Task 6: Wire the UI into FastAPI + `labels` in the response model

**Files:**
- Modify: `apps/api/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "ONCF" in resp.text


def test_static_assets_are_served(client):
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


def test_recommend_response_has_labels_dict(client):
    resp = client.post("/recommend", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert "labels" in body
    assert isinstance(body["labels"], dict)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: the 3 new tests FAIL (`GET /` → 404; `labels` not in body). Pre-existing `test_api.py` tests still pass.

> If `test_api.py` import itself fails with `RuntimeError: Directory 'apps/api/static' does not exist`, Task 5 was skipped — do Task 5 first. (The `app.mount` call below runs at import time.)

- [ ] **Step 3: Edit `apps/api/main.py`**

Add imports near the other FastAPI imports:

```python
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
```

Add a `STATIC_DIR` constant next to `PROJECT_ROOT` / `SRC_DIR`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
```

Immediately after `app = FastAPI(title="ONCF Recommender", lifespan=lifespan)`, mount the static dir:

```python
app = FastAPI(title="ONCF Recommender", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

Add a root route (place it just above `@app.get("/health")`):

```python
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
```

Add the `labels` field to `RecommendResponse`:

```python
class RecommendResponse(BaseModel):
    mode: str
    variant: str
    request_id: str
    recommendations: list[str]
    labels: dict[str, str] = {}
    schedules: dict[str, list[dict[str, str]]] | None = None
```

(No change is needed in the `recommend()` handler body: `rec.recommend(...)` already returns a `labels` key, which flows through `dict(...)` into the response. `response_model_exclude_none=True` keeps an empty `labels` dict in the payload because `{}` is not `None`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: all tests pass (pre-existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add apps/api/main.py tests/test_api.py
git commit -m "feat: serve demo UI at / and add labels to /recommend response"
```

---

## Task 7: Full verification, docs, manual smoke test

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`
Expected: all tests green. Note the new total (was 100; +3 popularity, +4 recommender, +3 api → expect ~110). Record the exact number for the docs.

- [ ] **Step 2: Manual smoke test of the running service**

Start the API in a terminal:
`.venv\Scripts\python.exe -m uvicorn apps.api.main:app --port 8123`
Wait for startup to finish (it loads the 281 MB model + ONNX — ~30-40 s; you'll see the "Challenger model not found" and "Redis unavailable" warnings, then it's ready).

Then, in another shell:
- `Invoke-RestMethod http://127.0.0.1:8123/health` → `{status=ok}`
- Open `http://127.0.0.1:8123/` in a browser → the ONCF demo page renders.
- In the page: enter a known client (find one with `.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_parquet('data/processed/oncf_clean.parquet', columns=['CodeClient'])['CodeClient'].value_counts().index[0])"`), k=3, click "Recommander" → 3 route cards with `GARE → GARE` headlines, a green `Modèle` badge, a `request_id`, a latency in ms.
- Enter a nonsense client (e.g. `__nobody__`) → a `Populaire` badge and 3 popularity route cards (NOT an empty list).
- Tick "Afficher les horaires", re-run for the known client → schedule lines appear (or "Horaires indisponibles" if scraping returned nothing).
- Confirm the browser URL stays `http://127.0.0.1:8123/` (no `?code_client=...`).

Stop the server (Ctrl-C / `Stop-Process`).

- [ ] **Step 3: Update `CLAUDE.md`**

Make these edits:
- **Repository Layout**: add `scripts/08_build_popularity.py` (`oncf_clean → models/popularity.joblib ✅ done`), add `src/rec_oncf/popularity.py` to the `src/rec_oncf/` list, add `tests/test_popularity.py` (3 tests), add `apps/api/static/` (`index.html`, `styles.css`, `app.js — demo web page`), bump `test_recommender.py` to 9 tests and `test_api.py` to its new count, update the test-suite total in the `tests/` header and the "Current Status" row.
- **API section**: under `/recommend`, document that the response now also includes `"labels": {liaison_id: "GARE → GARE", ...}`; add a new mode value `"popularity"` to the `mode` enum description; add a line: "Root path `GET /` serves a self-contained ONCF-styled demo page (static files in `apps/api/static/`); `GET /static/*` serves its assets."
- **`Recommender` section**: in the `recommend()` logic list, change the cold-start/empty branches to say they now return `{"mode": "popularity", ...}` from the global-popularity list (loaded from `models/popularity.joblib`) when that list is available, falling back to the old empty `cold_start` only if the artifact is absent; mention the `labels` enrichment; add `popularity: list[str]` and `liaison_label_lookup: dict[str,str]` to the "in-memory lookups" note.
- **Artifact Paths**: add `popularity_path = models_dir / "popularity.joblib"`.
- **Model Architecture / Cold-start rule**: update the cold-start sentence to mention the popularity fallback so it's never an empty response.
- **What's Left to Do** + **Current Status**: add a "Popularity fallback + demo UI" row marked ✅ done; if the empty-list behaviour was listed as a known gap anywhere, mark it resolved.
- **How to Run Everything**: insert after step 6 (ONNX): `# 7. Build global-popularity fallback list (~5 s)\n.venv\Scripts\python.exe scripts/08_build_popularity.py` and renumber the following steps.

- [ ] **Step 4: Re-run the suite one last time**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: popularity fallback + demo UI in CLAUDE.md"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** A.1→Task 2+3; A.2→Task 4; A.3 (labels)→Task 4 (`_labels_for`, `recommend()`) + Task 6 (response model); A.4→Tasks 2 & 4 tests; A.5→Task 7. B.1→Task 6; B.2/B.3/B.4→Task 5; B.5→Task 6. Non-goals untouched. No gaps.
- **`_COLD_START` removal:** the old module-level constant is deleted; nothing imports it (grep `_COLD_START` after Task 4 — only `recommender.py` referenced it, and `_fallback` now produces the equivalent dict literal).
- **Backward compat:** `from_data(arts, clean)` still works (popularity is keyword-only, defaults to `[]`); the API test fixture relies on this. `compute_inference_row`, ONNX path, etc. are byte-for-byte unchanged.
- **`StaticFiles` import-time check:** `app.mount("/static", StaticFiles(directory=...))` raises at import if the dir is missing — that's why Task 5 (create the dir) precedes Task 6 (the mount). The "if import fails" note in Task 6 Step 2 covers an out-of-order run.
- **`response_model_exclude_none=True`:** keeps `labels` (a `{}` dict, not `None`) in the payload while still dropping `schedules` when absent — matches the existing `test_recommend_no_schedules_field_by_default`.
