# CLAUDE.md — ONCF Zero-Click Search Recommender

> **Keep this file up to date.** Update it every time a module is added/changed, a bug is fixed, a script is run, or the status of any work item changes.

---

## Project Goal

Build a proactive route recommender ("zero-click search") for ONCF (Morocco's national railway).
The system predicts the most likely **O/D pair (LiaisonId)** a user will book next, based on their booking history and the context of the app launching, and returns the top-1 or top-3 recommendations to the mobile app via a REST API. The app shows them to the user without the user having to search manually.

**Privacy law:** Loi 09-08 / CNDP (Morocco). `CodeClient` is the lookup key only — it is never included as a model feature and must never be logged in API responses or request logs.

---

## Environment

- **OS:** Windows 11 Pro (PowerShell is the default shell — use PowerShell syntax everywhere)
- **Python:** 3.12.10, venv at `.venv\`
- **GPU:** NVIDIA RTX 3050 (4 GB VRAM) — CUDA available but **not used for training** (see Model Architecture)
- **Key deps:** pandas 3.x, xgboost 3.2.0, scikit-learn, fastapi 0.115, pydantic v2, uvicorn, joblib, requests, beautifulsoup4, redis

**Run Python:** `.venv\Scripts\python.exe`
**Run tests:** `.venv\Scripts\python.exe -m pytest tests/ -v`
**Run a script:** `.venv\Scripts\python.exe scripts/<name>.py`

> **Pandas 3 warning:** `dtype == object` returns `False` for string columns (uses `StringDtype`). Always use `pd.api.types.is_numeric_dtype()` / `pd.api.types.is_string_dtype()` for dtype checks. This affects cat_cols detection in `training.py` and test assertions in `test_features.py`.

---

## Repository Layout

```
Rec_ONCF/
├── src/rec_oncf/           # All library code (importable as rec_oncf.*)
│   ├── cleaning.py         # make_clean_dataset() — raw CSV → oncf_clean.parquet
│   ├── config.py           # default_paths() → Paths dataclass
│   ├── io.py               # read_csv / read_parquet / write_parquet / write_csv
│   ├── features.py         # build_training_rows() — clean → features.parquet (26 cols)
│   ├── metrics.py          # hit_rate_at_k(), mrr_at_k()
│   ├── training.py         # temporal_split, train_xgb_multiclass, predict_proba,
│   │                       #   save_artifacts, load_artifacts, top_k_labels
│   ├── candidates.py       # generate_candidates() — user history → candidate LiaisonIds
│   ├── recommender.py      # Recommender dataclass — from_paths / from_data / recommend()
│   ├── schedule.py         # ONCF live schedule scraping — STATION_CODES, build_liaison_station_map,
│   │                       #   fetch_departures, get_schedule (Redis/memory cache, HTML parsing)
│   ├── popularity.py       # build_popularity_list() — LiaisonIds ordered by global frequency;
│   │                       #   save_popularity / load_popularity
│   └── __init__.py
│
├── apps/
│   ├── __init__.py
│   └── api/
│       ├── __init__.py
│       ├── main.py         # FastAPI app — GET /, GET /health, POST /recommend, POST /feedback
│       └── static/         # Demo web page assets (served at GET /static/*)
│           ├── index.html  # ONCF-styled single-page demo
│           ├── styles.css
│           └── app.js
│
├── scripts/
│   ├── 01_make_dataset.py  # raw CSVs → data/processed/oncf_clean.parquet  ✅ done
│   ├── 02_build_features.py# oncf_clean → data/processed/features.parquet  ✅ done
│   ├── 03_train_ranker.py  # features → models/ + reports/offline_metrics.json  ✅ done
│   ├── 04_baselines.py     # baselines → reports/baseline_metrics.json  ✅ done
│   ├── 05_build_cold_start.py # oncf_clean → models/cold_start.joblib  ✅ done
│   ├── 06_export_onnx.py   # xgb_ranker.json → models/xgb_ranker.onnx + benchmark  ✅ done
│   ├── 07_retrain.py       # full retrain + KPI guardrail → promote models/  ✅ done
│   ├── 08_build_popularity.py # oncf_clean → models/popularity.joblib  ✅ done
│   └── _doc_gen.py         # utility — prints dataset stats to stdout (not part of pipeline)
│
├── tests/
│   ├── test_candidates.py  # 11 tests  ✅ passing
│   ├── test_features.py    # 12 tests  ✅ passing
│   ├── test_metrics.py     # 4 tests   ✅ passing
│   ├── test_recommender.py # 9 tests   ✅ passing
│   ├── test_training.py    # 2 tests   ✅ passing
│   ├── test_cleaning.py    # 5 tests   ✅ passing  (cancellation propagation, cold start, join, etc.)
│   ├── test_api.py         # 18 tests  ✅ passing  (FastAPI TestClient — /health, /recommend, validation, schedule enrichment, variant routing, /feedback, labels, popularity fallback, demo page)
│   ├── test_schedule.py    # 14 tests  ✅ passing  (station codes, HTML parser, HTTP mock, caching)
│   ├── test_cold_start.py  # 9 tests   ✅ passing  (co-occurrence, recommend, save/load)
│   ├── test_onnx.py        # 7 tests   ✅ passing  (export, proba parity, output shape, FastPreprocessor)
│   ├── test_retrain.py     # 16 tests  ✅ passing  (load_metrics, guardrail, evaluate, promote, pipeline, write_challenger)
│   ├── test_popularity.py  # 3 tests   ✅ passing  (build, save/load, order by frequency)
│   └── __init__.py
│
├── data/processed/
│   ├── oncf_clean.parquet  # 491,680 rows, produced by script 01  ✅ exists
│   └── features.parquet    # 491,680 rows × 26 cols, produced by script 02  ✅ exists
│
├── models/                 # ✅ populated — training completed 2026-05-03
│   ├── xgb_ranker.json     # 281 MB — saved with joblib despite .json ext (do NOT change)
│   ├── label_encoder.joblib# 8.5 KB
│   ├── cold_start.joblib   # 31 KB — co-occurrence lookup (produced by script 05)
│   ├── xgb_ranker.onnx     # 148 MB — ONNX export (produced by script 06)
│   └── popularity.joblib   # ~120 KB — global popularity fallback list (produced by script 08)
│
├── reports/
│   ├── cleaning_report.json
│   ├── cleaning_provenance.parquet
│   └── offline_metrics.json  # ✅ generated — see Offline Metrics section below
│
├── docs/superpowers/
│   ├── specs/2026-05-02-model-training-design.md
│   ├── specs/2026-05-03-api-design.md
│   ├── plans/2026-05-02-model-training.md
│   └── plans/2026-05-03-api.md
│
└── pyproject.toml          # pythonpath = ["src"] set for pytest
```

---

## Data

| File | Rows | Description |
|---|---|---|
| `Desktop/oncf_data.csv` | raw | Raw ONCF bookings CSV (on user's Desktop) |
| `Desktop/Liaison.csv` | raw | Route lookup table (on user's Desktop) |
| `data/processed/oncf_clean.parquet` | 491,680 | Cleaned bookings with all original fields |
| `data/processed/features.parquet` | 491,680 | Model-ready feature table (26 cols, see below) |

**Key stats:** 69,449 active users, 1,011 unique `LiaisonId` classes (after temporal split filtering).

---

## Feature Table Schema (`features.parquet` — 26 columns, since Sprint 2)

| Column | Type | Notes |
|---|---|---|
| `CodeClient` | str | User ID — lookup key only, **never a model feature** |
| `DateHeureDepartVoyageSegment` | datetime64[ns] | Departure datetime — used for temporal split, never a feature |
| `LiaisonId` | str | **Target label** — O/D route identifier |
| `TypeParcoursId` | str (Ordinal) | Trip type |
| `ClassificationId` | str (Ordinal) | Booking classification |
| `ClassePhysiqueId` | str (Ordinal) | Physical class |
| `NiveauPrixId` | str (Ordinal) | Price tier |
| `TrainAutocarId` | str (Ordinal) | Train/coach indicator |
| `CarteClientId` | str (Ordinal) | Client card type |
| `prev_liaison` | str (Ordinal) | Previous route taken (NaN → `"nan"`) |
| `user_top_liaison_share` | float64 | Share of past trips on user's most-frequent liaison (NaN for first trip). Captures user "loyalty". |
| `PrixParLiaison` | float64 | Price per route (nullable) |
| `NbrVoySegment` | float64 | Number of journey segments |
| `DelaiAnticipation` | float64 | Days booked in advance |
| `user_trip_index` | int64 | Cumulative trip count per user |
| `days_since_prev` | float64 | Days since last booking (NaN for first trip) |
| `depart_hour` | int32 | Hour of departure |
| `depart_dow` | int32 | Day of week (0=Mon) |
| `depart_month` | int32 | Month |
| `depart_hour_sin/cos` | float64 | Cyclic encoding of hour |
| `depart_dow_sin/cos` | float64 | Cyclic encoding of day of week |
| `depart_month_sin/cos` | float64 | Cyclic encoding of month |
| `is_self_purchase` | int64 | 1 if AchteurId == CodeClient |

---

## Model Architecture

**Algorithm:** XGBoost multiclass (`multi:softprob`), sklearn Pipeline
**Split:** temporal — 80% train / 20% test by `DateHeureDepartVoyageSegment` (393,344 train / 98,261 test after filtering unseen labels)
**Preprocessing:** `ColumnTransformer` — `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` for 7 cat cols, passthrough for 15 num cols
**Cold-start rule:** if user has < 3 bookings in history (or is unknown / produces no candidates) → fall back to `mode: "popularity"` (top-k global frequency list from `popularity.joblib`). Only returns empty `mode: "cold_start"` if `popularity.joblib` is absent.

**Why OrdinalEncoder (not OHE):** `prev_liaison` has 1,011 unique values — OHE would explode the feature matrix to 5,000+ columns. OrdinalEncoder keeps it at ~23 columns.

**Why CPU (not CUDA):** RTX 3050 has 4 GB VRAM which OOMs at 1,011 classes × depth 8 × 300 estimators. CPU training is stable and completes in ~43 minutes.

**XGBoost hyperparameters:**
```python
xgb.XGBClassifier(
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
```

**Target metrics (thresholds from spec):**
- `hit_rate@1 > 0.30` (primary)
- `hit_rate@3 > 0.50`
- `mrr@3 > 0.35`

---

## Offline Metrics (Sprint 2 retrain — 2026-05-04)

| Metric | Sprint 1 | **Sprint 2** | Δ | Threshold | Pass |
|---|---|---|---|---|---|
| `hit_rate@1` | 0.7395 | **0.7628** | +2.33 pp | > 0.30 | ✅ |
| `hit_rate@3` | 0.8877 | **0.9055** | +1.78 pp | > 0.50 | ✅ |
| `mrr@3`      | 0.8064 | **0.8277** | +2.13 pp | > 0.35 | ✅ |

Improvement driven by the new `user_top_liaison_share` feature.

- Train rows: 393,344 — Test rows: 98,261 (75 dropped — labels unseen in train)
- Classes: 1,011 — Training time: ~43 min on CPU
- random_state: 42 (reproducible)
- dataset_fingerprint: `4d0dfd12e0b60341`

### Metrics by user history size (Sprint 2)

| Segment | n test rows | HR@1 | HR@3 | MRR@3 |
|---|---|---|---|---|
| 0-2 trips in train (mostly new in test) | 44,397 | 0.7393 | 0.8930 | 0.8091 |
| 3-5 trips (new active) | 16,780 | 0.7517 | 0.9038 | 0.8210 |
| 6-20 trips (regular) | 23,737 | 0.7786 | 0.9159 | 0.8411 |
| 21+ trips (loyal) | 13,347 | **0.8268** | **0.9311** | **0.8741** |

Monotonic improvement with history depth — exactly the expected pattern.

## Baselines (Sprint 2 — `reports/baseline_metrics.json`)

| Model | HR@1 | HR@3 | MRR@3 |
|---|---|---|---|
| `global_top` (popularity only) | 0.0399 | 0.1125 | 0.0707 |
| `prev_liaison` (last seen) | 0.2620 | 0.3204 | 0.2881 |
| `most_frequent` (user freq + recency) | 0.2751 | 0.5128 | 0.3865 |
| **xgboost_multiclass (Sprint 2)** | **0.7628** | **0.9055** | **0.8277** |

XGBoost is **2.77× better** than the best baseline (`most_frequent`) on HR@1.

---

## Artifact Paths (from `config.py`)

```python
models_dir             = <project_root>/models/
xgb_model_path         = models_dir / "xgb_ranker.json"     # saved with joblib
label_encoder_path     = models_dir / "label_encoder.joblib"
cold_start_path        = models_dir / "cold_start.joblib"
onnx_model_path        = models_dir / "xgb_ranker.onnx"     # 148 MB, produced by script 06
popularity_path        = models_dir / "popularity.joblib"    # ~120 KB, produced by script 08
features_parquet       = data/processed/features.parquet
processed_dataset_parquet = data/processed/oncf_clean.parquet
```

> **Note:** `xgb_ranker.json` is saved using `joblib.dump` (not XGBoost native JSON), despite the `.json` extension. Do not change the filename or save/load method independently.

---

## API (`apps/api/main.py`)

FastAPI app — **written, model available, ready to start.**

**Endpoints:**
- `GET /` → ONCF-styled demo web page (self-contained HTML/CSS/JS; POSTs to `/recommend` and renders top-k routes). Static assets served at `GET /static/*` from `apps/api/static/`. `code_client` is only ever sent in the POST body — never in the URL or browser storage (Loi 09-08).
- `GET /health` → `{"status": "ok"}`
- `POST /recommend?variant=a|b` → `{"mode": "model"|"cold_start_cf"|"popularity"|"cold_start", "recommendations": [...], "labels": {"LiaisonId": "GARE DEPART → GARE ARRIVEE", ...}, "variant": "a"|"b", "request_id": "<uuid>"}`
- `POST /feedback` → `{"status": "ok"}` — log click event for CTR measurement

**Request body (`/recommend`):**
```json
{"code_client": "12345", "k": 3, "include_schedule": false}
```
`k` is clamped to [1, 3] by Pydantic validation.
`variant` query param (default `"a"`): `"b"` routes to challenger model; unknown values fall back to `"a"`.
`include_schedule` (bool, default false): when true, each recommended LiaisonId is enriched with next departures scraped from oncf.ma (cached 1hr per (origin, dest, date)). Stations without a known ONCF code return `[]` silently. Adds a `schedules` dict to the response.

**Request body (`/feedback`):**
```json
{"request_id": "<uuid>", "liaison_id": "R1", "clicked": true}
```
`request_id` must be a valid UUID4 matching the one returned by `/recommend`. Correlates serve + click events for CTR uplift measurement (no `code_client` ever logged — Loi 09-08).

**Startup:** Uses FastAPI lifespan — loads `recommender_a` (prod model) and `recommender_b` (challenger; falls back to A if challenger files absent) into `app.state`.

**Start command:**
```powershell
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

**Architecture:** `recommender.py` holds all business logic. `main.py` is a thin HTTP layer that delegates entirely to `Recommender.recommend()`.

---

## `Recommender` class (`src/rec_oncf/recommender.py`)

```python
Recommender.from_paths(paths)       # loads model + builds lookups from disk
Recommender.from_data(arts, clean_df, features_df)  # for testing
recommender.recommend(code_client, k=1)  # returns dict
```

**`recommend()` logic (TWO-STAGE Candidate Generation + Ranking):**
1. Look up `code_client` in `history_lookup` (built from `oncf_clean.parquet`)
2. If history is `None` → `_fallback(k)` → `{"mode": "popularity", "recommendations": <top-k global>}` (or `{"mode": "cold_start", "recommendations": []}` if `popularity.joblib` absent)
3. If `len(history) < 3` → `cold_start_rec.recommend()` → return `{"mode": "cold_start_cf", ...}` or `_fallback(k)`
4. `generate_candidates()` from history → if empty → `_fallback(k)`
5. `compute_inference_row(history)` — features built ON THE FLY from live history
6. **ONNX fast path**: `predict_proba_onnx(onnx_session, preprocessor, feat_row)` → probabilities over 1,011 classes in ~24ms
7. **Filter scores to candidates**: `le.transform(valid_candidates)` → keep only candidate indices
8. Sort filtered scores descending, take top-`k` → return `{"mode": "model", "recommendations": [...]}`
9. Edge case: if no candidate is known to encoder, fall back to raw `candidates[:k]` order
10. Every result dict also carries `"labels": {liaison_id: "GARE DEPART → GARE ARRIVEE", ...}` for all recommended ids that have an entry in `liaison_label_lookup` (unknown ids silently omitted).

**In-memory lookups (built at startup):**
- `history_lookup: dict[str, DataFrame]` — keyed by `CodeClient`, sorted by date
- `onnx_session: InferenceSession | None` — loaded from `xgb_ranker.onnx`; `None` in tests (sklearn fallback)
- `popularity: list[str]` — top LiaisonIds by global booking frequency, loaded from `popularity.joblib` (empty list if artifact absent)
- `liaison_label_lookup: dict[str, str]` — maps each LiaisonId to `"GARE DEPART → GARE ARRIVEE"`, built from `oncf_clean.parquet` at startup

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| Data cleaning (script 01) | ✅ Done | `oncf_clean.parquet` — 491,680 rows |
| Feature engineering (script 02) | ✅ Done | `features.parquet` — 491,680 × 26 cols |
| Model training (script 03) | ✅ Done | Models saved, metrics exceed all thresholds |
| Baselines (script 04) | ✅ Done | `reports/baseline_metrics.json` |
| Test suite (110 tests) | ✅ All passing | Last run: 110/110 green |
| FastAPI app (`apps/api/main.py`) | ✅ Ready | Model exists, lifespan loads at startup |
| API endpoint test (live) | ✅ Done | `/health` + `/recommend` × 5 cases tested |
| Two-stage filter in `recommender` | ✅ Fixed | Top-k now restricted to candidates |
| On-the-fly feature recomputation | ✅ Done | `compute_inference_row` replaces stale `feature_lookup` |
| Model metadata sidecar | ✅ Done | `models/xgb_ranker.meta.json` |
| .gitignore + README + CI | ✅ Done | Project polish + GH Actions |
| Cold-start CF (`cold_start.py`) | ✅ Done | Co-occurrence lookup for users with 1-2 trips; `models/cold_start.joblib` |
| ONNX Runtime inference | ✅ Done | `models/xgb_ranker.onnx` (148 MB); predict p50 ~24.5ms (was ~104ms, 4.2x speedup) |
| API latency p50 (ONNX + FastPreprocessor) | ✅ ~13.74 ms | FastPreprocessor replaced ColumnTransformer.transform (11.31ms → ~0.01ms); p99 ~16.89ms |
| Retraining pipeline (script 07) | ✅ Done | `scripts/07_retrain.py --dry-run`; guardrail blocks if HR@1 drops >5pp |
| Structured logging (`apps/api/main.py`) | ✅ Done | JSON logs → `logs/api.log`; per-request `mode`, `latency_ms`, `k`, `n_recommendations`; `code_client` never logged |
| ONCF schedule scraping (`schedule.py`) | ✅ Done | 24 stations, Redis+memory cache, `include_schedule` flag in API |
| A/B testing framework (`apps/api/main.py`) | ✅ Done | `?variant=a|b`, `/feedback`, `request_id` correlation |
| Popularity fallback + demo UI | ✅ Done | `mode: "popularity"` replaces empty cold_start; `"labels"` key in every `/recommend` response; ONCF-styled demo page at `GET /` |

---

## Bugs Fixed (session 2026-05-03)

| Bug | File | Fix |
|---|---|---|
| CUDA OOM / silent crash | `training.py` | Switched to `device="cpu"`, `n_estimators=200`, `max_depth=6` |
| `CodeClient` used as model feature (privacy violation) | `training.py` | `train_xgb_multiclass` and `predict_proba` now explicitly drop `CodeClient` |
| Unseen labels crash during eval | `03_train_ranker.py` | Test set filtered to known classes before `le.transform()` |
| `OneHotEncoder` in test vs `OrdinalEncoder` in production | `test_recommender.py` | Replaced `OneHotEncoder` with `OrdinalEncoder` to match production pipeline |
| Unused `import numpy as np` | `03_train_ranker.py` | Removed |

## Bugs Fixed (Sprint 1, session 2026-05-04)

| Bug | File | Fix |
|---|---|---|
| Two-stage architecture not actually applied (candidates were generated but never used to filter top-k) | `recommender.py` | After `predict_proba`, restrict scores to indices of `candidates` via `le.transform`, then take top-k |
| No reproducibility (random init each run) | `training.py` | Added `random_state=42` to `XGBClassifier` |
| No baseline comparison (HR@1=0.74 unjustified) | `scripts/04_baselines.py` | New script computes most_frequent / prev_liaison / global_top on same temporal split |
| `cleaning.py` had zero unit tests despite being most complex module | `tests/test_cleaning.py` | 5 new tests (cancellation propagation, cold start, join, missing essentials, cyclic encoding) |
| API endpoints had zero HTTP tests (only logic-layer tests) | `tests/test_api.py` | 5 new tests with FastAPI TestClient |
| Doc said 26 features but actual count is 25 | `CLAUDE.md`, `rapport_pfa.tex` | Corrected to 25 |

## Sprint 2 — Improvements (session 2026-05-04 cont.)

| Change | File | Effect |
|---|---|---|
| Model artifact metadata sidecar | `training.py`, `03_train_ranker.py` | `models/xgb_ranker.meta.json` next to model: training date, dataset fingerprint, package versions, hyperparams, metrics |
| `.gitignore` + `README.md` | new | Project polish; data/, models/, reports/ excluded from VCS |
| GitHub Actions CI | `.github/workflows/tests.yml` | pytest + ruff on push / PR |
| **On-the-fly feature recomputation** | `features.py` (`compute_inference_row`), `recommender.py` | API no longer uses stale `features.parquet` snapshot; features for the next-trip prediction are recomputed live from current history. Critical for production where new bookings happen between retrains. |
| New feature `user_top_liaison_share` | `features.py` | Captures user "loyalty" to dominant route. Multiclass-compatible; computed on strictly past observations (no leakage). 26 columns total. |
| `feature_lookup` removed from `Recommender` | `recommender.py` | Dataclass is leaner; only `history_lookup` is kept. `from_data(arts, clean)` signature simplified. |

---

## Cleanup Done (session 2026-05-03)

Deleted the following files that were no longer needed:
- `debug_train.py`, `debug_train_cpu.py` — one-off debugging scripts
- `train_err.txt`, `train_log.txt` — empty temp files
- `scripts/03_train_safe.py` — temporary CPU workaround script (now canonical)
- `scripts/04_eval_saved_model.py` — one-time evaluation script

---

## What's Left to Do (Phase 3)

1. **Inference latency** ✅ target met. p50 = ~13.74ms, p99 = ~16.89ms (both < 100ms).
   FastPreprocessor replaced sklearn ColumnTransformer.transform on the hot path
   (saved 11.31ms/request, 1469x speedup on that step alone). Profiling: candidates 2.7ms,
   compute_inference_row 1.4ms, FastPreprocessor ~0.01ms, ONNX 3.2ms, scoring 0.002ms.
   No model retraining was needed. Next latency lever if ever needed: replace pandas ops
   in generate_candidates and compute_inference_row with pure numpy (~10ms additional savings).

2. **Production-grade retraining pipeline** ✅ done — `scripts/07_retrain.py` (guardrail KPI) + `tasks/oncf_daily_retrain.xml` (Task Scheduler, 02h00 quotidien) + `scripts/retrain_job.bat` (wrapper avec logs rotatifs). Enregistrer avec : `schtasks /Create /XML tasks\oncf_daily_retrain.xml /TN "ONCF\DailyRetrain" /F` (PowerShell admin).

3. **A/B testing framework** ✅ done — `?variant=a|b` query param routes to `recommender_a` (prod) or `recommender_b` (challenger). `request_id` UUID in response correlates serve + click events. `POST /feedback {request_id, liaison_id, clicked}` logs to `api.log`. Challenger files: `models/xgb_ranker_challenger.{json,onnx}` + `models/label_encoder_challenger.joblib` — if absent, variant B silently falls back to A.

4. **Structured logging** ✅ done — JSON logs in `logs/api.log`; per-request `mode`, `latency_ms`, `k`, `n_recommendations`; `code_client` never logged.

5. **Popularity fallback + demo UI** ✅ done — `popularity.py` + `scripts/08_build_popularity.py` → `models/popularity.joblib` (~1,067 liaisons, ~120 KB). `_COLD_START` constant removed; all no-recommendation branches now call `_fallback(k)` returning `mode: "popularity"`. Every `/recommend` response includes a `"labels"` dict. ONCF-styled demo page served at `GET /`; static assets at `apps/api/static/`; `code_client` never leaves the POST body (Loi 09-08). 110 tests passing.

6. **Embeddings of liaisons** (long term) — Word2Vec on trip sequences
   to capture semantic similarity between routes.

---

## How to Run Everything (Fresh Setup)

```powershell
# 1. Clean data (already done — skip if oncf_clean.parquet exists)
.venv\Scripts\python.exe scripts/01_make_dataset.py

# 2. Build features (already done — skip if features.parquet exists)
.venv\Scripts\python.exe scripts/02_build_features.py

# 3. Train model (~43 min on CPU — run in a real terminal for live output)
.venv\Scripts\python.exe scripts/03_train_ranker.py

# 4. Compute baselines (~10 s)
.venv\Scripts\python.exe scripts/04_baselines.py

# 5. Build cold-start CF lookup (~30 s)
.venv\Scripts\python.exe scripts/05_build_cold_start.py

# 6. Export ONNX model + benchmark (~2 min — loads 281MB model)
.venv\Scripts\python.exe scripts/06_export_onnx.py

# 7. Build global-popularity fallback list (~5 s)
.venv\Scripts\python.exe scripts/08_build_popularity.py

# 8. Run tests (~10 s, 110 tests)
.venv\Scripts\python.exe -m pytest tests/ -v

# 9. Retrain with guardrail (optional — ~43 min on CPU)
.venv\Scripts\python.exe scripts/07_retrain.py --dry-run   # evaluate only
.venv\Scripts\python.exe scripts/07_retrain.py              # evaluate + promote

# 10. Start API
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```
