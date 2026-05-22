# CLAUDE.md — ONCF Zero-Click Search Recommender

> **Keep this file up to date.** Update it every time a module is added/changed, a bug is fixed, a script is run, or the status of any work item changes. Past session diaries belong in git log, not here.

---

## Contexte Académique et Stage

| Champ | Valeur |
|---|---|
| **Étudiant** | Omar Chekroun |
| **Formation** | Master Informatique, Gouvernance et Transformation Digitale (IGOV) — 1ère année (M1) |
| **Établissement** | Université Mohammed V de Rabat — Faculté des Sciences de Rabat |
| **Type de travail** | Stage PFA (Projet de Fin d'Année) — Master IGOV M1 |
| **Durée** | 3 mois — du 16 mars 2026 au 16 juin 2026 |
| **Organisme** | ONCF — Direction des Systèmes d'Information et Digital (SI Voyageurs) |

> **Règle de rédaction :** Toute rédaction doit refléter ce contexte. Ce travail est un **stage PFA (Projet de Fin d'Année)** de master M1. Le libellé "Rapport de Projet de Fin d'Année" est correct et doit être conservé. La filière est **Master IGOV**, pas "Sciences de l'Ingénieur Informatique".

### Règles rapport `rapport_pfa_v2.tex`

- **Compilation** : Overleaf uniquement — ne pas lancer pdflatex en local
- **Période des données** : 2018–2020
- **Header** : gauche = `oncf.png` (hauteur 0.7cm), droite = "Système de Recommandation Proactif ONCF" — pas de nom d'auteur
- **Jamais mentionner** : "Pivot Post-Réunion ONCF", justification CPU/GPU, manière dont on va tester
- **Structure chapitres** : Contexte → Analyse & Conception → Technologies et Outils → Sprint 1 → Sprint 2 → Phase 3 → Conclusion
- **Section Tests** : expliquer ce qu'on teste et pourquoi (pas juste lister les modules)

---

## État actuel

**Branch:** `main` — **144/144 tests passent** — modifs non commitées (ingestion test1 : `cleaning.py`, `features.py`, scripts 01/02, `tests/test_retrain_data_contract.py`).

### Résumé des livraisons

| Chantier | Statut | Détail |
|---|---|---|
| **Pipeline ML** | ✅ | Nettoyage, features, XGBoost, ONNX, cold-start CF, popularity fallback |
| **API REST** | ✅ | FastAPI : `/recommend`, `/health`, `/schedule`, `/feedback`, A/B testing, logging JSON |
| **Déploiement** | ✅ | `deploy/` : Dockerfile multi-stage, docker-compose, .env.example, .dockerignore |
| **Réentraînement** | ✅ | `scripts/07_retrain.py --window-months`, guardrail KPI, Task Scheduler XML |
| **Challenger + promotion** | ✅ | Challenger promu en prod le 2026-05-16 (ancien prod archivé dans `models/archive/20260516T163128Z/`) |
| **Horaire enrichi** | ✅ | `parse_horaire_csv` supporte nouveau format header + H:MM:SS — couverture **96.53% (1030/1067)** (revérifié 2026-05-22) |
| **Module extract_days** | ✅ | `src/rec_oncf/extract_days.py` + 5 tests TDD |
| **Ingestion retrain (test1)** | ✅ | `test1.csv` corrigé + débloqué : alias colonnes dans `cleaning.py`, scripts 01/02 paramétrés `--input/--output`, dtypes features figés. Features test1 **identiques** à oncf_data (4 tests contrat) |
| **Code simulation retrain** | ✅ écrit / 🔴 trainings à lancer | `simulation.py`, `12_simulate_daily_retrain.py`, `01 --extra-csv`, hyperparams overridables. Univers `oncf_full_*.parquet` construit. Reste : lancer les 2 entraînements |
| **Rapport + 23 figures** | ✅ | `rapport_pfa_v2.tex` à jour, layouts ONCF |
| **Lint (ruff)** | ✅ | `ruff check scripts/ src/ tests/` → 0 erreur (**161 tests**) |

### Données retrain — contrat « même pipeline que oncf_data »

Toute donnée de retrain (`test1.csv`, futurs fichiers) doit produire des features **structurellement identiques** à oncf_data en passant par le **même pipeline** (01 → 02). Garanti par :

- **Alias de colonnes** (`cleaning.py` : `COLUMN_ALIASES`, `_normalize_column_names`) — ex. `Acheteurid` (test1) → `AchteurId`. Sans ça, colonne acheteur silencieusement NA → `is_self_purchase` faux (tout à 0).
- **Dates day-first robustes** (`cleaning.py:_to_datetime`) — bascule `dayfirst=True` si >20% échouent. oncf_data = M/D/Y, test1 = D/M/Y ; les deux donnent le bon datetime. (warning du 1er probe silencé)
- **Dtypes figés** (`features.py:_OUTPUT_DTYPES`) — la présence/absence de NaN faisait flipper int↔float, le parse faisait flipper ns↔us. Map canonique appliquée en fin de `build_training_rows` (no-op pour oncf, normalise test1).
- **Scripts 01/02 paramétrés** `--input/--output` (défaut = oncf_data, donc l'existant ne change pas) ; noms de rapports dérivés du stem pour ne pas écraser les artefacts oncf.

**Vérification 2026-05-22** : `01 --input test1.csv` → `test1_clean.parquet` (805 093 lignes, 1238 liaisons) ; `02` → `test1_features.parquet` (62 423 users). Parité vs `features.parquet` : **mêmes 26 colonnes + ordre + dtypes (0 mismatch)**. Heure test1 réaliste (pics navette 7h 14% / 17h 12% / 8h 12% / 18h 10%) → **plus de reconstruction d'heure, plus de limitation scientifique à mentionner**.

### Simulation retrain quotidien — but, architecture, état (2026-05-22)

**But de test1** (fourni par l'ONCF) : vérifier que le **mécanisme de retrain** fonctionne et **observer le comportement des métriques** jour après jour. Validation du **retrain automatique** (cron Task Scheduler) **reportée**. Pas d'augmentation de données, pas de promotion prod.

**Architecture (nettoyer une fois, fenêtrer ensuite)** :
- **Univers** : `01 --input oncf_data.csv --extra-csv test1.csv --output oncf_full_clean.parquet` puis `02 → oncf_full_features.parquet`. Nettoyage **global** (cold-start/annulations/chaînage par client corrects, 11 % de clients à cheval). **1 326 559 lignes, 129 459 users, 1 379 liaisons.** Schéma **identique** à `features.parquet`.
- **Bug corrigé** : oncf=M/D/Y, test1=D/M/Y ⇒ concaténer en brut puis parser détruisait les dates oncf (−320k lignes). Fix = `load_and_concat._prepare` parse les dates **par fichier** (sa propre convention) avant concat.
- **Phase A baseline** : `12_simulate_daily_retrain.py --baseline` → train 2018→2021 **moins 7 jours** (`baseline_frame`, ~1.32 M lignes). **Pas de fenêtre.**
- **Phase B quotidien** : `--day N` (N∈[1,7]) → fenêtre glissante **365j** finissant au jour N, éval **honnête sur J+1** (historique tronqué ≤ D, `Recommender.from_data`), guardrail informatif, log `reports/simulation_daily.json`, modèles isolés `models/sim/`.
- **7 jours sim** = 7 derniers jours **denses** (`last_n_dates(min_count=200)`, car queue 2022 creuse 1 résa/j) → **2021-12-21, 23, 24, 25, 26, 27, 31** (J+1 = 204–292 résa). Hyperparams = challenger (depth 8, 250 arbres).

**⚠️ Coût** : chaque fenêtre 365j ≈ **816k lignes** (tout 2021) → entraînement **lourd** (~1.5–2 h/jour estimé), baseline ~2.5–3 h. À lancer en dernier.

### Prochaine action

1. **Lancer les 2 entraînements** (dernière étape) : `12_simulate_daily_retrain.py --baseline`, puis `--day 1..7`. Mesurer la durée réelle sur le jour 1.
2. Rapport : section "Intégration 2021 + simulation quotidienne" + figure stabilité HR@1 après chiffres réels.
3. Recompiler `rapport_pfa_v2.tex` sur Overleaf.

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
│   ├── cleaning.py         # make_clean_dataset() — raw CSV → oncf_clean.parquet ; COLUMN_ALIASES (Acheteurid→AchteurId), _to_datetime day-first robuste
│   ├── config.py           # default_paths() → Paths dataclass
│   ├── io.py               # read_csv / read_parquet / write_parquet / write_csv
│   ├── features.py         # build_training_rows() + compute_inference_row() — 26 cols ; _OUTPUT_DTYPES (dtypes figés)
│   ├── metrics.py          # hit_rate_at_k(), mrr_at_k()
│   ├── training.py         # temporal_split, train_xgb_multiclass, predict_proba, save/load_artifacts, top_k_labels
│   ├── candidates.py       # generate_candidates() — user history → candidate LiaisonIds
│   ├── recommender.py      # Recommender dataclass — from_paths / from_data / recommend()
│   ├── schedule.py         # ONCF live scraping (legacy fallback) — STATION_CODES, fetch_departures, get_schedule
│   ├── local_schedule.py   # offline schedule index — parse_horaire_csv, build_od_index, get_local_schedule(limit=3)
│   ├── extract_days.py     # extract_last_n_days() — split df en base + dict des n derniers jours
│   └── popularity.py       # build_popularity_list() + save/load_popularity
│
├── apps/api/
│   ├── main.py             # FastAPI — GET /, /health, POST /recommend, GET /schedule/{id}, POST /feedback
│   └── static/             # ONCF-styled single-page demo (index.html, styles.css, app.js)
│
├── scripts/
│   ├── 01_make_dataset.py       # raw CSVs → cleaned parquet (--input/--output ; défaut oncf_data → oncf_clean.parquet)
│   ├── 02_build_features.py     # cleaned parquet → features parquet (--input/--output ; défaut oncf_clean → features.parquet)
│   ├── 03_train_ranker.py       # features → models/ + reports/offline_metrics.json
│   ├── 04_baselines.py          # baselines → reports/baseline_metrics.json
│   ├── 05_build_cold_start.py   # oncf_clean → models/cold_start.joblib
│   ├── 06_export_onnx.py        # xgb_ranker.json → models/xgb_ranker.onnx + benchmark
│   ├── 07_retrain.py            # full retrain + KPI guardrail → promote models/
│   ├── 08_build_popularity.py   # oncf_clean → models/popularity.joblib
│   ├── 09_train_challenger.py   # entraîne challenger (max_depth=8, 250 arbres), exporte ONNX, compare vs prod
│   ├── 10_promote_challenger.py # archive prod, promeut challenger
│   ├── 11_build_schedule_index.py # horaire.csv → models/schedule_index.joblib
│   ├── 12a_extract_test1_days.py # test1.csv → base + 7 CSVs jour (caduc : pipeline standard 01/02 utilisé)
│   └── _doc_gen.py              # utility — prints dataset stats
│
├── tests/                   # 144 tests — pytest
│   └── test_*.py            # cleaning, features, metrics, training, candidates, recommender,
│                            #   api, schedule, local_schedule, cold_start, onnx, retrain, popularity, extract_days,
│                            #   retrain_data_contract (features test1 == oncf_data)
│
├── data/processed/
│   ├── oncf_clean.parquet  # 491,680 rows
│   └── features.parquet    # 491,680 × 26 cols
│
├── models/
│   ├── xgb_ranker.json      # ~448 MB — challenger promu (saved with joblib despite .json ext)
│   ├── label_encoder.joblib
│   ├── cold_start.joblib    # co-occurrence lookup
│   ├── xgb_ranker.onnx      # ~286 MB — ONNX export
│   ├── popularity.joblib    # ~120 KB — global popularity fallback
│   ├── schedule_index.joblib # ~200 KB — 2750 paires O/D
│   ├── xgb_ranker_challenger.{json,onnx} + label_encoder_challenger.joblib  # variant B (A/B testing)
│   └── archive/20260516T163128Z/ # prod précédent (rollback)
│
├── reports/
│   ├── cleaning_report.json, cleaning_provenance.parquet
│   └── offline_metrics.json, baseline_metrics.json
│
└── pyproject.toml          # pythonpath = ["src"] for pytest
```

---

## Data

| File | Rows | Description |
|---|---|---|
| `Desktop/oncf_data.csv` | raw | Raw ONCF bookings CSV (on user's Desktop) |
| `Desktop/Liaison.csv` | raw | Route lookup table (on user's Desktop) |
| `Desktop/horaire.csv` | raw | Train timetable — 2759 stops, 309 trains, 122 stations (UTF-8 BOM, comma-separated; nouveau format header + H:MM:SS) |
| `data/processed/oncf_clean.parquet` | 491,680 | Cleaned bookings with all original fields |
| `data/processed/features.parquet` | 491,680 | Model-ready feature table (26 cols) |

**Key stats:** 69,449 active users, 1,011 unique `LiaisonId` classes (after temporal split filtering).

---

## Feature Table Schema (`features.parquet` — 26 columns)

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

**Why CPU (not CUDA):** RTX 3050 has 4 GB VRAM which OOMs at 1,011 classes × depth 8 × 300 estimators. CPU training is stable and completes in ~43 minutes (base) / ~64 minutes (challenger hyperparams).

**XGBoost hyperparameters (prod actuel — challenger promu le 2026-05-16) :**
```python
xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    device="cpu",
    n_estimators=250,
    learning_rate=0.06,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_lambda=1.5,
    n_jobs=-1,
    random_state=42,
)
```

**Target metrics (thresholds from spec):**
- `hit_rate@1 > 0.50` (primary)
- `hit_rate@3 > 0.60`
- `mrr@3 > 0.60`

---

## Offline Metrics (prod actuel — challenger)

| Metric | Prod actuel | Threshold | Pass |
|---|---|---|---|
| `hit_rate@1` | **0.7691** | > 0.50 | ✅ |
| `hit_rate@3` | **0.9100** | > 0.60 | ✅ |
| `mrr@3` | **0.8333** | > 0.60 | ✅ |

- Train rows: 393,344 — Test rows: 98,261 (75 dropped — labels unseen in train)
- Classes: 1,011 — Training time: ~63.6 min sur CPU
- random_state: 42 — dataset_fingerprint: `4d0dfd12e0b60341`

## Baselines (`reports/baseline_metrics.json`)

| Model | HR@1 | HR@3 | MRR@3 |
|---|---|---|---|
| `global_top` (popularity only) | 0.0399 | 0.1125 | 0.0707 |
| `prev_liaison` (last seen) | 0.2620 | 0.3204 | 0.2881 |
| `most_frequent` (user freq + recency) | 0.2751 | 0.5128 | 0.3865 |
| **xgboost_multiclass (prod actuel)** | **0.7691** | **0.9100** | **0.8333** |

XGBoost is **2.80× better** than the best baseline (`most_frequent`) on HR@1.

---

## Artifact Paths (from `config.py`)

```python
models_dir             = <project_root>/models/
xgb_model_path         = models_dir / "xgb_ranker.json"                  # ~448 MB
label_encoder_path     = models_dir / "label_encoder.joblib"
cold_start_path        = models_dir / "cold_start.joblib"
onnx_model_path        = models_dir / "xgb_ranker.onnx"                  # ~286 MB
popularity_path        = models_dir / "popularity.joblib"                # ~120 KB
horaire_csv_path       = desktop / "horaire.csv"
schedule_index_path    = models_dir / "schedule_index.joblib"             # ~200 KB
# Challenger (A/B testing)
challenger_model       = models_dir / "xgb_ranker_challenger.json"
challenger_le          = models_dir / "label_encoder_challenger.joblib"
challenger_onnx        = models_dir / "xgb_ranker_challenger.onnx"
# Archive (rollback)
archive_dir            = models_dir / "archive" / "20260516T163128Z"
features_parquet       = data/processed/features.parquet
processed_dataset_parquet = data/processed/oncf_clean.parquet
```

> **Note:** `xgb_ranker.json` is saved using `joblib.dump` (not XGBoost native JSON), despite the `.json` extension. Do not change the filename or save/load method independently.

---

## API (`apps/api/main.py`)

**Endpoints:**
- `GET /` → ONCF-styled demo web page (self-contained HTML/CSS/JS; POSTs to `/recommend`). Static assets at `GET /static/*`. `code_client` is only ever sent in the POST body — never in the URL or browser storage (Loi 09-08).
- `GET /health` → `{"status": "ok"}`
- `POST /recommend?variant=a|b` → `{"mode": "model"|"cold_start_cf"|"popularity"|"cold_start", "recommendations": [...], "labels": {"LiaisonId": "GARE DEPART → GARE ARRIVEE", ...}, "variant": "a"|"b", "request_id": "<uuid>"}`
- `GET /schedule/{liaison_id}` → `{"liaison_id": "...", "schedule": [{"depart": "HH:MM", "arrive": "HH:MM", "train": "..."}]}` — serves from `schedule_index.joblib`; returns `[]` for LiaisonIds not covered (LGV / correspondance).
- `POST /feedback` → `{"status": "ok"}` — log click event for CTR measurement

**Request body (`/recommend`):**
```json
{"code_client": "12345", "k": 3, "include_schedule": false}
```
`k` is clamped to [1, 3] by Pydantic validation.
`variant` query param (default `"a"`): `"b"` routes to challenger model; unknown values fall back to `"a"`.
`include_schedule` (bool, default false): when true, enriches each recommended LiaisonId with next departures from the local schedule index. Adds a `schedules` dict to the response.

**Request body (`/feedback`):**
```json
{"request_id": "<uuid>", "liaison_id": "R1", "clicked": true}
```
`request_id` must be a valid UUID4 matching the one returned by `/recommend`. Correlates serve + click events for CTR uplift measurement (no `code_client` ever logged — Loi 09-08).

**Startup:** FastAPI lifespan — loads `recommender_a` (prod) and `recommender_b` (challenger; falls back to A if challenger files absent) + schedule index into `app.state`.

**Schedule source:** `local_schedule.get_local_schedule()` (offline, depuis `schedule_index.joblib`). Le scraper oncf.ma (`schedule.py`) reste dans le repo mais n'est plus appelé. `get_local_schedule(now=...)` lève `ValueError` si `now` est naïf (TZ-aware obligatoire pour éviter bugs CET/UTC silencieux).

**Start command:**
```powershell
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

**Latency:** p50 ~13.74 ms, p99 ~16.89 ms (ONNX + FastPreprocessor on hot path).

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
2. If history is `None` → `_fallback(k)` → `{"mode": "popularity", ...}` (or `{"mode": "cold_start", ...}` if `popularity.joblib` absent)
3. If `len(history) < 3` → `cold_start_rec.recommend()` → `{"mode": "cold_start_cf", ...}` or `_fallback(k)`
4. `generate_candidates()` from history → if empty → `_fallback(k)`
5. `compute_inference_row(history)` — features built ON THE FLY from live history (no stale parquet snapshot)
6. **ONNX fast path**: `predict_proba_onnx(...)` → probabilities over 1,011 classes
7. **Filter scores to candidates**: `le.transform(valid_candidates)` → keep only candidate indices
8. Sort filtered scores descending, take top-`k` → `{"mode": "model", ...}`
9. Edge case: if no candidate is known to encoder, fall back to raw `candidates[:k]` order
10. Every result dict also carries `"labels": {liaison_id: "GARE DEPART → GARE ARRIVEE", ...}` (unknown ids silently omitted).

**In-memory lookups (built at startup):**
- `history_lookup: dict[str, DataFrame]` — keyed by `CodeClient`, sorted by date
- `onnx_session: InferenceSession | None` — loaded from `xgb_ranker.onnx`; `None` in tests (sklearn fallback)
- `popularity: list[str]` — top LiaisonIds by global booking frequency
- `liaison_label_lookup: dict[str, str]` — `LiaisonId → "GARE DEPART → GARE ARRIVEE"`

---

## How to Run Everything (Fresh Setup)

```powershell
# 1. Clean data (skip if oncf_clean.parquet exists)
.venv\Scripts\python.exe scripts/01_make_dataset.py

# 2. Build features (skip if features.parquet exists)
.venv\Scripts\python.exe scripts/02_build_features.py

# 3. Train model (~43 min on CPU — run in a real terminal for live output)
.venv\Scripts\python.exe scripts/03_train_ranker.py

# 4. Compute baselines (~10 s)
.venv\Scripts\python.exe scripts/04_baselines.py

# 5. Build cold-start CF lookup (~30 s)
.venv\Scripts\python.exe scripts/05_build_cold_start.py

# 6. Export ONNX model + benchmark (~2 min)
.venv\Scripts\python.exe scripts/06_export_onnx.py

# 7. Build global-popularity fallback list (~5 s)
.venv\Scripts\python.exe scripts/08_build_popularity.py

# 8. Build local schedule index from horaire.csv (~5 s)
.venv\Scripts\python.exe scripts/11_build_schedule_index.py

# 9. Run tests (~10 s, 140 tests)
.venv\Scripts\python.exe -m pytest tests/ -v

# 10. Retrain with guardrail (optional — ~43 min on CPU)
.venv\Scripts\python.exe scripts/07_retrain.py --dry-run   # evaluate only
.venv\Scripts\python.exe scripts/07_retrain.py              # evaluate + promote

# 11. Start API
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
```

### Retrain quotidien (ONCF)

Wrapper : `scripts/retrain_job.bat` (logs rotatifs). Tâche planifiée 02h00 : `tasks/oncf_daily_retrain.xml`.
Enregistrer (PowerShell admin) : `schtasks /Create /XML tasks\oncf_daily_retrain.xml /TN "ONCF\DailyRetrain" /F`.
