# ONCF Zero-Click Search Recommender

Proactive railway trip recommender for ONCF (Office National des Chemins de Fer
du Maroc). Predicts the most likely next O/D pair (`LiaisonId`) for each
client based on booking history and contextual signals, then exposes the
top-1 / top-3 recommendations to the ONCF Voyages mobile app via a REST API.

The project is the deliverable of a Projet de Fin d'Année at the Faculty of
Sciences of Rabat, Université Mohammed V.

> **Privacy:** complies with Loi 09-08 / CNDP (Morocco). `CodeClient` is the
> lookup key only — it is never used as a model feature and never appears in
> API logs.

---

## Quick start

```powershell
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\pip install -e .

# 2. Run the offline pipeline (raw CSVs on Desktop -> features -> trained model)
.venv\Scripts\python.exe scripts/01_make_dataset.py    # ~30 s
.venv\Scripts\python.exe scripts/02_build_features.py  # ~10 s
.venv\Scripts\python.exe scripts/03_train_ranker.py    # ~43 min CPU

# 3. Compute baselines for comparison
.venv\Scripts\python.exe scripts/04_baselines.py       # ~10 s

# 4. Run the test suite
.venv\Scripts\python.exe -m pytest tests/ -v           # 36 tests

# 5. Start the API
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload
# -> http://127.0.0.1:8000/docs   (Swagger UI)
```

---

## Repository layout

```
src/rec_oncf/      Library code (importable as rec_oncf.*)
  cleaning.py      Raw CSV -> oncf_clean.parquet (business rules)
  features.py      clean -> features.parquet (25 cols)
  candidates.py    Candidate Generation heuristic (history -> top-10)
  training.py      XGBoost multiclass training + artifacts
  recommender.py   Recommender class (two-stage Candidate + Ranking)
  metrics.py       HR@k, MRR@k
  config.py        Paths dataclass
  io.py            CSV / Parquet helpers

apps/api/main.py   FastAPI service (POST /recommend, GET /health)
scripts/           Numbered pipeline + baselines
tests/             36 unit + integration tests
configs/privacy.md CNDP compliance notes
docs/superpowers/  Specs and implementation plans
```

---

## Offline metrics (Sprint 2 model)

| Metric | XGBoost | most_frequent | prev_liaison | global_top |
|---|---|---|---|---|
| Hit Rate @1 | **0.7628** | 0.2751 | 0.2620 | 0.0399 |
| Hit Rate @3 | **0.9055** | 0.5128 | 0.3204 | 0.1125 |
| MRR @3      | **0.8277** | 0.3865 | 0.2881 | 0.0707 |

Train rows: 393,344 — Test rows: 98,261 — Classes: 1,011 — Temporal split 80/20.

XGBoost is **2.77x better** than the strongest baseline on HR@1.

### Metrics by user history depth

| Segment (trips in train) | n | HR@1 | HR@3 | MRR@3 |
|---|---|---|---|---|
| 0–2 | 44,397 | 0.7393 | 0.8930 | 0.8091 |
| 3–5 | 16,780 | 0.7517 | 0.9038 | 0.8210 |
| 6–20 | 23,737 | 0.7786 | 0.9159 | 0.8411 |
| 21+ | 13,347 | **0.8268** | **0.9311** | **0.8741** |

---

## Architecture

Two-stage Candidate Generation + Ranking, inspired by Uber Eats:

1. **Candidate Generation** — heuristic on user history (last 50 trips, sorted
   by frequency then recency, top 10).
2. **Ranking** — XGBoost multiclass (1,011 classes) scores all classes; scores
   are then **restricted to the candidate set** before taking top-`k`.

Cold-start rule: clients with fewer than 3 historical bookings get an explicit
`{"mode": "cold_start", "recommendations": []}` response.

---

## API

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Liveness probe -> `{"status": "ok"}` |
| POST | `/recommend` | Body: `{"code_client": "<id>", "k": <1..3>}` |

Response examples:

```json
{"mode": "model", "recommendations": ["4512", "3801", "1122"]}
{"mode": "cold_start", "recommendations": []}
```

Validation: `k` is constrained to `[1, 3]` by Pydantic; out-of-range
returns HTTP 422 with a structured error.

---

## Documentation

- Detailed PFA report: [`rapport_pfa.tex`](rapport_pfa.tex)
- Architecture decisions: [`docs/superpowers/specs/`](docs/superpowers/specs/)
- Implementation plans: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- Project guide for contributors / agents: [`CLAUDE.md`](CLAUDE.md)

---

## License

Internal academic project — UM5 Rabat / ONCF.
