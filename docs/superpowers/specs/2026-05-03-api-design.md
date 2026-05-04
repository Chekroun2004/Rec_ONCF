# FastAPI Recommendation Endpoint — Design

**Date:** 2026-05-03
**Status:** Approved
**Scope:** `apps/api/main.py` + `src/rec_oncf/recommender.py`
**Deferred:** ONCF schedule API integration (last phase)

---

## 1. Objective

Expose the XGBoost recommendation model as an HTTP API so the ONCF mobile app can request top-k route recommendations for a user by `CodeClient`.

The app sends the user's `CodeClient`, the API looks up their history from the ONCF database (`oncf_clean.parquet`), runs the model, and returns the top-k `LiaisonId` recommendations.

---

## 2. Flow

```
ONCF App
  │  POST /recommend  { code_client, k }
  ▼
apps/api/main.py  (HTTP layer)
  │
  ▼
src/rec_oncf/recommender.py  (business logic)
  ├── cold-start check  (nb_trips < 3 → cold_start)
  ├── candidate generation  (candidates.py)
  ├── feature lookup  (last feature row for user)
  └── XGBoost predict_proba → top-k
  │
  ▼
ONCF App  ← { mode, recommendations }
```

---

## 3. Endpoints

### `POST /recommend`

**Request body:**
```json
{
  "code_client": "CLI123456",
  "k": 3
}
```
- `code_client` (str, required): user identifier — key to look up history
- `k` (int, optional, default=1, max=3): number of recommendations to return

**Response — warm user (≥ 3 trips):**
```json
{
  "mode": "model",
  "recommendations": ["345", "678", "901"]
}
```

**Response — cold-start user (< 3 trips):**
```json
{
  "mode": "cold_start",
  "recommendations": []
}
```

**Response — unknown user (not in history):**
```json
{
  "mode": "cold_start",
  "recommendations": []
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 4. Files

| File | Action | Responsibility |
|---|---|---|
| `src/rec_oncf/recommender.py` | Create | All business logic: load artifacts, build lookups, expose `recommend()` |
| `apps/__init__.py` | Create | Empty — makes `apps/` a package |
| `apps/api/__init__.py` | Create | Empty — makes `apps/api/` a package |
| `apps/api/main.py` | Create | FastAPI app — HTTP layer only, delegates to `recommender.py` |

---

## 5. Startup — `recommender.py`

Loaded once at process start via FastAPI `lifespan`:

```python
artifacts       = load_artifacts(model_path=..., label_encoder_path=...)
history_lookup  = {code_client: DataFrame of all trips}      # from oncf_clean.parquet
feature_lookup  = {code_client: single-row DataFrame}        # from features.parquet — last row per user
```

- `history_lookup`: used by `candidates.py` (trip count + candidate generation)
- `feature_lookup`: used by `predict_proba` (model input vector)

Memory footprint: ~150MB total (both parquets in RAM). Acceptable for a single-instance API.

---

## 6. Inference Logic — `recommend(code_client, k)`

```
1. history = history_lookup.get(code_client)
   → if None or len(history) < 3: return cold_start

2. candidates = generate_candidates(history_df, user_id=code_client, max_candidates=10)
   → if empty: return cold_start

3. feat_row = feature_lookup.get(code_client)
   → if None: return cold_start

4. proba = predict_proba(artifacts, feat_row_as_df, label_col="LiaisonId")
5. top_k = top_k_labels(proba, artifacts.label_encoder, k=k)
6. return { mode: "model", recommendations: top_k[0] }
```

---

## 7. Constraints

| Constraint | Value |
|---|---|
| Cold-start threshold | < 3 trips in history |
| Max k | 3 |
| No GPS | Never used |
| `CodeClient` | Treated as personal data (Loi 09-08 / CNDP) — never logged |
| ONCF schedule API | Deferred — not part of this spec |

---

## 8. Error Handling

| Case | Behaviour |
|---|---|
| Unknown `code_client` | Return `cold_start` (not an error) |
| `k` > 3 | Clamp to 3 |
| Model not loaded | 503 Service Unavailable |

---

## 9. Run

```bash
uvicorn apps.api.main:app --reload
```
