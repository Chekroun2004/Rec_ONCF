# A/B Testing Framework — Design Spec

**Date:** 2026-05-08
**Status:** Approved

---

## Goal

Add a lightweight A/B testing framework to the ONCF recommender API so that a retrained challenger model can be shadow-deployed alongside the production model and its recommendation quality measured via CTR uplift — without changing any library code (`rec_oncf.*`).

---

## Context

The retrain pipeline (`scripts/07_retrain.py`) produces a new model and promotes it if HR@1 doesn't drop by more than 5 pp. The A/B framework adds an online validation layer: before full promotion, the challenger can be served to a subset of traffic (those whose app passes `?variant=b`) and its real-world CTR compared against the production model.

---

## Architecture

Two `Recommender` instances are loaded at startup:

- **Variant A** — production model (`xgb_ranker.onnx`, `label_encoder.joblib`, `xgb_ranker.json`)
- **Variant B** — challenger model (`xgb_ranker_challenger.onnx`, `label_encoder_challenger.joblib`, `xgb_ranker_challenger.json`)

If challenger files are absent at startup, variant B silently falls back to variant A (no error returned to the client). The assignment is **client-driven**: the mobile app passes `?variant=a` or `?variant=b`. The server never reassigns.

CTR is measured via a correlation of two log events — serve and feedback — joined on a `request_id` UUID that the server generates per request. `code_client` is **never** logged (Loi 09-08 / CNDP compliance).

---

## Files Changed

| File | Change |
|---|---|
| `apps/api/main.py` | Lifespan loads two `Recommender` instances; `/recommend` routes by variant; `/feedback` endpoint added |
| `tests/test_api.py` | New tests: variant routing, fallback B→A, feedback endpoint |

No changes to `rec_oncf.*` library code or `config.py`. Challenger paths are derived at runtime from `paths.models_dir` with fixed filenames (see Challenger Model Loading).

---

## API Changes

### `POST /recommend`

**New query param:** `variant: str = "a"` (case-insensitive, default `"a"`; any value other than `"b"` routes to A)

**New response fields:**

```json
{
  "mode": "model",
  "variant": "a",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "recommendations": ["L123", "L456"],
  "schedules": { ... }
}
```

- `variant` — which variant was actually served (`"a"` or `"b"`)
- `request_id` — UUID4 generated server-side; used to join serve and feedback events in logs

### `POST /feedback`

New endpoint. Body:

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "liaison_id": "L123",
  "clicked": true
}
```

Returns `{"status": "ok"}`. Logs the event to `api.log` and returns immediately (no DB write).

---

## Logging

All events are written to the existing `api.log` (loguru JSON serialization).

**Serve event** (existing fields + new):
```json
{
  "event": "recommend",
  "variant": "b",
  "request_id": "uuid4",
  "mode": "model",
  "k": 3,
  "n_recommendations": 3,
  "latency_ms": 14.2
}
```

**Feedback event** (new):
```json
{
  "event": "feedback",
  "request_id": "uuid4",
  "liaison_id": "L123",
  "clicked": true
}
```

`code_client` is **never** present in either event.

**Offline CTR analysis:** join serve + feedback on `request_id`, group by `variant`, compute `sum(clicked) / count(*)`.

---

## Challenger Preparation

The challenger files are placed manually (or by a modified retrain script) at:
- `models/xgb_ranker_challenger.json`
- `models/label_encoder_challenger.joblib`
- `models/xgb_ranker_challenger.onnx`

For testing without a real second model, copy the production files:
```powershell
Copy-Item models\xgb_ranker.json      models\xgb_ranker_challenger.json
Copy-Item models\label_encoder.joblib models\label_encoder_challenger.joblib
Copy-Item models\xgb_ranker.onnx      models\xgb_ranker_challenger.onnx
```

All three files must be present; if any is missing, variant B falls back to A.

---

## Challenger Model Loading

In `lifespan`, after loading the production recommender:

```python
import dataclasses

challenger_onnx = paths.models_dir / "xgb_ranker_challenger.onnx"
challenger_model = paths.models_dir / "xgb_ranker_challenger.json"
challenger_le = paths.models_dir / "label_encoder_challenger.joblib"

if challenger_onnx.exists() and challenger_model.exists() and challenger_le.exists():
    c_paths = dataclasses.replace(
        paths,
        xgb_model_path=challenger_model,
        label_encoder_path=challenger_le,
        onnx_model_path=challenger_onnx,
    )
    app.state.recommender_b = Recommender.from_paths(c_paths)
    logger.info("Challenger model loaded — variant B active")
else:
    app.state.recommender_b = app.state.recommender_a
    logger.warning("Challenger model not found — variant B falls back to A")
```

A server restart is required to load a new challenger. This is acceptable: the retrain pipeline runs at 02:00 and the server can be restarted immediately after.

---

## Pydantic Models

```python
class RecommendResponse(BaseModel):
    mode: str
    variant: str
    request_id: str
    recommendations: list[str]
    schedules: dict[str, list[dict[str, str]]] | None = None

class FeedbackRequest(BaseModel):
    request_id: str
    liaison_id: str
    clicked: bool

class FeedbackResponse(BaseModel):
    status: str
```

---

## Error Handling

- Invalid `variant` value (not `"a"` or `"b"`) → treated as `"a"` (no error, no 400)
- `request_id` in `/feedback` not found in logs → accepted silently (server is stateless, no lookup)
- Challenger files partially present (e.g., onnx exists but label_encoder missing) → fall back to A, log warning

---

## Testing

| Test | What it verifies |
|---|---|
| `test_recommend_variant_a` | `?variant=a` routes to production recommender, response contains `variant="a"` and a `request_id` |
| `test_recommend_variant_b_fallback` | When no challenger exists, `?variant=b` still returns `variant="b"` with A's recommendations |
| `test_recommend_default_variant` | No `variant` param → defaults to `"a"` |
| `test_feedback_ok` | `POST /feedback` with valid body returns `{"status": "ok"}` |
| `test_feedback_validation` | Missing required fields → 422 |

No test loads a real challenger model (files don't exist in CI). Variant B fallback path is fully testable without extra model files.

---

## Out of Scope

- Server-side sticky assignment (hash on CodeClient) — client-driven is sufficient
- Admin hot-reload endpoint — restart is acceptable
- Statistical significance calculator — offline analysis in a notebook
- Persisting feedback to a database — logs are sufficient for the current scale
