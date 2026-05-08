# A/B Testing Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add variant A/B routing, request_id, and a /feedback endpoint to the ONCF recommender API so that a challenger model can be shadow-deployed and CTR-measured without touching any library code.

**Architecture:** Two `Recommender` instances (`recommender_a`, `recommender_b`) loaded at FastAPI startup; `recommender_b` falls back to `recommender_a` when challenger files are absent. The client passes `?variant=a|b`; the server routes accordingly and returns `variant` + `request_id` in the response. A new `POST /feedback` endpoint logs click events to `api.log`, joined to serve events via `request_id`.

**Tech Stack:** FastAPI 0.115, Pydantic v2, loguru, Python `uuid` stdlib, `dataclasses.replace` (stdlib)

---

## File Map

| File | What changes |
|---|---|
| `apps/api/main.py` | Lifespan loads two `Recommender` slots; `/recommend` adds `variant` query param + `request_id`; new `/feedback` endpoint; updated Pydantic models |
| `tests/test_api.py` | Fixture updated to set `recommender_a`/`recommender_b`; 5 new tests added |

No changes to `src/rec_oncf/*` or `config.py`.

---

## Task 1: Dual recommender slots + challenger loading

**Files:**
- Modify: `apps/api/main.py`
- Modify: `tests/test_api.py`

The existing code uses `app.state.recommender`. This task renames it to `app.state.recommender_a` and adds `app.state.recommender_b`, with challenger-file loading logic in lifespan.

- [ ] **Step 1: Update the test fixture to use the new slot names**

In `tests/test_api.py`, find the `client` fixture (line 99) and replace it:

```python
@pytest.fixture(scope="module")
def client():
    arts = _build_artifacts()
    clean = _build_clean_df()
    rec = Recommender.from_data(arts, clean)
    app.state.recommender_a = rec
    app.state.recommender_b = rec   # same rec — no challenger in tests
    app.state.liaison_map = {}
    app.state.redis = None
    return TestClient(app)
```

- [ ] **Step 2: Run existing tests to confirm they now fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: multiple failures — handler still reads `app.state.recommender` which is no longer set.

- [ ] **Step 3: Update lifespan and the /recommend handler in main.py**

Replace the entire content of `apps/api/main.py` with:

```python
from __future__ import annotations

import dataclasses
import sys
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.recommender import Recommender
from rec_oncf.schedule import build_liaison_station_map, get_schedule


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = default_paths()
    if not paths.xgb_model_path.exists():
        raise RuntimeError(
            f"Model not found: {paths.xgb_model_path}. Run scripts/03_train_ranker.py first."
        )
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.add(
        logs_dir / "api.log",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    app.state.recommender_a = Recommender.from_paths(paths)

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

    clean = read_parquet(paths.processed_dataset_parquet)
    app.state.liaison_map = build_liaison_station_map(clean)
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        app.state.redis = r
        logger.info("Redis schedule cache connected")
    except ImportError:
        app.state.redis = None
        logger.warning("Redis package not installed — schedule cache disabled")
    except Exception:
        app.state.redis = None
        logger.warning("Redis unavailable — using in-memory schedule cache")
    yield


app = FastAPI(title="ONCF Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    code_client: str
    k: int = Field(default=1, ge=1, le=3)
    include_schedule: bool = False


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse, response_model_exclude_none=True)
def recommend(req: RecommendRequest, variant: str = "a"):
    t0 = time.perf_counter()
    served_variant = "b" if variant.lower() == "b" else "a"
    rec = app.state.recommender_b if served_variant == "b" else app.state.recommender_a
    result: dict = dict(rec.recommend(req.code_client, req.k))
    result["variant"] = served_variant
    result["request_id"] = str(_uuid.uuid4())

    if req.include_schedule and result["recommendations"]:
        now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
        result["schedules"] = {
            lid: get_schedule(lid, app.state.liaison_map, now, redis_client=app.state.redis)
            for lid in result["recommendations"]
        }

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.bind(
        event="recommend",
        variant=served_variant,
        request_id=result["request_id"],
        mode=result["mode"],
        k=req.k,
        latency_ms=latency_ms,
        n_recommendations=len(result["recommendations"]),
    ).info("recommend")
    return result


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    logger.bind(
        event="feedback",
        request_id=req.request_id,
        liaison_id=req.liaison_id,
        clicked=req.clicked,
    ).info("feedback")
    return {"status": "ok"}
```

- [ ] **Step 4: Run existing tests to confirm they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: all 9 existing tests pass. (The new `variant` and `request_id` fields appear in responses — existing tests don't assert their absence, so no breakage.)

- [ ] **Step 5: Commit**

```powershell
git add apps/api/main.py tests/test_api.py
git commit -m "feat: dual recommender slots + challenger loading in lifespan"
```

---

## Task 2: Tests for variant routing and request_id

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add 4 new tests at the bottom of tests/test_api.py**

```python
def test_recommend_default_variant_is_a(client):
    resp = client.post("/recommend", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["variant"] == "a"


def test_recommend_variant_b_returns_b_label(client):
    # recommender_b == recommender_a in fixture (no challenger) — routing still works
    resp = client.post("/recommend?variant=b", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["variant"] == "b"
    assert body["mode"] == "model"


def test_recommend_unknown_variant_treated_as_a(client):
    resp = client.post("/recommend?variant=xyz", json={"code_client": "1001", "k": 1})
    assert resp.status_code == 200
    assert resp.json()["variant"] == "a"


def test_recommend_has_uuid_request_id(client):
    import uuid
    resp = client.post("/recommend", json={"code_client": "1001", "k": 1})
    body = resp.json()
    assert "request_id" in body
    # Raises ValueError if not a valid UUID
    uuid.UUID(body["request_id"])
```

- [ ] **Step 2: Run the new tests to confirm they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py::test_recommend_default_variant_is_a tests/test_api.py::test_recommend_variant_b_returns_b_label tests/test_api.py::test_recommend_unknown_variant_treated_as_a tests/test_api.py::test_recommend_has_uuid_request_id -v
```

Expected: all 4 PASS (implementation was already done in Task 1).

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_api.py
git commit -m "test: variant routing and request_id tests for A/B framework"
```

---

## Task 3: Tests for /feedback endpoint

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Add 2 new tests at the bottom of tests/test_api.py**

```python
def test_feedback_ok(client):
    resp = client.post("/feedback", json={
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "liaison_id": "L123",
        "clicked": True,
    })
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_feedback_missing_field_returns_422(client):
    # liaison_id and clicked are required
    resp = client.post("/feedback", json={"request_id": "abc-123"})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run the new tests to confirm they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py::test_feedback_ok tests/test_api.py::test_feedback_missing_field_returns_422 -v
```

Expected: both PASS (endpoint was implemented in Task 1).

- [ ] **Step 3: Run the full test suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all tests pass (count should be 99 — 93 existing + 6 new: 4 variant + 2 feedback).

- [ ] **Step 4: Commit**

```powershell
git add tests/test_api.py
git commit -m "test: /feedback endpoint tests for A/B framework"
```

---

## Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark A/B testing as done in the status table**

Find this row in the "Current Status" table:

```
| A/B testing framework (`apps/api/main.py`) | ❌ TODO | `/recommend?variant=A\|B` + `/feedback` |
```

If that row doesn't exist, find the "What's Left to Do" section item 3:

```
3. **A/B testing framework** — `/recommend?variant=A|B` to compare
   two models in production and measure CTR uplift.
```

Replace item 3 with:

```
3. **A/B testing framework** ✅ done — `?variant=a|b` query param routes to `recommender_a` (prod) or `recommender_b` (challenger). `request_id` UUID in response correlates serve + click events. `POST /feedback {request_id, liaison_id, clicked}` logs to `api.log`. Challenger files: `models/xgb_ranker_challenger.{json,onnx}` + `models/label_encoder_challenger.joblib` — if absent, variant B silently falls back to A.
```

Also update the test count in CLAUDE.md from 93 to 98 (two places: the test suite table and the "Run tests" comment).

- [ ] **Step 2: Commit**

```powershell
git add CLAUDE.md
git commit -m "docs: mark A/B testing framework as done in CLAUDE.md"
```

---

## Verification

After all tasks, run the full suite one final time:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected output: **99 passed** (93 original + 4 variant/request_id + 2 feedback).

To manually smoke-test the A/B endpoints with the live server:

```powershell
# Start server
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --reload

# In another terminal:
# Variant A (default)
curl -s -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"code_client\": \"12345\", \"k\": 1}" | python -m json.tool

# Variant B (fallback to A if no challenger)
curl -s -X POST "http://localhost:8000/recommend?variant=b" -H "Content-Type: application/json" -d "{\"code_client\": \"12345\", \"k\": 1}" | python -m json.tool

# Feedback
curl -s -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d "{\"request_id\": \"<paste request_id from above>\", \"liaison_id\": \"<paste first recommendation>\", \"clicked\": true}" | python -m json.tool
```

Expected: both `/recommend` responses contain `"variant"` and `"request_id"` fields; `/feedback` returns `{"status": "ok"}`.
