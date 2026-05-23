from __future__ import annotations

import dataclasses
import json
import sys
import time
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rec_oncf.config import default_paths
from rec_oncf.io import read_parquet
from rec_oncf.recommender import Recommender
from rec_oncf.local_schedule import get_local_schedule, load_schedule_index
from rec_oncf.schedule import build_liaison_station_map


def _read_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _load_sim_recommender(base_paths, sim_dir: Path) -> Recommender:
    sim_paths = dataclasses.replace(
        base_paths,
        xgb_model_path=sim_dir / "xgb_ranker.json",
        label_encoder_path=sim_dir / "label_encoder.joblib",
        onnx_model_path=sim_dir / "xgb_ranker.onnx",
    )
    return Recommender.from_paths(sim_paths, require_onnx=False)


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

    # ── Variant B — current prod (challenger promoted 2026-05-16) ──
    rec_b = Recommender.from_paths(paths)
    app.state.recommender_a = rec_b  # kept for /health backward compat
    logger.info("Variant B (current prod) loaded")

    # ── Variant A — archived original prod (depth=6, 200 arbres) ──
    archive_dir = paths.models_dir / "archive" / "20260516T163128Z"
    if (archive_dir / "xgb_ranker.json").exists():
        a_paths = dataclasses.replace(
            paths,
            xgb_model_path=archive_dir / "xgb_ranker.json",
            label_encoder_path=archive_dir / "label_encoder.joblib",
            onnx_model_path=archive_dir / "xgb_ranker.onnx",
        )
        rec_a = Recommender.from_paths(a_paths)
        logger.info("Variant A (archive) loaded")
    else:
        rec_a = rec_b
        logger.warning("Archive model not found — variant A falls back to B")

    # ── Variant C — corpus étendu (test1 2021-2022, no ONNX) ──
    sim_baseline_dir = paths.models_dir / "sim" / "baseline"
    if (sim_baseline_dir / "xgb_ranker.json").exists():
        rec_c = _load_sim_recommender(paths, sim_baseline_dir)
        logger.info("Variant C (corpus étendu) loaded")
    else:
        rec_c = rec_b
        logger.warning("Sim baseline model not found — variant C falls back to B")

    # ── Variant D — fenêtre glissante 365j (no ONNX) — défaut ──
    sim_day1_dir = paths.models_dir / "sim" / "day_1"
    if (sim_day1_dir / "xgb_ranker.json").exists():
        rec_d = _load_sim_recommender(paths, sim_day1_dir)
        logger.info("Variant D (fenêtre 365j) loaded — DEFAULT")
    else:
        rec_d = rec_b
        logger.warning("Sim day_1 model not found — variant D falls back to B")

    app.state.variants = {"a": rec_a, "b": rec_b, "c": rec_c, "d": rec_d}

    # ── Model metadata for /models endpoint ──
    def _metrics(meta: dict) -> dict:
        m = meta.get("metrics", {})
        return {
            "hit_rate@1": round(m.get("hit_rate@1", 0), 4),
            "hit_rate@3": round(m.get("hit_rate@3", 0), 4),
            "mrr@3": round(m.get("mrr@3", 0), 4),
        }

    app.state.model_meta = [
        {
            "variant": "a",
            "label": "Prod initial",
            "description": "oncf 2018–2020 · depth=6 · 200 arbres",
            "dataset": "oncf",
            "metrics": _metrics(_read_meta(archive_dir / "xgb_ranker.meta.json")),
            "trained_at": "2026-05-04",
            "is_default": False,
            "available": (archive_dir / "xgb_ranker.json").exists(),
        },
        {
            "variant": "b",
            "label": "Challenger (promu)",
            "description": "oncf 2018–2020 · depth=8 · 250 arbres",
            "dataset": "oncf",
            "metrics": _metrics(_read_meta(paths.models_dir / "xgb_ranker_challenger.meta.json")),
            "trained_at": "2026-05-16",
            "is_default": False,
            "available": True,
        },
        {
            "variant": "c",
            "label": "Corpus étendu",
            "description": "test1 2021–2022 · depth=8 · 250 arbres",
            "dataset": "test1",
            "metrics": _metrics(_read_meta(sim_baseline_dir / "xgb_ranker.meta.json")),
            "trained_at": "2026-05-22",
            "is_default": False,
            "available": (sim_baseline_dir / "xgb_ranker.json").exists(),
        },
        {
            "variant": "d",
            "label": "Fenêtre glissante 365j",
            "description": "oncf+test1 · 365 derniers jours",
            "dataset": "oncf+test1",
            "metrics": _metrics(_read_meta(sim_day1_dir / "xgb_ranker.meta.json")),
            "trained_at": "2026-05-22",
            "is_default": True,
            "available": (sim_day1_dir / "xgb_ranker.json").exists(),
        },
    ]

    clean = read_parquet(paths.processed_dataset_parquet)
    app.state.liaison_map = build_liaison_station_map(clean)
    app.state.schedule_index = load_schedule_index(paths.schedule_index_path)
    if app.state.schedule_index:
        logger.info(f"Schedule index loaded: {len(app.state.schedule_index)} O/D pairs")
    else:
        logger.warning("schedule_index.joblib not found — run scripts/11_build_schedule_index.py")
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
# NOTE: StaticFiles validates the directory at import time. Keep apps/api/static/
# committed (do NOT add it to .gitignore) — the guard below only avoids an
# unhelpful import-time crash if it is ever missing.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


class RecommendRequest(BaseModel):
    code_client: str
    k: int = Field(default=1, ge=1, le=3)
    include_schedule: bool = False


class RecommendResponse(BaseModel):
    mode: str
    variant: str
    request_id: str
    recommendations: list[str]
    labels: dict[str, str] = {}
    schedules: dict[str, list[dict[str, str]]] | None = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    liaison_id: str
    clicked: bool


class FeedbackResponse(BaseModel):
    status: str


@app.get("/health")
def health():
    rec = getattr(app.state, "recommender_a", None)
    model_loaded = rec is not None and rec.onnx_session is not None
    popularity_loaded = rec is not None and bool(rec.popularity)
    n_users_history = len(rec.history_lookup) if rec is not None else 0
    status = "ok" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "popularity_loaded": popularity_loaded,
        "n_users_history": n_users_history,
    }


@app.get("/models")
def list_models():
    return app.state.model_meta


@app.post("/recommend", response_model=RecommendResponse, response_model_exclude_none=True)
def recommend(req: RecommendRequest, variant: str = "a"):
    t0 = time.perf_counter()
    served_variant = variant.lower() if variant.lower() in app.state.variants else "a"
    rec = app.state.variants[served_variant]
    result: dict = dict(rec.recommend(req.code_client, req.k))
    result["variant"] = served_variant
    result["request_id"] = str(_uuid.uuid4())

    if req.include_schedule and result["recommendations"]:
        now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
        def _fetch(lid: str) -> tuple[str, list]:
            return lid, get_local_schedule(
                lid, app.state.liaison_map, app.state.schedule_index, now
            )
        with ThreadPoolExecutor(max_workers=3) as pool:
            result["schedules"] = dict(pool.map(_fetch, result["recommendations"]))

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


class ScheduleResponse(BaseModel):
    liaison_id: str
    schedule: list[dict[str, str]]


@app.get("/schedule/{liaison_id}", response_model=ScheduleResponse)
def schedule_endpoint(liaison_id: str):
    now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
    deps = get_local_schedule(
        liaison_id, app.state.liaison_map, app.state.schedule_index, now
    )
    return {"liaison_id": liaison_id, "schedule": deps}


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    logger.bind(
        event="feedback",
        request_id=req.request_id,
        liaison_id=req.liaison_id,
        clicked=req.clicked,
    ).info("feedback")
    return {"status": "ok"}
