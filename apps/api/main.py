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
