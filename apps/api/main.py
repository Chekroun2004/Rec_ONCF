from __future__ import annotations

import sys
import time
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
    app.state.recommender = Recommender.from_paths(paths)
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
    recommendations: list[str]
    schedules: dict[str, list[dict[str, str]]] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse, response_model_exclude_none=True)
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    result: dict = dict(app.state.recommender.recommend(req.code_client, req.k))

    if req.include_schedule and result["recommendations"]:
        now = datetime.now(tz=ZoneInfo("Africa/Casablanca"))
        result["schedules"] = {
            lid: get_schedule(lid, app.state.liaison_map, now, redis_client=app.state.redis)
            for lid in result["recommendations"]
        }

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.bind(
        mode=result["mode"],
        k=req.k,
        latency_ms=latency_ms,
        n_recommendations=len(result["recommendations"]),
    ).info("recommend")
    return result
