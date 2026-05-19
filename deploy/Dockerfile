# syntax=docker/dockerfile:1.6
# ============================================================
# Stage 1 — builder : install deps in a venv we can copy over
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install --prefix=/install \
        "fastapi==0.115.*" \
        "uvicorn[standard]==0.30.*" \
        "pydantic>=2.0,<3.0" \
        "pandas>=3.0,<4.0" \
        "pyarrow>=16.0,<20.0" \
        "xgboost==3.2.*" \
        "scikit-learn>=1.4,<2.0" \
        "onnxruntime>=1.18,<2.0" \
        "joblib>=1.4,<2.0" \
        "requests>=2.31,<3.0" \
        "beautifulsoup4>=4.12,<5.0" \
        "loguru>=0.7,<1.0" \
        "redis>=5.0,<6.0"

# ============================================================
# Stage 2 — runtime : minimal image, copy only what we need
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app/src:/install/lib/python3.12/site-packages"

RUN groupadd --gid 1000 oncf \
    && useradd --uid 1000 --gid oncf --shell /bin/bash --create-home oncf

WORKDIR /app

COPY --from=builder /install /install

COPY --chown=oncf:oncf src/ ./src/
COPY --chown=oncf:oncf apps/ ./apps/
COPY --chown=oncf:oncf scripts/ ./scripts/

RUN mkdir -p /app/data/processed /app/models /app/logs \
    && chown -R oncf:oncf /app

USER oncf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://localhost:8000/health', timeout=3); exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
