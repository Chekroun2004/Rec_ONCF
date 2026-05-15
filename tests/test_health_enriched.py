from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_returns_enriched_status():
    """The /health endpoint must expose model/popularity/users counts for monitoring."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ok", "degraded"}
        assert "model_loaded" in data
        assert "popularity_loaded" in data
        assert "n_users_history" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["popularity_loaded"], bool)
        assert isinstance(data["n_users_history"], int)
