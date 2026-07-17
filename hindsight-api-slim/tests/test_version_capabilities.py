"""Database-specific capability reporting for the version endpoint."""

from types import SimpleNamespace

import httpx
import pytest

from hindsight_api.api import create_app
from hindsight_api.config import _get_raw_config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("database_backend", "expected"),
    [("postgresql", True), ("oracle", False)],
)
async def test_version_reports_retain_consolidation_exclusion_only_when_supported(
    monkeypatch, database_backend, expected
):
    config = _get_raw_config()
    monkeypatch.setattr(config, "database_backend", database_backend)
    app = create_app(SimpleNamespace(audit_logger=None), initialize_memory=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "0.8.4+morgan.2"
    assert body["features"]["retain_consolidation_exclusion"] is expected
