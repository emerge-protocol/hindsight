"""Unit regressions for fail-safe worker decommission SQL."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.admin.cli import _decommission_all_workers


@pytest.mark.asyncio
async def test_decommission_all_null_safely_releases_ownerless_processing_rows() -> None:
    connection = MagicMock()
    connection.fetch = AsyncMock(
        return_value=[
            {
                "operation_id": "00000000-0000-0000-0000-000000000001",
                "worker_id": None,
                "operation_type": "retain",
            }
        ]
    )
    connection.close = AsyncMock()

    with (
        patch("hindsight_api.admin.cli.resolve_database_url", new=AsyncMock(return_value="postgresql://db/test")),
        patch("hindsight_api.admin.cli.asyncpg.connect", new=AsyncMock(return_value=connection)),
    ):
        rows = await _decommission_all_workers("postgresql://db/test")

    sql = connection.fetch.await_args.args[0]
    assert "operations.worker_id IS NOT DISTINCT FROM claimed.worker_id" in sql
    assert "operations.claim_token IS NOT DISTINCT FROM claimed.claim_token" in sql
    assert rows[0]["worker_id"] is None
    connection.close.assert_awaited_once()
