"""Regression tests for safe claim-token rollout over legacy queue rows."""

from __future__ import annotations

import importlib
from unittest.mock import patch

MIGRATION = importlib.import_module("hindsight_api.alembic.versions.d7e8f9a0b1c2_add_async_operation_claim_token")


def test_postgres_upgrade_grandfathers_only_legacy_terminal_rows() -> None:
    with (
        patch.object(MIGRATION, "_pg_schema_prefix", return_value='"tenant".'),
        patch.object(MIGRATION.op, "execute") as execute,
    ):
        MIGRATION._pg_upgrade()

    statements = [call.args[0] for call in execute.call_args_list]
    assert statements[0] == ('ALTER TABLE "tenant".async_operations ADD COLUMN IF NOT EXISTS claim_token TEXT NULL')
    assert "status IN ('completed', 'failed', 'cancelled')" in statements[1]
    assert "claim_token IS NULL" in statements[1]
    assert "worker_id IS NOT NULL" in statements[1]
    assert "pending" not in statements[1]
    assert "processing" not in statements[1]
    assert MIGRATION._LEGACY_TERMINAL_CLAIM_TOKEN in statements[1]


def test_oracle_upgrade_grandfathers_only_legacy_terminal_rows() -> None:
    with patch.object(MIGRATION.op, "execute") as execute:
        MIGRATION._oracle_upgrade()

    statements = [call.args[0] for call in execute.call_args_list]
    assert statements[0] == "ALTER TABLE async_operations ADD claim_token VARCHAR2(64) NULL"
    assert "status IN ('completed', 'failed', 'cancelled')" in statements[1]
    assert "claim_token IS NULL" in statements[1]
    assert "worker_id IS NOT NULL" in statements[1]
    assert "pending" not in statements[1]
    assert "processing" not in statements[1]
    assert MIGRATION._LEGACY_TERMINAL_CLAIM_TOKEN in statements[1]
