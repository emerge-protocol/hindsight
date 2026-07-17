"""Add a per-claim generation token to async worker operations.

``worker_id`` identifies a process role but can be reused after a restart.  A
fresh ``claim_token`` on every pending -> processing transition prevents an old
same-worker execution from completing, retrying, or requeueing a successor
claim (the ABA case).

Revision ID: d7e8f9a0b1c2
Revises: c6d7e8f9a0b1
Create Date: 2026-07-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d7e8f9a0b1c2"
down_revision: str | Sequence[str] | None = "c6d7e8f9a0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    op.execute(f"ALTER TABLE {_pg_schema_prefix()}async_operations ADD COLUMN IF NOT EXISTS claim_token TEXT NULL")


def _pg_downgrade() -> None:
    op.execute(f"ALTER TABLE {_pg_schema_prefix()}async_operations DROP COLUMN IF EXISTS claim_token")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE async_operations ADD claim_token VARCHAR2(64) NULL")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE async_operations DROP COLUMN claim_token")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
