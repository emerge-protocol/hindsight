"""Add a durable per-memory consolidation exclusion flag.

Revision ID: c6d7e8f9a0b1
Revises: b57a7c9e0d13
Create Date: 2026-07-13

PostgreSQL only. The provider abstraction keeps Oracle on its historical
always-eligible behaviour until that backend gains equivalent schema support.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c6d7e8f9a0b1"
down_revision: str | Sequence[str] | None = "b57a7c9e0d13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Schema-qualifier for PostgreSQL tenant tables."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _should_install_public_routine() -> bool:
    """Install the shared maintenance routine once, from the public run."""
    target_schema = context.config.get_main_option("target_schema")
    return not target_schema or target_schema == "public"


def _install_banks_needing_consolidation(*, include_exclusion: bool) -> None:
    exclusion_clause = "AND m.exclude_from_consolidation = false" if include_exclusion else ""
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.banks_needing_consolidation()
        RETURNS TABLE(schema_name text, bank_id text)
        LANGUAGE plpgsql STABLE
        AS $fn$
        DECLARE
            sch text;
        BEGIN
            FOR sch IN
                SELECT n.nspname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'memory_units' AND c.relkind = 'r'
            LOOP
                BEGIN
                    RETURN QUERY EXECUTE format($q$
                        SELECT %1$L::text, m.bank_id
                        FROM %1$I.memory_units m
                        JOIN %1$I.banks b ON b.bank_id = m.bank_id
                        WHERE m.consolidated_at IS NULL
                          AND m.consolidation_failed_at IS NULL
                          {exclusion_clause}
                          AND m.fact_type IN ('experience', 'world')
                          AND COALESCE(b.config -> 'enable_auto_consolidation', 'true'::jsonb) <> 'false'::jsonb
                          AND NOT EXISTS (
                              SELECT 1 FROM %1$I.async_operations o
                              WHERE o.bank_id = m.bank_id
                                AND o.operation_type = 'consolidation'
                                AND o.status IN ('pending', 'processing')
                          )
                        GROUP BY m.bank_id
                    $q$, sch);
                EXCEPTION
                    WHEN undefined_table OR invalid_schema_name OR undefined_column THEN
                        CONTINUE;
                END;
            END LOOP;
        END;
        $fn$;
        """
    )


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(
            f"""
            ALTER TABLE IF EXISTS {schema}{table}
            ADD COLUMN IF NOT EXISTS exclude_from_consolidation BOOLEAN NOT NULL DEFAULT FALSE
            """
        )

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_unconsolidated")
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_unconsolidated
        ON {schema}memory_units (bank_id, created_at)
        WHERE consolidated_at IS NULL
          AND exclude_from_consolidation = false
          AND fact_type IN ('experience', 'world')
        """
    )
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_consolidation_failed")
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_consolidation_failed
        ON {schema}memory_units (bank_id, consolidation_failed_at)
        WHERE consolidation_failed_at IS NOT NULL
          AND exclude_from_consolidation = false
          AND fact_type IN ('experience', 'world')
        """
    )

    if _should_install_public_routine():
        _install_banks_needing_consolidation(include_exclusion=True)


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    if _should_install_public_routine():
        _install_banks_needing_consolidation(include_exclusion=False)

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_unconsolidated")
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_unconsolidated
        ON {schema}memory_units (bank_id, created_at)
        WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')
        """
    )
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_consolidation_failed")
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_memory_units_consolidation_failed
        ON {schema}memory_units (bank_id, consolidation_failed_at)
        WHERE consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')
        """
    )

    for table in ("invalidated_memory_units", "memory_units"):
        op.execute(f"ALTER TABLE IF EXISTS {schema}{table} DROP COLUMN IF EXISTS exclude_from_consolidation")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
