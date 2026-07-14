"""Shape checks for the provider consolidation-exclusion migration."""

from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parents[1] / "hindsight_api/alembic/versions/c6d7e8f9a0b1_add_consolidation_exclusion.py"
)


def test_migration_covers_live_archive_and_both_consolidation_indexes():
    source = MIGRATION.read_text()

    assert 'for table in ("memory_units", "invalidated_memory_units")' in source
    assert "idx_memory_units_unconsolidated" in source
    assert "idx_memory_units_consolidation_failed" in source
    assert source.count("exclude_from_consolidation = false") >= 3
    assert "m.exclude_from_consolidation = false" in source


def test_migration_downgrade_restores_pre_feature_index_predicates():
    source = MIGRATION.read_text()
    downgrade = source.split("def _pg_downgrade()", maxsplit=1)[1]

    assert "WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')" in downgrade
    assert "WHERE consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')" in downgrade
    assert "DROP COLUMN IF EXISTS exclude_from_consolidation" in downgrade
