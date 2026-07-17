"""Deterministic availability boundaries for the standalone worker."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import (
    DEFAULT_WORKER_SATURATION_TIMEOUT_SECONDS,
    ENV_WORKER_SATURATION_TIMEOUT_SECONDS,
    HindsightConfig,
)
from hindsight_api.engine.memory_engine import UnsupportedWorkerTaskError
from hindsight_api.worker.main import _wait_for_shutdown_or_worker_failure, create_worker_app
from hindsight_api.worker.poller import (
    MAX_CONSECUTIVE_POLL_ERRORS,
    ActiveTaskInfo,
    ClaimedTask,
    WorkerBackgroundTaskError,
    WorkerPoller,
    WorkerPollingUnavailableError,
    WorkerSaturationTimeoutError,
    WorkerSchemaPollingError,
)
from hindsight_api.worker.stage import StageHolder


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_supervisor_propagates_unexpected_poller_failure():
    async def fail():
        raise RuntimeError("poller crashed")

    never = asyncio.Event()
    poller_task = asyncio.create_task(fail())
    http_task = asyncio.create_task(never.wait())
    try:
        with pytest.raises(RuntimeError, match="poller task failed") as raised:
            await _wait_for_shutdown_or_worker_failure(asyncio.Event(), poller_task, http_task)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert str(raised.value.__cause__) == "poller crashed"
    finally:
        await _cancel(http_task)


@pytest.mark.asyncio
async def test_supervisor_rejects_clean_http_exit():
    never = asyncio.Event()
    poller_task = asyncio.create_task(never.wait())
    http_task = asyncio.create_task(asyncio.sleep(0))
    try:
        with pytest.raises(RuntimeError, match="HTTP server task exited unexpectedly"):
            await _wait_for_shutdown_or_worker_failure(asyncio.Event(), poller_task, http_task)
    finally:
        await _cancel(poller_task)


@pytest.mark.asyncio
async def test_supervisor_accepts_only_explicit_shutdown_while_tasks_are_live():
    never = asyncio.Event()
    poller_task = asyncio.create_task(never.wait())
    http_task = asyncio.create_task(never.wait())
    shutdown = asyncio.Event()
    shutdown.set()
    try:
        await _wait_for_shutdown_or_worker_failure(shutdown, poller_task, http_task)
        assert not poller_task.done()
        assert not http_task.done()
    finally:
        await _cancel(poller_task)
        await _cancel(http_task)


def _health_endpoint(app):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/health")


@pytest.mark.asyncio
async def test_health_requires_live_ready_poller():
    poller = MagicMock()
    poller.worker_id = "worker-test"
    poller.is_shutdown = False
    poller.is_ready = False
    memory = MagicMock()
    memory._pool = None
    memory.health_check = AsyncMock(return_value={"status": "healthy", "database": "connected"})
    app = create_worker_app(poller, memory)
    endpoint = _health_endpoint(app)

    response = await endpoint()
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "unhealthy",
        "database": "connected",
        "worker_id": "worker-test",
        "is_shutdown": False,
        "poller_alive": False,
        "poller_ready": False,
        "reason": "poller_not_running",
    }

    never = asyncio.Event()
    poller_task = asyncio.create_task(never.wait())
    app.state.poller_task = poller_task
    try:
        response = await endpoint()
        assert response.status_code == 503
        assert json.loads(response.body)["reason"] == "poller_not_ready"

        poller.is_ready = True
        response = await endpoint()
        assert response.status_code == 200
        payload = json.loads(response.body)
        assert payload["poller_alive"] is True
        assert payload["poller_ready"] is True

        poller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await poller_task
        response = await endpoint()
        assert response.status_code == 503
        assert json.loads(response.body)["reason"] == "poller_not_running"
    finally:
        await _cancel(poller_task)


def _poller() -> WorkerPoller:
    tenant_extension = MagicMock()
    return WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=tenant_extension,
    )


def test_saturation_timeout_is_explicit_bounded_config(monkeypatch):
    monkeypatch.delenv(ENV_WORKER_SATURATION_TIMEOUT_SECONDS, raising=False)
    assert HindsightConfig.from_env().worker_saturation_timeout_seconds == DEFAULT_WORKER_SATURATION_TIMEOUT_SECONDS

    monkeypatch.setenv(ENV_WORKER_SATURATION_TIMEOUT_SECONDS, "1200")
    assert HindsightConfig.from_env().worker_saturation_timeout_seconds == 1200

    monkeypatch.setenv(ENV_WORKER_SATURATION_TIMEOUT_SECONDS, "299")
    with pytest.raises(ValueError, match="must be at least 300 seconds"):
        HindsightConfig.from_env()


async def _install_active_tasks(
    poller: WorkerPoller,
    *,
    started_at: float,
    stage_updates: list[float],
) -> list[asyncio.Task]:
    tasks = [asyncio.create_task(asyncio.Event().wait()) for _ in stage_updates]
    async with poller._in_flight_lock:
        poller._active_tasks = {
            f"operation-{index}": ActiveTaskInfo(
                op_type="retain",
                bank_id=f"bank-{index}",
                schema="tenant_a",
                bg_task=task,
                started_at=started_at,
                stage_holder=StageHolder(stage="llm.openrouter.retain", updated_at=updated_at),
                task_type="batch_retain",
            )
            for index, (task, updated_at) in enumerate(zip(tasks, stage_updates, strict=True))
        }
        poller._in_flight_count = len(tasks)
        poller._in_flight_by_type = {"retain": len(tasks)}
    return tasks


async def _cancel_all(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_full_saturation_without_progress_clears_readiness_and_fails():
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
        max_slots=2,
        saturation_timeout_seconds=300,
    )
    tasks = await _install_active_tasks(poller, started_at=100.0, stage_updates=[200.0, 250.0])
    poller._ready = True
    try:
        with (
            patch("hindsight_api.worker.poller.time.monotonic", return_value=550.0),
            pytest.raises(WorkerSaturationTimeoutError, match="occupied all 2 slots") as raised,
        ):
            await poller._raise_if_saturated_without_progress()
        assert "without task/stage progress for 300.0s" in str(raised.value)
        assert poller.is_ready is False
    finally:
        await _cancel_all(tasks)


@pytest.mark.asyncio
async def test_full_saturation_with_recent_stage_progress_remains_ready():
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
        max_slots=2,
        saturation_timeout_seconds=300,
    )
    tasks = await _install_active_tasks(poller, started_at=100.0, stage_updates=[200.0, 500.0])
    poller._ready = True
    try:
        with patch("hindsight_api.worker.poller.time.monotonic", return_value=550.0):
            await poller._raise_if_saturated_without_progress()
        assert poller.is_ready is True
    finally:
        await _cancel_all(tasks)


@pytest.mark.asyncio
async def test_partial_occupancy_does_not_trigger_saturation_recovery():
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
        max_slots=2,
        saturation_timeout_seconds=300,
    )
    tasks = await _install_active_tasks(poller, started_at=100.0, stage_updates=[200.0])
    poller._ready = True
    try:
        with patch("hindsight_api.worker.poller.time.monotonic", return_value=1000.0):
            await poller._raise_if_saturated_without_progress()
        assert poller.is_ready is True
    finally:
        await _cancel_all(tasks)


@pytest.mark.asyncio
async def test_saturation_timeout_is_supervised_fatal_without_poll_retries():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller._raise_if_saturated_without_progress = AsyncMock(
        side_effect=WorkerSaturationTimeoutError("all slots wedged")
    )
    poller.claim_batch = AsyncMock()

    poller_task = asyncio.create_task(poller.run())
    http_task = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(RuntimeError, match="poller task failed") as raised:
            await _wait_for_shutdown_or_worker_failure(asyncio.Event(), poller_task, http_task)
        assert isinstance(raised.value.__cause__, WorkerSaturationTimeoutError)
        assert str(raised.value.__cause__) == "all slots wedged"
    finally:
        await _cancel(http_task)

    poller.claim_batch.assert_not_awaited()
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_transient_recovery_failure_then_success_gates_readiness():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(side_effect=[RuntimeError("database unavailable"), 0])

    with pytest.raises(RuntimeError, match="database unavailable"):
        await poller.run()
    assert poller.is_ready is False

    readiness_seen = []

    async def claim_once():
        readiness_seen.append(poller.is_ready)
        return []

    async def stop_after_successful_poll():
        readiness_seen.append(poller.is_ready)
        poller._shutdown.set()

    poller.claim_batch = AsyncMock(side_effect=claim_once)
    poller._log_progress_if_due = AsyncMock()
    poller._wait_for_poll_interval_or_stop = AsyncMock(side_effect=stop_after_successful_poll)
    await poller.run()

    assert readiness_seen == [False, True]
    assert poller.recover_own_tasks.await_count == 2
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_persistent_claim_failure_degrades_health_then_fails_process():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller.claim_batch = AsyncMock(side_effect=RuntimeError("queue query denied"))

    memory = MagicMock()
    memory._pool = None
    memory.health_check = AsyncMock(return_value={"status": "healthy", "database": "connected"})
    app = create_worker_app(poller, memory)
    endpoint = _health_endpoint(app)

    first_backoff = asyncio.Event()
    release_backoff = asyncio.Event()

    async def controlled_backoff(_seconds):
        first_backoff.set()
        await release_backoff.wait()

    with patch("hindsight_api.worker.poller.asyncio.sleep", side_effect=controlled_backoff):
        poller_task = asyncio.create_task(poller.run())
        app.state.poller_task = poller_task
        await asyncio.wait_for(first_backoff.wait(), timeout=1)

        response = await endpoint()
        assert response.status_code == 503
        payload = json.loads(response.body)
        assert payload["poller_alive"] is True
        assert payload["poller_ready"] is False
        assert payload["reason"] == "poller_not_ready"

        release_backoff.set()
        with pytest.raises(WorkerPollingUnavailableError, match="polling remained unavailable") as raised:
            await poller_task

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "queue query denied"
    assert poller.claim_batch.await_count == MAX_CONSECUTIVE_POLL_ERRORS
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_any_configured_schema_scan_failure_fails_the_poll_cycle():
    poller = _poller()
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=[True, RuntimeError("tenant_b scan denied")])

    with pytest.raises(WorkerSchemaPollingError, match='scan failed for configured schema "tenant_b"') as raised:
        await poller._scan_active_schemas_by_exists(conn, ["tenant_a", "tenant_b"])

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "tenant_b scan denied"
    assert conn.fetchval.await_count == 2


@pytest.mark.asyncio
async def test_any_configured_schema_claim_failure_fails_the_poll_cycle():
    poller = _poller()
    poller._claim_batch_for_schema_inner = AsyncMock(side_effect=RuntimeError("tenant_a claim denied"))

    with pytest.raises(WorkerSchemaPollingError, match='claim failed for configured schema "tenant_a"') as raised:
        await poller._claim_batch_for_schema("tenant_a", {"retain": 1}, 1)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "tenant_a claim denied"


@pytest.mark.asyncio
async def test_successful_claim_cycle_restores_readiness_after_transient_failure():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller.claim_batch = AsyncMock(side_effect=[RuntimeError("database failover"), []])
    poller._log_progress_if_due = AsyncMock()
    readiness_seen = []

    async def stop_after_successful_poll():
        readiness_seen.append(poller.is_ready)
        poller._shutdown.set()

    poller._wait_for_poll_interval_or_stop = AsyncMock(side_effect=stop_after_successful_poll)

    with patch("hindsight_api.worker.poller.asyncio.sleep", new=AsyncMock()):
        await poller.run()

    assert poller.claim_batch.await_count == 2
    assert readiness_seen == [True]
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_per_schema_and_batch_recovery_failures_propagate():
    poller = _poller()
    poller._get_schemas = AsyncMock(return_value=["tenant_a"])
    poller._recover_schema_tasks = AsyncMock(side_effect=RuntimeError("batch reset failed"))

    with pytest.raises(RuntimeError, match='startup recovery for schema "tenant_a"') as raised:
        await poller.recover_own_tasks()
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "batch reset failed"


@pytest.mark.asyncio
async def test_terminal_queue_update_failure_fails_ready_poller():
    async def poison_executor(_task):
        raise RuntimeError("executor rejected payload")

    poller = _poller()
    poller._executor = poison_executor
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller._mark_failed = AsyncMock(side_effect=RuntimeError("queue update unavailable"))
    task = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000002",
        task_dict={"type": "poison", "operation_type": "poison", "bank_id": "bank-test"},
        schema="tenant_a",
    )
    claims = [[task], []]

    async def claim_batch():
        return claims.pop(0) if claims else []

    poller.claim_batch = AsyncMock(side_effect=claim_batch)
    poller._log_progress_if_due = AsyncMock()

    with (
        patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()),
        pytest.raises(WorkerBackgroundTaskError, match="authoritative queue state") as raised,
    ):
        await poller.run()

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "queue update unavailable"
    assert poller.is_ready is False
    poller._mark_failed.assert_awaited_once_with(task.operation_id, "executor rejected payload", "tenant_a")


@pytest.mark.asyncio
async def test_unknown_task_escapes_for_poller_terminal_update_without_delete():
    # Build a minimal engine object: no operation_id avoids the preflight DB
    # query, while the dispatch still proves unknown payloads fail outward.
    from hindsight_api.engine.memory_engine import MemoryEngine

    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()

    with pytest.raises(UnsupportedWorkerTaskError, match="Unknown task type: future_task"):
        await memory.execute_task({"type": "future_task", "bank_id": "bank-test"})

    # The obsolete cleanup helper was the only DELETE path for unknown tasks.
    assert not hasattr(MemoryEngine, "_delete_operation_record")

    async def reject_unknown(_task):
        raise UnsupportedWorkerTaskError("Unknown task type: future_task")

    poller = _poller()
    poller._executor = reject_unknown
    poller._mark_failed = AsyncMock()
    task = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000001",
        task_dict={"type": "future_task", "bank_id": "bank-test"},
        schema="tenant_a",
    )
    with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
        await poller._execute_task_inner(task)
    poller._mark_failed.assert_awaited_once_with(
        task.operation_id,
        "Unknown task type: future_task",
        "tenant_a",
    )
