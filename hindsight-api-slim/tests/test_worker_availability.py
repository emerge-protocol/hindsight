"""Deterministic availability boundaries for the standalone worker."""

import ast
import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.memory_engine import UnsupportedWorkerTaskError
from hindsight_api.worker import main as worker_main
from hindsight_api.worker.main import _wait_for_shutdown_or_worker_failure, create_worker_app
from hindsight_api.worker.poller import (
    MAX_CONSECUTIVE_POLL_ERRORS,
    ClaimedTask,
    WorkerBackgroundTaskError,
    WorkerPoller,
    WorkerPollingUnavailableError,
)


def test_worker_output_never_serializes_database_url():
    """Connection authority must not cross the worker's output boundary."""
    tree = ast.parse(inspect.getsource(worker_main))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_print = isinstance(target, ast.Name) and target.id == "print"
        is_logger = (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "logger"
        )
        if is_print or is_logger:
            assert "database_url" not in ast.unparse(node)


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
