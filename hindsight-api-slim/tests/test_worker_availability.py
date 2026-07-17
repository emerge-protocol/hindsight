"""Deterministic availability boundaries for the standalone worker."""

import ast
import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import (
    DEFAULT_WORKER_SATURATION_TIMEOUT_SECONDS,
    ENV_WORKER_SATURATION_TIMEOUT_SECONDS,
    HindsightConfig,
)
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.memory_engine import MemoryEngine, UnsupportedWorkerTaskError
from hindsight_api.worker import main as worker_main
from hindsight_api.worker.exceptions import (
    OperationPayloadIntegrityError,
    OperationQueueAuthorityError,
    OperationTerminalStateError,
    RetryTaskAt,
)
from hindsight_api.worker.main import _wait_for_shutdown_or_worker_failure, create_worker_app
from hindsight_api.worker.poller import (
    MAX_CONSECUTIVE_POLL_ERRORS,
    ActiveTaskInfo,
    ClaimedTask,
    SlotAvailability,
    WorkerBackgroundTaskError,
    WorkerPartialClaimReleaseError,
    WorkerPoller,
    WorkerPollingUnavailableError,
    WorkerSaturationTimeoutError,
)
from hindsight_api.worker.stage import StageHolder

CLAIM_TOKEN = "claim-token-test"


def test_worker_output_never_serializes_database_url():
    """Connection authority must not cross the worker's output boundary."""
    tree = ast.parse(inspect.getsource(worker_main))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_print = isinstance(target, ast.Name) and target.id == "print"
        is_logger = (
            isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "logger"
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


@pytest.mark.asyncio
async def test_supervisor_accepts_clean_http_exit_racing_explicit_shutdown():
    never = asyncio.Event()
    poller_task = asyncio.create_task(never.wait())
    http_task = asyncio.create_task(asyncio.sleep(0))
    shutdown = asyncio.Event()
    await http_task
    assert http_task.done()
    shutdown.set()
    try:
        await _wait_for_shutdown_or_worker_failure(shutdown, poller_task, http_task)
    finally:
        await _cancel(poller_task)


@pytest.mark.asyncio
async def test_supervisor_preserves_peer_failure_racing_explicit_shutdown():
    async def fail():
        raise RuntimeError("poller crashed during shutdown")

    poller_task = asyncio.create_task(fail())
    http_task = asyncio.create_task(asyncio.sleep(0))
    shutdown = asyncio.Event()
    with pytest.raises(RuntimeError, match="poller crashed during shutdown"):
        await poller_task
    assert poller_task.done()
    shutdown.set()
    with pytest.raises(RuntimeError, match="poller task failed") as raised:
        await _wait_for_shutdown_or_worker_failure(shutdown, poller_task, http_task)
    assert isinstance(raised.value.__cause__, RuntimeError)


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


class _DeniedAuthorityContext:
    async def __aenter__(self):
        # RuntimeError models a non-transient backend/ACL rejection.  The
        # acquire helper must not spend its connection-failover retry budget on
        # it, keeping this test focused on terminal-state propagation.
        raise RuntimeError("terminal queue update denied")

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _DeniedAuthorityBackend:
    """Permit the preflight read, then deny the authoritative terminal write."""

    _wraps_backend = True

    def __init__(self):
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        if self.acquire_count == 1:
            connection = AsyncMock()
            connection.fetchrow.return_value = {
                "status": "processing",
                "worker_id": "worker-test",
                "claim_token": CLAIM_TOKEN,
            }
            context = AsyncMock()
            context.__aenter__.return_value = connection
            context.__aexit__.return_value = False
            return context
        return _DeniedAuthorityContext()


class _AlwaysDeniedAuthorityBackend:
    """Deny even the preflight queue-authority read."""

    _wraps_backend = True

    def acquire(self):
        return _DeniedAuthorityContext()


class _StaticAuthorityBackend:
    """Return one fixed queue-authority record."""

    _wraps_backend = True

    def __init__(self, row):
        self.row = row

    def acquire(self):
        connection = AsyncMock()
        connection.fetchrow.return_value = self.row
        context = AsyncMock()
        context.__aenter__.return_value = connection
        context.__aexit__.return_value = False
        return context


class _ConnectionBackend:
    """Reuse one deterministic connection across engine authority checkpoints."""

    _wraps_backend = True

    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        context = AsyncMock()
        context.__aenter__.return_value = self.connection
        context.__aexit__.return_value = False
        return context


def _transactional_connection() -> MagicMock:
    connection = MagicMock()
    connection.fetchrow = AsyncMock()
    connection.fetch = AsyncMock()
    connection.execute = AsyncMock()
    transaction = AsyncMock()
    transaction.__aenter__.return_value = None
    transaction.__aexit__.return_value = False
    connection.transaction.return_value = transaction
    return connection


def _claimed_engine_task(operation_id: str, *, claim_token: str = CLAIM_TOKEN) -> dict:
    return {
        "type": "graph_maintenance",
        "operation_id": operation_id,
        "bank_id": "bank-test",
        "_worker_id": "worker-test",
        "_claim_token": claim_token,
        "_operation_id": operation_id,
    }


@pytest.mark.parametrize("ops_class", [PostgreSQLOps, OracleOps])
@pytest.mark.asyncio
async def test_provider_claim_persists_exact_generation_before_returning_rows(ops_class):
    operation_id = "00000000-0000-0000-0000-000000000073"
    row = {
        "operation_id": operation_id,
        "operation_type": "graph_maintenance",
        "task_payload": {"type": "graph_maintenance"},
        "retry_count": 0,
    }
    connection = _transactional_connection()
    connection.fetch.return_value = [row]
    connection.execute.return_value = "UPDATE 1"

    rows = await ops_class().claim_tasks(
        connection,
        "async_operations",
        "worker-test",
        CLAIM_TOKEN,
        {},
        1,
    )

    assert rows == [row]
    sql, worker_id, claim_token, operation_ids = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "claim_token = $2" in sql
    assert "status = 'pending'" in sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN
    assert operation_ids == [operation_id]


@pytest.mark.parametrize("ops_class", [PostgreSQLOps, OracleOps])
@pytest.mark.asyncio
async def test_provider_claim_update_count_mismatch_rolls_back_claim_transaction(ops_class):
    operation_id = "00000000-0000-0000-0000-000000000072"
    connection = _transactional_connection()
    connection.fetch.return_value = [
        {
            "operation_id": operation_id,
            "operation_type": "graph_maintenance",
            "task_payload": {"type": "graph_maintenance"},
            "retry_count": 0,
        }
    ]
    connection.execute.return_value = "UPDATE 0"

    with pytest.raises(RuntimeError, match="persisted 0 claim generations"):
        await ops_class().claim_tasks(
            connection,
            "async_operations",
            "worker-test",
            CLAIM_TOKEN,
            {},
            1,
        )


@pytest.mark.asyncio
async def test_queue_authority_read_failure_prevents_task_side_effects():
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=_AlwaysDeniedAuthorityBackend())
    memory._handle_graph_maintenance = AsyncMock(return_value=None)
    operation_id = "00000000-0000-0000-0000-000000000098"

    with pytest.raises(OperationQueueAuthorityError, match="Failed to prove runnable queue authority") as raised:
        await memory.execute_task(
            {
                "type": "graph_maintenance",
                "operation_id": operation_id,
                "bank_id": "bank-test",
                "_worker_id": "worker-test",
                "_claim_token": CLAIM_TOKEN,
                "_operation_id": operation_id,
            }
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    memory._handle_graph_maintenance.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_worker_aba_preflight_prevents_task_side_effects(caplog):
    operation_id = "00000000-0000-0000-0000-000000000082"
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(
        return_value=_StaticAuthorityBackend(
            {"status": "processing", "worker_id": "worker-test", "claim_token": "successor-token"}
        )
    )
    memory._handle_graph_maintenance = AsyncMock(return_value=None)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation moved"):
        await memory.execute_task(_claimed_engine_task(operation_id))

    memory._handle_graph_maintenance.assert_not_awaited()
    assert not any(record.levelname == "ERROR" for record in caplog.records)


@pytest.mark.asyncio
async def test_same_worker_aba_terminal_update_zero_fails_closed_after_handler():
    operation_id = "00000000-0000-0000-0000-000000000081"
    connection = _transactional_connection()
    connection.fetchrow.side_effect = [
        {"status": "processing", "worker_id": "worker-test", "claim_token": CLAIM_TOKEN},
        None,
    ]
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=_ConnectionBackend(connection))
    memory._handle_graph_maintenance = AsyncMock(return_value=None)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await memory.execute_task(_claimed_engine_task(operation_id))

    memory._handle_graph_maintenance.assert_awaited_once()
    terminal_sql, _operation_id, worker_id, claim_token = connection.fetchrow.await_args_list[1].args
    assert "status = 'processing'" in terminal_sql
    assert "worker_id = $2" in terminal_sql
    assert "claim_token = $3" in terminal_sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN


@pytest.mark.asyncio
async def test_same_worker_aba_checkpoint_fails_before_terminal_write():
    operation_id = "00000000-0000-0000-0000-000000000080"
    connection = _transactional_connection()
    connection.fetchrow.side_effect = [
        {"status": "processing", "worker_id": "worker-test", "claim_token": CLAIM_TOKEN},
        {"status": "processing", "worker_id": "worker-test", "claim_token": "successor-token"},
    ]
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=_ConnectionBackend(connection))

    async def checkpoint(_task):
        await memory._check_op_alive(operation_id)

    memory._handle_graph_maintenance = AsyncMock(side_effect=checkpoint)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation moved"):
        await memory.execute_task(_claimed_engine_task(operation_id))

    assert connection.fetchrow.await_count == 2


@pytest.mark.asyncio
async def test_checkpoint_queue_authority_read_failure_is_not_treated_as_alive():
    memory = object.__new__(MemoryEngine)
    memory._get_backend = AsyncMock(return_value=_AlwaysDeniedAuthorityBackend())

    with pytest.raises(OperationQueueAuthorityError, match="Failed to prove queue liveness") as raised:
        await memory._check_op_alive("00000000-0000-0000-0000-000000000091")

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_moved_queue_authority_prevents_task_side_effects():
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(
        return_value=_StaticAuthorityBackend(
            {"status": "processing", "worker_id": "worker-peer", "claim_token": "peer-token"}
        )
    )
    memory._handle_graph_maintenance = AsyncMock(return_value=None)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation moved"):
        await memory.execute_task(
            {
                "type": "graph_maintenance",
                "operation_id": "00000000-0000-0000-0000-000000000096",
                "bank_id": "bank-test",
                "_worker_id": "worker-test",
                "_claim_token": CLAIM_TOKEN,
                "_operation_id": "00000000-0000-0000-0000-000000000096",
            }
        )

    memory._handle_graph_maintenance.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_startup_recovery_is_scoped_to_exact_worker_owner():
    connection = MagicMock()
    connection.fetch = AsyncMock()
    connection.fetch.return_value = [
        {
            "operation_id": "00000000-0000-0000-0000-000000000097",
            "task_payload": {"type": "batch_retain"},
            "result_metadata": {"batch_id": "batch-1", "batch_provider": "openai"},
            "claim_token": CLAIM_TOKEN,
        }
    ]
    connection.execute = AsyncMock(return_value="UPDATE 1")
    transaction = AsyncMock()
    transaction.__aenter__.return_value = None
    transaction.__aexit__.return_value = False
    connection.transaction.return_value = transaction
    context = AsyncMock()
    context.__aenter__.return_value = connection
    context.__aexit__.return_value = False
    backend = MagicMock()
    backend.acquire.return_value = context
    poller = WorkerPoller(
        backend=backend,
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )

    assert await poller._recover_batch_operations("tenant_a") == 1

    select_sql, select_worker_id = connection.fetch.await_args.args
    assert "AND worker_id = $1" in select_sql
    assert select_worker_id == "worker-test"
    update_sql, _operation_id, update_worker_id, update_claim_token = connection.execute.await_args.args
    assert "AND worker_id = $2" in update_sql
    assert "claim_token = $3" in update_sql
    assert "AND status = 'processing'" in update_sql
    assert update_worker_id == "worker-test"
    assert update_claim_token == CLAIM_TOKEN


async def _claim_one_for_authority_test(task_payload: dict, operation_id: str) -> ClaimedTask:
    connection = MagicMock()
    transaction = AsyncMock()
    transaction.__aenter__.return_value = None
    transaction.__aexit__.return_value = False
    connection.transaction.return_value = transaction
    context = AsyncMock()
    context.__aenter__.return_value = connection
    context.__aexit__.return_value = False
    backend = MagicMock()
    backend.acquire.return_value = context
    backend.ops.claim_tasks = AsyncMock(
        return_value=[
            {
                "operation_id": operation_id,
                "operation_type": "graph_maintenance",
                "retry_count": 0,
                "task_payload": task_payload,
            }
        ]
    )
    poller = WorkerPoller(
        backend=backend,
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    claimed = await poller._claim_batch_for_schema_inner(None, {}, 1)
    assert len(claimed) == 1
    claim_call = backend.ops.claim_tasks.await_args.args
    assert claim_call[2] == "worker-test"
    assert isinstance(claim_call[3], str) and claim_call[3]
    assert claimed[0].claim_token == claim_call[3]
    assert claimed[0].task_dict["_claim_token"] == claim_call[3]
    return claimed[0]


def _authority_test_memory(claim_token: str = CLAIM_TOKEN) -> MemoryEngine:
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(
        return_value=_StaticAuthorityBackend(
            {"status": "processing", "worker_id": "worker-test", "claim_token": claim_token}
        )
    )
    memory._handle_graph_maintenance = AsyncMock(return_value=None)
    memory._mark_operation_completed = AsyncMock(return_value=None)
    return memory


@pytest.mark.asyncio
async def test_poller_binds_missing_payload_id_to_exact_claim_before_engine_side_effects():
    operation_id = "00000000-0000-0000-0000-000000000095"
    claimed = await _claim_one_for_authority_test(
        {"type": "graph_maintenance", "bank_id": "bank-test"},
        operation_id,
    )
    memory = _authority_test_memory(claimed.claim_token or "")
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=memory.execute_task,
        tenant_extension=MagicMock(),
    )

    with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
        await poller._execute_task_inner(claimed)

    assert claimed.task_dict["operation_id"] == operation_id
    memory._handle_graph_maintenance.assert_awaited_once()
    memory._mark_operation_completed.assert_awaited_once_with(operation_id)


@pytest.mark.asyncio
async def test_poller_marks_payload_id_mismatch_failed_without_restart_loop():
    operation_id = "00000000-0000-0000-0000-000000000094"
    claimed = await _claim_one_for_authority_test(
        {
            "type": "graph_maintenance",
            "operation_id": "00000000-0000-0000-0000-000000000093",
            "bank_id": "bank-test",
        },
        operation_id,
    )
    memory = _authority_test_memory(claimed.claim_token or "")
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=memory.execute_task,
        tenant_extension=MagicMock(),
    )
    poller._mark_failed = AsyncMock(return_value=None)

    with patch("hindsight_api.worker.poller.get_metrics_collector", return_value=MagicMock()):
        await poller._execute_task_inner(claimed)

    memory._handle_graph_maintenance.assert_not_awaited()
    memory._mark_operation_completed.assert_not_awaited()
    poller._mark_failed.assert_awaited_once_with(
        operation_id,
        "Worker task payload operation id does not match its database-authoritative claim",
        None,
        claimed.claim_token,
    )


@pytest.mark.asyncio
async def test_missing_claimed_id_is_deterministic_payload_integrity_failure():
    memory = _authority_test_memory()

    with pytest.raises(OperationPayloadIntegrityError, match="missing its database-authoritative"):
        await memory.execute_task(
            {
                "type": "graph_maintenance",
                "operation_id": "00000000-0000-0000-0000-000000000092",
                "bank_id": "bank-test",
                "_worker_id": "worker-test",
                "_claim_token": CLAIM_TOKEN,
            }
        )

    memory._handle_graph_maintenance.assert_not_awaited()


def _claimed_webhook_memory() -> MemoryEngine:
    memory = _authority_test_memory()
    memory._webhook_manager = None
    memory._http_client = MagicMock()
    memory._update_webhook_delivery_metadata = AsyncMock(return_value=None)
    return memory


def _claimed_webhook_task(operation_id: str) -> dict:
    return {
        "type": "webhook_delivery",
        "operation_id": operation_id,
        "bank_id": "bank-test",
        "url": "https://example.com/hook",
        "secret": None,
        "event_type": "retain.completed",
        "payload": {"status": "completed"},
        "_retry_count": 0,
        "_worker_id": "worker-test",
        "_claim_token": CLAIM_TOKEN,
        "_operation_id": operation_id,
    }


@pytest.mark.asyncio
async def test_claimed_webhook_success_writes_metadata_for_authoritative_operation():
    operation_id = "00000000-0000-0000-0000-000000000089"
    memory = _claimed_webhook_memory()
    response = MagicMock(status_code=204, text="accepted")
    response.raise_for_status.return_value = None
    memory._http_client.post = AsyncMock(return_value=response)

    await memory.execute_task(_claimed_webhook_task(operation_id))

    memory._update_webhook_delivery_metadata.assert_awaited_once_with(operation_id, 204, "accepted")
    memory._mark_operation_completed.assert_awaited_once_with(operation_id)


@pytest.mark.asyncio
async def test_claimed_webhook_failure_writes_metadata_for_authoritative_operation():
    operation_id = "00000000-0000-0000-0000-000000000088"
    memory = _claimed_webhook_memory()
    response = MagicMock(status_code=503, text="busy")
    response.raise_for_status.side_effect = RuntimeError("upstream unavailable")
    memory._http_client.post = AsyncMock(return_value=response)

    with pytest.raises(RetryTaskAt, match="upstream unavailable"):
        await memory.execute_task(_claimed_webhook_task(operation_id))

    memory._update_webhook_delivery_metadata.assert_awaited_once_with(operation_id, 503, "busy")
    memory._mark_operation_completed.assert_not_awaited()


@pytest.mark.asyncio
async def test_claimed_webhook_integrates_authoritative_id_with_fenced_metadata_and_terminal_writes():
    operation_id = "00000000-0000-0000-0000-000000000076"
    connection = _transactional_connection()
    connection.fetchrow.side_effect = [
        {"status": "processing", "worker_id": "worker-test", "claim_token": CLAIM_TOKEN},
        {"status": "processing", "worker_id": "worker-test", "claim_token": CLAIM_TOKEN},
        {"operation_id": operation_id},
    ]
    connection.execute.return_value = "UPDATE 1"
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=_ConnectionBackend(connection))
    memory._webhook_manager = None
    memory._http_client = MagicMock()
    memory._maybe_update_parent_operation = AsyncMock(return_value=None)
    response = MagicMock(status_code=204, text="accepted")
    response.raise_for_status.return_value = None
    memory._http_client.post = AsyncMock(return_value=response)

    await memory.execute_task(_claimed_webhook_task(operation_id))

    metadata_sql, _operation_id, _metadata, worker_id, claim_token = connection.execute.await_args.args
    assert "status = 'processing'" in metadata_sql
    assert "worker_id = $3" in metadata_sql
    assert "claim_token = $4" in metadata_sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN

    terminal_sql, _operation_id, worker_id, claim_token = connection.fetchrow.await_args_list[2].args
    assert "status = 'processing'" in terminal_sql
    assert "worker_id = $2" in terminal_sql
    assert "claim_token = $3" in terminal_sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN
    memory._maybe_update_parent_operation.assert_awaited_once()


@pytest.mark.asyncio
async def test_claimed_webhook_rechecks_claim_immediately_before_http_side_effect():
    operation_id = "00000000-0000-0000-0000-000000000077"
    connection = _transactional_connection()
    connection.fetchrow.side_effect = [
        {"status": "processing", "worker_id": "worker-test", "claim_token": CLAIM_TOKEN},
        {
            "status": "processing",
            "worker_id": "worker-test",
            "claim_token": "ffffffffffffffffffffffffffffffff",
        },
    ]
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=_ConnectionBackend(connection))
    memory._webhook_manager = None
    memory._http_client = MagicMock()
    memory._http_client.post = AsyncMock()
    memory._update_webhook_delivery_metadata = AsyncMock(return_value=None)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation moved"):
        await memory.execute_task(_claimed_webhook_task(operation_id))

    memory._http_client.post.assert_not_awaited()
    memory._update_webhook_delivery_metadata.assert_not_awaited()
    assert connection.fetchrow.await_count == 2


@pytest.mark.asyncio
async def test_claimed_webhook_payload_id_mismatch_is_rejected_before_http_side_effect():
    claimed_operation_id = "00000000-0000-0000-0000-000000000075"
    task = _claimed_webhook_task(claimed_operation_id)
    task["operation_id"] = "00000000-0000-0000-0000-000000000074"
    memory = _claimed_webhook_memory()
    memory._http_client.post = AsyncMock()

    with pytest.raises(OperationPayloadIntegrityError, match="does not match"):
        await memory.execute_task(task)

    memory._http_client.post.assert_not_awaited()
    memory._update_webhook_delivery_metadata.assert_not_awaited()


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
async def test_completed_task_waiting_for_cleanup_does_not_trigger_saturation_recovery():
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
        max_slots=1,
        saturation_timeout_seconds=300,
    )
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    async with poller._in_flight_lock:
        poller._active_tasks = {
            "operation-0": ActiveTaskInfo(
                op_type="retain",
                bank_id="bank-0",
                schema="tenant_a",
                bg_task=task,
                started_at=100.0,
                stage_holder=StageHolder(stage="done", updated_at=100.0),
                task_type="batch_retain",
            )
        }
        poller._in_flight_count = 1
        poller._in_flight_by_type = {"retain": 1}
    poller._ready = True

    with patch("hindsight_api.worker.poller.time.monotonic", return_value=1000.0):
        await poller._raise_if_saturated_without_progress()

    assert poller.is_ready is True


@pytest.mark.asyncio
async def test_empty_active_registry_does_not_crash_saturation_check():
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
        max_slots=0,
        saturation_timeout_seconds=300,
    )
    poller._ready = True

    with patch("hindsight_api.worker.poller.time.monotonic", return_value=1000.0):
        await poller._raise_if_saturated_without_progress()

    assert poller.is_ready is True


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
async def test_graceful_shutdown_propagates_terminal_queue_failure_after_drain():
    release = asyncio.Event()

    async def fail_terminally(_task):
        await release.wait()
        raise OperationTerminalStateError("completed queue write denied")

    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id="worker-test",
        executor=fail_terminally,
        tenant_extension=MagicMock(),
    )
    claimed = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000090",
        task_dict={"type": "graph_maintenance", "bank_id": "bank-test"},
        schema=None,
    )
    await poller.execute_task(claimed)
    shutdown_task = asyncio.create_task(poller.shutdown_graceful(timeout=2.0))
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(WorkerBackgroundTaskError, match="authoritative queue state") as raised:
        await shutdown_task

    assert isinstance(raised.value.__cause__, OperationTerminalStateError)
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
async def test_real_schema_claim_failure_degrades_health_then_fails_process():
    """The real claim path must not translate denied authority into an empty queue."""
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller._get_available_slots = AsyncMock(return_value=SlotAvailability(reserved={}, shared=1))
    poller._get_schemas = AsyncMock(return_value=["tenant_a"])
    poller._scan_active_schemas = AsyncMock(return_value={"tenant_a"})
    poller._claim_batch_for_schema_inner = AsyncMock(side_effect=PermissionError("queue trigger denied"))

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
        assert json.loads(response.body)["reason"] == "poller_not_ready"

        release_backoff.set()
        with pytest.raises(WorkerPollingUnavailableError, match="polling remained unavailable") as raised:
            await poller_task

    claim_error = raised.value.__cause__
    assert isinstance(claim_error, RuntimeError)
    assert 'failed to claim tasks for schema "tenant_a"' in str(claim_error)
    assert isinstance(claim_error.__cause__, PermissionError)
    assert poller._claim_batch_for_schema_inner.await_count == MAX_CONSECUTIVE_POLL_ERRORS
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_real_schema_scan_failure_propagates_through_claim_batch():
    """A failed EXISTS probe is an unavailable poll, never an idle schema."""
    poller = _poller()
    poller._get_available_slots = AsyncMock(return_value=SlotAvailability(reserved={}, shared=1))
    poller._get_schemas = AsyncMock(return_value=["tenant_a"])
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=PermissionError("schema select denied"))

    async def real_scan(schemas):
        return await poller._scan_active_schemas_by_exists(conn, schemas)

    poller._scan_active_schemas = real_scan  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='failed to scan schema "tenant_a"') as raised:
        await poller.claim_batch()

    assert isinstance(raised.value.__cause__, PermissionError)
    conn.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_cross_schema_claim_failure_releases_committed_claim_before_retry():
    poller = _poller()
    poller._get_available_slots = AsyncMock(return_value=SlotAvailability(reserved={}, shared=2))
    poller._get_schemas = AsyncMock(return_value=["tenant_a", "tenant_b"])
    poller._scan_active_schemas = AsyncMock(return_value={"tenant_a", "tenant_b"})
    operation_id = "00000000-0000-0000-0000-000000000087"
    claimed = ClaimedTask(
        operation_id=operation_id,
        task_dict={"type": "graph_maintenance", "operation_type": "graph_maintenance"},
        schema="tenant_a",
    )
    claim_state = "pending"
    tenant_b_unavailable = True

    async def claim_schema(schema, _reserved, _shared):
        nonlocal claim_state
        if schema == "tenant_a" and claim_state == "pending":
            claim_state = "processing"
            return [claimed]
        if schema == "tenant_b" and tenant_b_unavailable:
            raise PermissionError("tenant B queue unavailable")
        return []

    async def release_claims(tasks):
        nonlocal claim_state
        assert tasks == [claimed]
        assert claim_state == "processing"
        claim_state = "pending"

    poller._claim_batch_for_schema_inner = AsyncMock(side_effect=claim_schema)
    poller._release_claimed_tasks = AsyncMock(side_effect=release_claims)

    with pytest.raises(RuntimeError, match='failed to claim tasks for schema "tenant_b"'):
        await poller.claim_batch()

    assert claim_state == "pending"
    assert operation_id not in poller._active_tasks
    poller._release_claimed_tasks.assert_awaited_once_with([claimed])

    tenant_b_unavailable = False
    tasks = await poller.claim_batch()
    assert tasks == [claimed]
    assert claim_state == "processing"


@pytest.mark.asyncio
async def test_partial_claim_release_is_fenced_to_exact_worker_and_processing_row():
    connection = MagicMock()
    connection.execute = AsyncMock(return_value="UPDATE 1")
    transaction = AsyncMock()
    transaction.__aenter__.return_value = None
    transaction.__aexit__.return_value = False
    connection.transaction.return_value = transaction
    context = AsyncMock()
    context.__aenter__.return_value = connection
    context.__aexit__.return_value = False
    backend = MagicMock()
    backend.acquire.return_value = context
    poller = WorkerPoller(
        backend=backend,
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    task = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000086",
        task_dict={},
        schema="tenant_a",
        claim_token=CLAIM_TOKEN,
    )

    await poller._release_claimed_tasks([task])

    sql, operation_id, worker_id, claim_token = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "worker_id = $2" in sql
    assert "claim_token = $3" in sql
    assert operation_id == task.operation_id
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN


@pytest.mark.parametrize(
    ("method_name", "transition_args"),
    [
        ("_mark_completed", ()),
        ("_mark_failed", ("provider failed",)),
        ("_schedule_retry", ("later", "provider unavailable")),
        ("_defer_operation", ("later", "backpressure")),
    ],
)
@pytest.mark.asyncio
async def test_same_worker_aba_poller_transition_update_zero_fails_closed(method_name, transition_args):
    connection = _transactional_connection()
    connection.execute.return_value = "UPDATE 0"
    poller = WorkerPoller(
        backend=_ConnectionBackend(connection),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    operation_id = "00000000-0000-0000-0000-000000000079"

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await getattr(poller, method_name)(operation_id, *transition_args, None, CLAIM_TOKEN)

    sql, *args = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "worker_id" in sql and "claim_token" in sql
    assert args[-2:] == ["worker-test", CLAIM_TOKEN]


@pytest.mark.asyncio
async def test_same_worker_aba_partial_release_update_zero_fails_closed():
    connection = _transactional_connection()
    connection.execute.return_value = "UPDATE 0"
    poller = WorkerPoller(
        backend=_ConnectionBackend(connection),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    task = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000078",
        task_dict={},
        schema=None,
        claim_token=CLAIM_TOKEN,
    )

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await poller._release_claimed_tasks([task])

    sql, _operation_id, worker_id, claim_token = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "claim_token = $3" in sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN


@pytest.mark.asyncio
async def test_partial_release_malformed_command_status_fails_as_authority_loss():
    connection = _transactional_connection()
    connection.execute.return_value = "unexpected-status"
    poller = WorkerPoller(
        backend=_ConnectionBackend(connection),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    task = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000076",
        task_dict={},
        schema=None,
        claim_token=CLAIM_TOKEN,
    )

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await poller._release_claimed_tasks([task])


@pytest.mark.asyncio
async def test_same_worker_aba_recovery_update_zero_fails_closed():
    connection = _transactional_connection()
    connection.fetch.return_value = [
        {
            "operation_id": "00000000-0000-0000-0000-000000000077",
            "claim_token": CLAIM_TOKEN,
        }
    ]
    connection.execute.return_value = "UPDATE 0"
    poller = WorkerPoller(
        backend=_ConnectionBackend(connection),
        worker_id="worker-test",
        executor=AsyncMock(),
        tenant_extension=MagicMock(),
    )
    poller._recover_batch_operations = AsyncMock(return_value=0)

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await poller._recover_schema_tasks(None)

    sql, _operation_id, worker_id, claim_token = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "claim_token = $3" in sql
    assert worker_id == "worker-test"
    assert claim_token == CLAIM_TOKEN


@pytest.mark.asyncio
async def test_partial_claim_release_failure_exits_before_readiness_can_recover():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller._get_available_slots = AsyncMock(return_value=SlotAvailability(reserved={}, shared=2))
    poller._get_schemas = AsyncMock(return_value=["tenant_a", "tenant_b"])
    poller._scan_active_schemas = AsyncMock(return_value={"tenant_a", "tenant_b"})
    claimed = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000085",
        task_dict={"type": "graph_maintenance", "operation_type": "graph_maintenance"},
        schema="tenant_a",
    )
    tenant_b_calls = 0

    async def claim_schema(schema, _reserved, _shared):
        nonlocal tenant_b_calls
        if schema == "tenant_a":
            return [claimed]
        tenant_b_calls += 1
        if tenant_b_calls == 1:
            raise PermissionError("tenant B transient failure")
        return []

    poller._claim_batch_for_schema_inner = AsyncMock(side_effect=claim_schema)
    poller._release_claimed_tasks = AsyncMock(side_effect=PermissionError("release unavailable"))

    with pytest.raises(WorkerPartialClaimReleaseError, match="partial cross-schema claim batch") as raised:
        await poller.run()

    assert isinstance(raised.value.__cause__, PermissionError)
    assert tenant_b_calls == 1
    assert poller.is_ready is False


@pytest.mark.asyncio
async def test_cancelled_cross_schema_claim_releases_already_committed_rows():
    poller = _poller()
    poller._get_available_slots = AsyncMock(return_value=SlotAvailability(reserved={}, shared=2))
    poller._get_schemas = AsyncMock(return_value=["tenant_a", "tenant_b"])
    poller._scan_active_schemas = AsyncMock(return_value={"tenant_a", "tenant_b"})
    claimed = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000084",
        task_dict={"type": "graph_maintenance", "operation_type": "graph_maintenance"},
        schema="tenant_a",
    )
    tenant_b_entered = asyncio.Event()
    block_tenant_b = asyncio.Event()

    async def claim_schema(schema, _reserved, _shared):
        if schema == "tenant_a":
            return [claimed]
        tenant_b_entered.set()
        await block_tenant_b.wait()
        return []

    poller._claim_batch_for_schema_inner = AsyncMock(side_effect=claim_schema)
    poller._release_claimed_tasks = AsyncMock(return_value=None)

    claim_task = asyncio.create_task(poller.claim_batch())
    await asyncio.wait_for(tenant_b_entered.wait(), timeout=1)
    claim_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await claim_task

    poller._release_claimed_tasks.assert_awaited_once_with([claimed])


@pytest.mark.asyncio
async def test_shutdown_waits_for_claim_and_releases_without_spawning_new_work():
    poller = _poller()
    poller.recover_own_tasks = AsyncMock(return_value=0)
    claimed = ClaimedTask(
        operation_id="00000000-0000-0000-0000-000000000083",
        task_dict={"type": "graph_maintenance", "operation_type": "graph_maintenance"},
        schema="tenant_a",
    )
    claim_entered = asyncio.Event()
    release_claim = asyncio.Event()

    async def controlled_claim():
        claim_entered.set()
        await release_claim.wait()
        return [claimed]

    poller.claim_batch = AsyncMock(side_effect=controlled_claim)
    poller._release_claimed_tasks = AsyncMock(return_value=None)
    poller.execute_task = AsyncMock(return_value=None)

    run_task = asyncio.create_task(poller.run())
    await asyncio.wait_for(claim_entered.wait(), timeout=1)
    shutdown_task = asyncio.create_task(poller.shutdown_graceful(timeout=1))
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    release_claim.set()
    await asyncio.wait_for(run_task, timeout=1)
    await asyncio.wait_for(shutdown_task, timeout=1)

    poller._release_claimed_tasks.assert_awaited_once_with([claimed])
    poller.execute_task.assert_not_awaited()
    assert poller._active_tasks == {}
    assert poller._in_flight_count == 0


@pytest.mark.asyncio
async def test_shutdown_fails_closed_when_claim_cycle_cannot_quiesce():
    poller = _poller()
    poller._claim_cycle_done.clear()

    with pytest.raises(WorkerPartialClaimReleaseError, match="claim cycle did not quiesce") as raised:
        await poller.shutdown_graceful(timeout=0.01)

    assert isinstance(raised.value.__cause__, TimeoutError)
    assert poller._shutdown.is_set()


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


@pytest.mark.parametrize("terminal_path", ["completed", "failed", "consolidation"])
@pytest.mark.asyncio
async def test_real_terminal_helper_failure_drops_health_during_claim_and_fails_supervisor(terminal_path):
    """Every worker-owned terminal write fails closed, even during an in-flight claim."""
    backend = _DeniedAuthorityBackend()
    memory = object.__new__(MemoryEngine)
    memory._audit_logger = None
    memory._ext_ctx = MagicMock()
    memory._get_backend = AsyncMock(return_value=backend)
    memory._webhook_manager = None
    memory._handle_graph_maintenance = AsyncMock(return_value=None)
    memory._handle_consolidation = AsyncMock(return_value={"observations_created": 1})

    if terminal_path == "failed":
        memory._handle_graph_maintenance = AsyncMock(
            side_effect=ValueError("embedding 0 has dimension 0; expected 384")
        )

    operation_id = "00000000-0000-0000-0000-000000000099"
    task_type = "consolidation" if terminal_path == "consolidation" else "graph_maintenance"
    claimed = ClaimedTask(
        operation_id=operation_id,
        task_dict={
            "type": task_type,
            "operation_type": task_type,
            "operation_id": operation_id,
            "bank_id": "bank-test",
            "_worker_id": "worker-test",
            "_claim_token": CLAIM_TOKEN,
            "_operation_id": operation_id,
        },
        schema=None,
        claim_token=CLAIM_TOKEN,
    )

    poller = WorkerPoller(
        backend=backend,
        worker_id="worker-test",
        executor=memory.execute_task,
        tenant_extension=MagicMock(),
    )
    poller.recover_own_tasks = AsyncMock(return_value=0)
    poller._log_progress_if_due = AsyncMock()

    claim_in_flight = asyncio.Event()
    release_claim = asyncio.Event()
    claim_count = 0

    async def controlled_claim():
        nonlocal claim_count
        claim_count += 1
        if claim_count == 1:
            return [claimed]
        claim_in_flight.set()
        await release_claim.wait()
        return []

    poller.claim_batch = AsyncMock(side_effect=controlled_claim)

    health_memory = MagicMock()
    health_memory._pool = None
    health_memory.health_check = AsyncMock(return_value={"status": "healthy", "database": "connected"})
    app = create_worker_app(poller, health_memory)
    endpoint = _health_endpoint(app)

    poller_task = asyncio.create_task(poller.run())
    app.state.poller_task = poller_task
    http_task = asyncio.create_task(asyncio.Event().wait())
    supervisor_task = asyncio.create_task(_wait_for_shutdown_or_worker_failure(asyncio.Event(), poller_task, http_task))
    try:
        await asyncio.wait_for(claim_in_flight.wait(), timeout=1)
        await asyncio.wait_for(poller._fatal_task_event.wait(), timeout=1)

        assert poller.is_ready is False
        response = await endpoint()
        assert response.status_code == 503
        assert json.loads(response.body)["reason"] == "poller_not_ready"

        release_claim.set()
        with pytest.raises(RuntimeError, match="poller task failed") as raised:
            await supervisor_task

        poller_failure = raised.value.__cause__
        assert isinstance(poller_failure, WorkerBackgroundTaskError)
        terminal_failure = poller_failure.__cause__
        assert isinstance(terminal_failure, OperationTerminalStateError)
        assert terminal_path in str(terminal_failure) or terminal_path == "failed"
        assert isinstance(terminal_failure.__cause__, RuntimeError)
        assert poller.claim_batch.await_count == 2
    finally:
        release_claim.set()
        await _cancel(supervisor_task)
        await _cancel(poller_task)
        await _cancel(http_task)


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
