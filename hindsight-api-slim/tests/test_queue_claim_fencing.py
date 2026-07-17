"""Deterministic same-worker ABA tests for deep retain queue state access."""

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.operation_metadata import RetainExtractionErrors
from hindsight_api.engine.queue_claim import QueueClaimPredicate, bind_queue_claim, queue_claim_predicate
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    RetainContent,
    _read_batch_operation_metadata,
    _write_batch_extraction_errors,
    _write_batch_operation_state,
    extract_facts_from_contents_batch_api,
)
from hindsight_api.engine.retain.orchestrator import (
    _persist_facts_committed_checkpoint,
    _persist_operation_document_id,
    _read_operation_metadata,
    _streaming_retain_batch,
)
from hindsight_api.worker.exceptions import OperationQueueAuthorityError
from hindsight_api.worker.poller import ClaimedTask, WorkerPoller

WORKER_ID = "worker-test"
CLAIM_TOKEN = "claim-generation-a"
OPERATION_ID = "00000000-0000-0000-0000-000000000076"


def test_queue_claim_predicate_has_named_sql_and_args() -> None:
    predicate = queue_claim_predicate(1)

    assert predicate == QueueClaimPredicate(sql="", args=())


class _ConnectionBackend:
    _wraps_backend = True

    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        context = AsyncMock()
        context.__aenter__.return_value = self.connection
        context.__aexit__.return_value = False
        return context


async def _under_claim(action: Callable[[], Awaitable[object]]) -> object:
    @bind_queue_claim("task_dict")
    async def invoke(task_dict):
        return await action()

    return await invoke({"_worker_id": WORKER_ID, "_claim_token": CLAIM_TOKEN})


@pytest.mark.parametrize(
    "path",
    ["document_id", "facts_committed", "extraction_errors", "batch_state"],
)
@pytest.mark.asyncio
async def test_deep_retain_metadata_write_zero_fails_closed(path):
    connection = MagicMock()
    connection.execute = AsyncMock(return_value="UPDATE 0")
    pool = _ConnectionBackend(connection)

    async def write():
        if path == "document_id":
            await _persist_operation_document_id(pool, OPERATION_ID, "doc-1")
        elif path == "facts_committed":
            await _persist_facts_committed_checkpoint(pool, OPERATION_ID, "doc-1", 3)
        elif path == "extraction_errors":
            await _write_batch_extraction_errors(
                pool,
                OPERATION_ID,
                None,
                RetainExtractionErrors(count=1, sample=["chunk failed"]),
            )
        else:
            await _write_batch_operation_state(
                pool,
                "async_operations",
                OPERATION_ID,
                {"batch_id": "batch-1"},
            )

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(write)

    sql, *args = connection.execute.await_args.args
    assert "status = 'processing'" in sql
    assert "worker_id" in sql and "claim_token" in sql
    assert args[-2:] == [WORKER_ID, CLAIM_TOKEN]


@pytest.mark.parametrize("path", ["orchestrator", "batch_api"])
@pytest.mark.asyncio
async def test_deep_retain_recovery_read_zero_fails_closed(path):
    connection = MagicMock()
    connection.fetchrow = AsyncMock(return_value=None)
    pool = _ConnectionBackend(connection)

    async def read():
        if path == "orchestrator":
            return await _read_operation_metadata(pool, OPERATION_ID, "recover retain state for")
        return await _read_batch_operation_metadata(
            pool,
            "async_operations",
            OPERATION_ID,
            "recover provider batch state for",
        )

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(read)

    sql, *args = connection.fetchrow.await_args.args
    assert "status = 'processing'" in sql
    assert "worker_id" in sql and "claim_token" in sql
    assert args[-2:] == [WORKER_ID, CLAIM_TOKEN]


@pytest.mark.asyncio
async def test_direct_retain_metadata_write_keeps_legacy_unfenced_behavior():
    connection = MagicMock()
    connection.execute = AsyncMock(return_value="UPDATE 0")
    pool = _ConnectionBackend(connection)

    await _persist_operation_document_id(pool, OPERATION_ID, "doc-1")

    sql, *_args = connection.execute.await_args.args
    assert "status = 'processing'" not in sql
    assert "claim_token" not in sql


@pytest.mark.asyncio
async def test_poller_does_not_reclassify_queue_authority_as_task_failure():
    executor = AsyncMock(side_effect=OperationQueueAuthorityError("claim moved"))
    poller = WorkerPoller(
        backend=MagicMock(),
        worker_id=WORKER_ID,
        executor=executor,
        tenant_extension=MagicMock(),
    )
    poller._mark_failed = AsyncMock()
    task = ClaimedTask(
        operation_id=OPERATION_ID,
        task_dict={
            "type": "batch_retain",
            "_worker_id": WORKER_ID,
            "_claim_token": CLAIM_TOKEN,
        },
        schema=None,
        claim_token=CLAIM_TOKEN,
    )

    with pytest.raises(OperationQueueAuthorityError, match="claim moved"):
        await poller._execute_task_inner(task)

    poller._mark_failed.assert_not_awaited()


def _batch_config() -> MagicMock:
    config = MagicMock()
    config.retain_extract_causal_links = False
    config.retain_chunk_size = 4000
    config.retain_structured_chunk_size = 4000
    config.retain_batch_poll_interval_seconds = 1
    return config


def _batch_llm() -> MagicMock:
    llm = MagicMock()
    llm.provider = "openai"
    llm._provider_impl = MagicMock()
    llm._provider_impl.supports_batch_api = AsyncMock(return_value=True)
    llm._provider_impl.submit_batch = AsyncMock(return_value={"batch_id": "batch-1"})
    llm._provider_impl.get_batch_status = AsyncMock()
    llm._provider_impl.retrieve_batch_results = AsyncMock(return_value=[])
    return llm


async def _run_minimal_batch_extract(pool, llm):
    with (
        patch(
            "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
            return_value=("prompt", {"type": "object"}),
        ),
        patch("hindsight_api.engine.retain.fact_extraction.chunk_text", return_value=["chunk"]),
        patch("hindsight_api.engine.retain.fact_extraction._build_user_message", return_value="message"),
        patch("hindsight_api.engine.retain.fact_extraction._build_request_body", return_value={}),
    ):
        return await extract_facts_from_contents_batch_api(
            contents=[RetainContent(content="hello")],
            llm_config=llm,
            agent_name="agent",
            config=_batch_config(),
            pool=pool,
            operation_id=OPERATION_ID,
            schema=None,
        )


@pytest.mark.asyncio
async def test_batch_submit_rechecks_claim_immediately_before_provider_side_effect():
    connection = MagicMock()
    connection.fetchrow = AsyncMock(
        side_effect=[
            {"result_metadata": {}},
            None,
        ]
    )
    pool = _ConnectionBackend(connection)
    llm = _batch_llm()

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(lambda: _run_minimal_batch_extract(pool, llm))

    llm._provider_impl.submit_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_submit_post_side_effect_state_write_zero_fails_closed():
    connection = MagicMock()
    connection.fetchrow = AsyncMock(
        side_effect=[
            {"result_metadata": {}},
            {"result_metadata": {}},
        ]
    )
    connection.execute = AsyncMock(return_value="UPDATE 0")
    pool = _ConnectionBackend(connection)
    llm = _batch_llm()

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(lambda: _run_minimal_batch_extract(pool, llm))

    llm._provider_impl.submit_batch.assert_awaited_once()
    llm._provider_impl.get_batch_status.assert_not_awaited()


def _batch_status(status: str) -> dict:
    return {
        "status": status,
        "request_counts": {"completed": int(status == "completed"), "total": 1},
    }


@pytest.mark.asyncio
async def test_batch_poll_rechecks_claim_before_waiting_or_consuming_results():
    connection = MagicMock()
    connection.fetchrow = AsyncMock(
        side_effect=[
            {"result_metadata": {}},
            {"result_metadata": {}},
            None,
        ]
    )
    connection.execute = AsyncMock(return_value="UPDATE 1")
    pool = _ConnectionBackend(connection)
    llm = _batch_llm()
    llm._provider_impl.get_batch_status = AsyncMock(return_value=_batch_status("in_progress"))

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(lambda: _run_minimal_batch_extract(pool, llm))

    llm._provider_impl.get_batch_status.assert_awaited_once_with("batch-1")
    llm._provider_impl.retrieve_batch_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_results_recheck_claim_after_provider_retrieval():
    connection = MagicMock()
    connection.fetchrow = AsyncMock(
        side_effect=[
            {"result_metadata": {}},
            {"result_metadata": {}},
            {"result_metadata": {"batch_id": "batch-1"}},
            None,
        ]
    )
    connection.execute = AsyncMock(return_value="UPDATE 1")
    pool = _ConnectionBackend(connection)
    llm = _batch_llm()
    llm._provider_impl.get_batch_status = AsyncMock(return_value=_batch_status("completed"))
    llm._provider_impl.retrieve_batch_results = AsyncMock(return_value=[])

    with pytest.raises(OperationQueueAuthorityError, match="claim generation"):
        await _under_claim(lambda: _run_minimal_batch_extract(pool, llm))

    llm._provider_impl.get_batch_status.assert_awaited_once_with("batch-1")
    llm._provider_impl.retrieve_batch_results.assert_awaited_once_with("batch-1")


@pytest.mark.asyncio
async def test_streaming_progress_authority_loss_stops_after_committed_batch():
    connection = MagicMock()
    connection.fetchrow = AsyncMock(return_value=None)
    connection.fetchval = AsyncMock(return_value="__pending__")
    connection.execute = AsyncMock(return_value="INSERT 0 1")
    transaction = AsyncMock()
    transaction.__aenter__.return_value = None
    transaction.__aexit__.return_value = False
    connection.transaction.return_value = transaction
    pool = _ConnectionBackend(connection)
    pool.ops = MagicMock()
    progress = AsyncMock(side_effect=OperationQueueAuthorityError("claim moved"))

    with (
        patch(
            "hindsight_api.engine.retain.orchestrator._extract_and_embed",
            new=AsyncMock(return_value=([], [], [], TokenUsage())),
        ),
        patch(
            "hindsight_api.engine.retain.orchestrator._read_operation_metadata",
            new=AsyncMock(return_value={"result_metadata": {}}),
        ),
        patch(
            "hindsight_api.engine.retain.orchestrator.fact_storage.handle_document_tracking",
            new=AsyncMock(),
        ),
        patch(
            "hindsight_api.engine.retain.orchestrator._redact_document_body",
            side_effect=lambda body, _config: body,
        ),
    ):
        with pytest.raises(OperationQueueAuthorityError, match="claim moved"):
            await _streaming_retain_batch(
                pool=pool,
                embeddings_model=MagicMock(),
                llm_config=MagicMock(),
                entity_resolver=MagicMock(),
                format_date_fn=MagicMock(),
                bank_id="bank-test",
                contents_dicts=[{"content": "hello"}],
                contents=[RetainContent(content="hello")],
                config=MagicMock(),
                document_id="doc-1",
                is_first_batch=True,
                fact_type_override=None,
                document_tags=None,
                agent_name="agent",
                log_buffer=[],
                start_time=0.0,
                all_pre_chunks=["chunk"],
                chunk_to_content=[0],
                chunk_batch_size=1,
                operation_id=OPERATION_ID,
                progress_callback=progress,
            )

    progress.assert_awaited_once()
