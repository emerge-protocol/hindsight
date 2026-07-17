"""
Command-line interface for Hindsight Worker.

Run the worker with:
    hindsight-worker

Stop with Ctrl+C (graceful shutdown).
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import socket
import sys
import warnings
from collections.abc import Callable

from ..config import get_config
from ..engine.task_backend import WorkerTaskBackend
from .poller import WorkerPoller

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def _install_shutdown_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    handler: Callable[[], None],
) -> bool:
    """Register SIGINT/SIGTERM handlers on the asyncio loop.

    Returns True when handlers were installed via ``loop.add_signal_handler``.
    Returns False on platforms (Windows ProactorEventLoop) where asyncio
    does not implement signal handlers; the caller falls back to Python's
    default SIGINT behavior, which still terminates the process on Ctrl+C
    but loses the in-loop two-stage graceful shutdown.
    """
    try:
        loop.add_signal_handler(signal.SIGINT, handler)
        loop.add_signal_handler(signal.SIGTERM, handler)
    except NotImplementedError:
        return False
    return True


async def _wait_for_shutdown_or_worker_failure(
    shutdown_requested: asyncio.Event,
    poller_task: asyncio.Task,
    http_task: asyncio.Task,
) -> None:
    """Wait for an operator shutdown, failing on either runtime task exit.

    The standalone worker is one availability unit: a poller without its HTTP
    readiness endpoint cannot be operated safely, and an HTTP server without a
    poller must never remain up reporting database-only health.  Therefore any
    completion of either task before an explicit shutdown is fatal, including a
    clean return or cancellation.
    """
    shutdown_task = asyncio.create_task(shutdown_requested.wait())
    try:
        done, _ = await asyncio.wait(
            {shutdown_task, poller_task, http_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task_name, task in (("poller", poller_task), ("HTTP server", http_task)):
            if task not in done:
                continue
            if task.cancelled():
                raise RuntimeError(f"Worker {task_name} task was cancelled unexpectedly")
            error = task.exception()
            if error is not None:
                raise RuntimeError(f"Worker {task_name} task failed") from error
            raise RuntimeError(f"Worker {task_name} task exited unexpectedly")

        # Only the explicit shutdown task completed; the runtime tasks are
        # still live and will be drained by the caller's graceful-shutdown path.
        if shutdown_task not in done:  # pragma: no cover - defensive invariant
            raise RuntimeError("Worker supervisor woke without a completed task")
    finally:
        if not shutdown_task.done():
            shutdown_task.cancel()
        try:
            await shutdown_task
        except asyncio.CancelledError:
            pass


async def _cancel_task(task: asyncio.Task | None) -> None:
    """Cancel and consume a task without replacing the supervisor's failure."""
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def create_worker_app(poller: WorkerPoller, memory):
    """Create a minimal FastAPI app for worker metrics and health."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from ..metrics import create_metrics_collector, get_metrics_collector, initialize_metrics

    app = FastAPI(
        title="Hindsight Worker",
        description="Worker process for distributed task execution",
    )
    # Filled immediately after task creation in ``run``.  Health remains 503
    # through the startup-recovery barrier and for any poller task exit.
    app.state.poller_task = None

    # Initialize OpenTelemetry metrics
    try:
        prometheus_reader = initialize_metrics(service_name="hindsight-worker", service_version="1.0.0")
        create_metrics_collector()
        app.state.prometheus_reader = prometheus_reader
        logger.info("Metrics initialized - available at /metrics endpoint")
    except Exception as e:
        logger.warning(f"Failed to initialize metrics: {e}. Metrics will be disabled.")
        app.state.prometheus_reader = None

    # Set up DB pool metrics if available
    metrics_collector = get_metrics_collector()
    if memory._pool is not None and hasattr(metrics_collector, "set_db_pool"):
        metrics_collector.set_db_pool(memory._pool)
        logger.info("DB pool metrics configured")

    @app.get(
        "/health",
        summary="Health check endpoint",
        description="Returns worker health status including database connectivity",
        tags=["Monitoring"],
    )
    async def health_endpoint():
        """Health check endpoint."""
        # Copy the engine payload before adding worker-local state so a cached
        # or test-double result cannot retain an earlier readiness downgrade.
        health = dict(await memory.health_check())
        poller_task = app.state.poller_task
        poller_alive = poller_task is not None and not poller_task.done()
        poller_ready = poller.is_ready
        health["worker_id"] = poller.worker_id
        health["is_shutdown"] = poller.is_shutdown
        health["poller_alive"] = poller_alive
        health["poller_ready"] = poller_ready
        if not poller_alive or not poller_ready or poller.is_shutdown:
            health["status"] = "unhealthy"
            if poller.is_shutdown:
                health["reason"] = "worker_shutting_down"
            elif not poller_alive:
                health["reason"] = "poller_not_running"
            else:
                health["reason"] = "poller_not_ready"
        status_code = 200 if health.get("status") == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)

    @app.get(
        "/metrics",
        summary="Prometheus metrics endpoint",
        description="Exports metrics in Prometheus format for scraping",
        tags=["Monitoring"],
    )
    async def metrics_endpoint():
        """Return Prometheus metrics."""
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/",
        summary="Worker info",
        description="Basic worker information",
        tags=["Info"],
    )
    async def root():
        """Return basic worker info."""
        return {
            "service": "hindsight-worker",
            "worker_id": poller.worker_id,
            "is_shutdown": poller.is_shutdown,
        }

    return app


def main():
    """Main entry point for the hindsight-worker CLI."""
    # Load configuration from environment
    config = get_config()

    parser = argparse.ArgumentParser(
        prog="hindsight-worker",
        description="Hindsight Worker - distributed task processor",
    )

    # Worker options
    parser.add_argument(
        "--worker-id",
        default=config.worker_id,
        help="Worker identifier (default: hostname, env: HINDSIGHT_API_WORKER_ID)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=config.worker_poll_interval_ms,
        help=f"Poll interval in milliseconds (default: {config.worker_poll_interval_ms}, env: HINDSIGHT_API_WORKER_POLL_INTERVAL_MS)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=config.worker_max_retries,
        help=f"Max retries before marking failed (default: {config.worker_max_retries}, env: HINDSIGHT_API_WORKER_MAX_RETRIES)",
    )

    # HTTP server options
    parser.add_argument(
        "--http-port",
        type=int,
        default=config.worker_http_port,
        help=f"HTTP port for metrics/health endpoints (default: {config.worker_http_port}, env: HINDSIGHT_API_WORKER_HTTP_PORT)",
    )
    parser.add_argument(
        "--http-host",
        default="0.0.0.0",
        help="HTTP host to bind (default: 0.0.0.0)",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {config.log_level}, env: HINDSIGHT_API_LOG_LEVEL)",
    )

    args = parser.parse_args()

    # Configure logging
    config.configure_logging()

    from ..utils import warn_if_container_default_worker_id

    warn_if_container_default_worker_id(args.worker_id)
    worker_id = args.worker_id or socket.gethostname()
    worker_id_source = "HINDSIGHT_API_WORKER_ID/--worker-id" if args.worker_id else "hostname (default)"
    logger.info(f"Worker id: {worker_id} (source: {worker_id_source})")

    # Import MemoryEngine here to avoid circular imports
    from .. import MemoryEngine

    print(f"Starting Hindsight Worker: {worker_id}")
    print(f"  Poll interval: {args.poll_interval}ms")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Max slots: {config.worker_max_slots}")
    reservations = config.worker_slot_reservations
    reservations_str = ", ".join(f"{k}={v}" for k, v in reservations.items()) if reservations else "none"
    shared_pool = max(0, config.worker_max_slots - sum(reservations.values()))
    print(f"  Slot reservations: {reservations_str}")
    print(f"  Shared pool: {shared_pool}")
    print(f"  HTTP server: {args.http_host}:{args.http_port}")
    print()

    # Global references for cleanup
    memory = None
    poller = None

    async def run():
        nonlocal memory, poller
        import uvicorn

        from ..extensions import OperationValidatorExtension, TenantExtension, load_extension

        # Load tenant extension BEFORE creating MemoryEngine so it can
        # set correct schema context during task execution. Without this,
        # _authenticate_tenant sees no extension and resets schema to "public",
        # causing worker writes to land in the wrong schema.
        tenant_extension = load_extension("TENANT", TenantExtension)

        # Load operation validator so workers can record usage metering
        # for async operations (e.g. refresh_mental_model after consolidation)
        operation_validator = load_extension("OPERATION_VALIDATOR", OperationValidatorExtension)
        if operation_validator:
            logger.info(f"Loaded operation validator: {operation_validator.__class__.__name__}")

        # Initialize MemoryEngine
        # Workers use WorkerTaskBackend: submit_task is a no-op because the
        # row already exists in async_operations.  Child tasks (e.g. consolidation
        # triggered by retain) will be picked up by the poller on the next cycle
        # instead of being executed inline, which avoids blocking the parent task.
        memory = MemoryEngine(
            run_migrations=False,  # Workers don't run migrations
            task_backend=WorkerTaskBackend(),
            tenant_extension=tenant_extension,
            operation_validator=operation_validator,
        )

        await memory.initialize()

        print(f"Database connected: {config.database_url}")

        if tenant_extension:
            print("Tenant extension loaded - schemas will be discovered dynamically on each poll")
        else:
            print(f"No tenant extension configured, using schema: {config.database_schema}")

        # Check if the backend supports the async worker/poller.
        if not memory._backend.supports_worker_poller:
            print("ERROR: Standalone worker is not supported on this database backend.")
            print("Operations run synchronously within the API process.")
            sys.exit(1)

        # Create a single poller that handles all schemas dynamically
        # Convert default schema to None for SQL compatibility (no schema prefix)
        from hindsight_api.config import DEFAULT_DATABASE_SCHEMA

        schema = None if config.database_schema == DEFAULT_DATABASE_SCHEMA else config.database_schema
        poller = WorkerPoller(
            backend=memory._backend,
            worker_id=worker_id,
            executor=memory.execute_task,
            poll_interval_ms=args.poll_interval,
            schema=schema,
            tenant_extension=tenant_extension,
            max_slots=config.worker_max_slots,
            slot_reservations=config.worker_slot_reservations,
            consolidation_bank_priority=config.worker_consolidation_bank_priority or None,
        )

        # Create the HTTP app for metrics/health
        app = create_worker_app(poller, memory)

        # Setup signal handlers for graceful shutdown using asyncio
        shutdown_requested = asyncio.Event()
        force_exit = False
        async_handlers_installed = False

        loop = asyncio.get_event_loop()

        def signal_handler():
            nonlocal force_exit
            if shutdown_requested.is_set():
                # Second signal = force exit
                print("\nReceived second signal, forcing immediate exit...")
                force_exit = True
                # Restore default handler so third signal kills process
                if async_handlers_installed:
                    loop.remove_signal_handler(signal.SIGINT)
                    loop.remove_signal_handler(signal.SIGTERM)
                sys.exit(1)
            else:
                print("\nReceived shutdown signal, initiating graceful shutdown...")
                print("(Press Ctrl+C again to force immediate exit)")
                shutdown_requested.set()

        async_handlers_installed = _install_shutdown_signal_handlers(loop, signal_handler)
        if not async_handlers_installed:
            # Windows ProactorEventLoop: asyncio.add_signal_handler is Unix-only
            # and raises NotImplementedError. Default Python SIGINT handler still
            # terminates the worker on Ctrl+C, just without the two-stage path.
            print(
                f"WARN: asyncio signal handlers unavailable on this platform "
                f"({sys.platform}); graceful two-stage shutdown disabled, "
                f"default Python SIGINT handler remains active.",
                flush=True,
            )

        # Create uvicorn config and server
        uvicorn_config = uvicorn.Config(
            app,
            host=args.http_host,
            port=args.http_port,
            log_level="info",  # Reduce uvicorn noise
            access_log=False,
        )
        server = uvicorn.Server(uvicorn_config)

        # Run and supervise the poller and HTTP server as one availability
        # unit.  Health is not ready until the poller has passed recovery.
        poller_task = asyncio.create_task(poller.run(), name="hindsight-worker-poller")
        app.state.poller_task = poller_task
        http_task = asyncio.create_task(server.serve(), name="hindsight-worker-http")

        print(f"Worker started. Metrics available at http://{args.http_host}:{args.http_port}/metrics")

        try:
            await _wait_for_shutdown_or_worker_failure(shutdown_requested, poller_task, http_task)
        except KeyboardInterrupt:
            print("\nReceived interrupt, initiating graceful shutdown...")
        finally:
            # Graceful shutdown also runs after a supervised task failure so no
            # peer task, DB pool, or in-flight operation survives the process.
            print("Shutting down HTTP server...")
            server.should_exit = True

            print("Waiting for poller to finish...")
            await poller.shutdown_graceful(timeout=30.0)
            await _cancel_task(poller_task)

            try:
                await asyncio.wait_for(asyncio.shield(http_task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            await _cancel_task(http_task)

            # Close memory engine
            await memory.close()
            print("Worker shutdown complete")

    def cleanup():
        """Synchronous cleanup for atexit."""
        if memory is not None and memory._pg0 is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(memory._pg0.stop())
                loop.close()
                print("\npg0 stopped.")
            except Exception as e:
                print(f"\nError stopping pg0: {e}")

    atexit.register(cleanup)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nWorker interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
