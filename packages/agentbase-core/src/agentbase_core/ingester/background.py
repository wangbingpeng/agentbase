"""BackgroundJobRunner — persistent async job queue with startup recovery.

Jobs are persisted in the background_jobs SQLite table and processed
by a single asyncio task. On startup, any "running" jobs are reset
to "pending" for automatic recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

from ..exceptions import BackgroundJobError
from ..models.context_entry import _new_ulid, _utcnow
from ..store.connection import ConnectionPool

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundJob:
    """A persistent background job record."""

    def __init__(
        self,
        id: str | None = None,
        job_type: str = "",
        payload: dict[str, Any] | None = None,
        status: str = JobStatus.PENDING,
        attempts: int = 0,
        max_attempts: int = 3,
        error_message: str | None = None,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        self.id = id or _new_ulid()
        self.job_type = job_type
        self.payload = payload or {}
        self.status = status
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.error_message = error_message
        self.created_at = created_at or _utcnow()
        self.started_at = started_at
        self.completed_at = completed_at


# Type alias for job handlers
JobHandler = Callable[[dict[str, Any]], Awaitable[None]]


class BackgroundJobRunner:
    """Persistent background job runner with automatic recovery.

    Features:
    - Jobs persisted in SQLite for durability
    - Automatic recovery on startup (resets "running" → "pending")
    - Configurable max retries per job
    - Async processing loop
    """

    def __init__(
        self,
        pool: ConnectionPool,
        poll_interval: float = 1.0,
        max_attempts: int = 3,
    ) -> None:
        self._pool = pool
        self._poll_interval = poll_interval
        self._max_attempts = max_attempts
        self._handlers: dict[str, JobHandler] = {}
        self._task: asyncio.Task | None = None
        self._running = False

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler

    async def submit(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> BackgroundJob:
        """Submit a new job to the queue."""
        if job_type not in self._handlers:
            raise BackgroundJobError(f"No handler registered for job type: {job_type}")

        job = BackgroundJob(
            job_type=job_type,
            payload=payload or {},
            max_attempts=max_attempts or self._max_attempts,
        )

        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO background_jobs (id, job_type, payload, status, attempts, max_attempts, error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id, job.job_type, json.dumps(job.payload),
                    job.status, job.attempts, job.max_attempts,
                    job.error_message, job.created_at.isoformat(),
                ),
            )
            await conn.commit()

        logger.info(f"Job submitted: {job.id} ({job.job_type})")
        return job

    async def get_job(self, job_id: str) -> BackgroundJob | None:
        """Get a job by ID."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT * FROM background_jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    async def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
    ) -> list[BackgroundJob]:
        """List jobs with optional filters."""
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                f"SELECT * FROM background_jobs{where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            rows = await cursor.fetchall()
            return [self._row_to_job(r) for r in rows]

    async def start(self) -> None:
        """Start the background job processing loop."""
        if self._running:
            return

        # Recovery: reset any "running" jobs back to "pending"
        await self._recover_jobs()

        self._running = True
        self._task = asyncio.create_task(self._processing_loop())
        logger.info("BackgroundJobRunner started")

    async def stop(self) -> None:
        """Stop the background job processing loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("BackgroundJobRunner stopped")

    async def _processing_loop(self) -> None:
        """Main processing loop — polls for pending jobs."""
        while self._running:
            try:
                job = await self._fetch_next_pending()
                if job is not None:
                    asyncio.create_task(self._execute_job(job))
                else:
                    await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Job processing loop error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _fetch_next_pending(self) -> BackgroundJob | None:
        """Fetch the next pending job (FIFO)."""
        async with self._pool.get_write_conn() as conn:
            cursor = await conn.execute(
                "SELECT * FROM background_jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            job = self._row_to_job(row)
            # Mark as running
            now = _utcnow().isoformat()
            await conn.execute(
                "UPDATE background_jobs SET status = 'running', attempts = ?, started_at = ? WHERE id = ?",
                (job.attempts + 1, now, job.id),
            )
            await conn.commit()

        job.status = JobStatus.RUNNING
        job.attempts += 1
        return job

    async def _execute_job(self, job: BackgroundJob) -> None:
        """Execute a single job with error handling."""
        handler = self._handlers.get(job.job_type)
        if handler is None:
            await self._mark_failed(job, f"No handler for type: {job.job_type}")
            return

        try:
            await handler(job.payload)
            await self._mark_completed(job)
            logger.info(f"Job completed: {job.id} ({job.job_type})")
        except Exception as e:
            logger.error(f"Job failed: {job.id} ({job.job_type}): {e}")
            if job.attempts >= job.max_attempts:
                await self._mark_failed(job, str(e))
            else:
                await self._mark_pending(job)

    async def _mark_completed(self, job: BackgroundJob) -> None:
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'completed', completed_at = ? WHERE id = ?",
                (now, job.id),
            )
            await conn.commit()

    async def _mark_failed(self, job: BackgroundJob, error: str) -> None:
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'failed', error_message = ?, completed_at = ? WHERE id = ?",
                (error[:500], now, job.id),
            )
            await conn.commit()

    async def _mark_pending(self, job: BackgroundJob) -> None:
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'pending' WHERE id = ?",
                (job.id,),
            )
            await conn.commit()

    async def _recover_jobs(self) -> None:
        """Reset any 'running' jobs back to 'pending' on startup."""
        async with self._pool.get_write_conn() as conn:
            cursor = await conn.execute(
                "UPDATE background_jobs SET status = 'pending' WHERE status = 'running'"
            )
            count = cursor.rowcount
            await conn.commit()
            if count > 0:
                logger.info(f"Recovered {count} running jobs → pending")

    @staticmethod
    def _row_to_job(row: tuple) -> BackgroundJob:
        return BackgroundJob(
            id=row[0],
            job_type=row[1],
            payload=json.loads(row[2]) if row[2] else {},
            status=row[3],
            attempts=row[4] or 0,
            max_attempts=row[5] or 3,
            error_message=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else _utcnow(),
            started_at=datetime.fromisoformat(row[8]) if row[8] else None,
            completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )
