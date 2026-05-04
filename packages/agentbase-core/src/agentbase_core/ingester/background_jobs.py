"""BackgroundJobRunner — persistent background task executor."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiosqlite

from ..models.context_entry import _new_ulid, _utcnow
from ..store.connection import ConnectionPool

logger = logging.getLogger(__name__)


class BackgroundJobRunner:
    """Background task runner — SQLite-persisted + in-memory queue acceleration.

    Job types: embedding, layer_gen, reindex, graph_extract, session_extract
    """

    def __init__(self, pool: ConnectionPool, max_concurrent: int = 5) -> None:
        self._pool = pool
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._memory_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._handlers: dict[str, callable] = {}

    def register_handler(self, job_type: str, handler: callable) -> None:
        """Register a handler function for a job type."""
        self._handlers[job_type] = handler

    async def submit(self, job_type: str, target_id: str) -> str:
        """Submit a background job, persist to background_jobs table."""
        job_id = _new_ulid()
        now = _utcnow().isoformat()

        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO background_jobs (id, job_type, target_id, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
                (job_id, job_type, target_id, now, now),
            )
            await conn.commit()

        # Put in memory queue for acceleration
        try:
            self._memory_queue.put_nowait(job_id)
        except asyncio.QueueFull:
            pass

        return job_id

    async def retry_failed(self, job_id: str) -> None:
        """Retry a failed job."""
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'pending', updated_at = ? WHERE id = ? AND status = 'failed'",
                (now, job_id),
            )
            await conn.commit()
        try:
            self._memory_queue.put_nowait(job_id)
        except asyncio.QueueFull:
            pass

    async def resume_pending(self) -> int:
        """Resume all pending jobs on startup. Returns count of resumed jobs."""
        resumed = 0
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT id FROM background_jobs WHERE status = 'pending' ORDER BY created_at"
            )
            rows = await cursor.fetchall()

        for row in rows:
            job_id = row[0]
            try:
                self._memory_queue.put_nowait(job_id)
                resumed += 1
            except asyncio.QueueFull:
                break

        if resumed > 0:
            logger.info(f"Resumed {resumed} pending background jobs")
        return resumed

    async def start(self) -> None:
        """Start the background worker."""
        self._running = True
        await self.resume_pending()
        asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False

    async def _worker(self) -> None:
        """Background worker loop."""
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._memory_queue.get(), timeout=1.0)
                async with self._semaphore:
                    await self._execute_job(job_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Background worker error: {e}")

    async def _execute_job(self, job_id: str) -> None:
        """Execute a single background job."""
        # Read job details
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT job_type, target_id, retry_count, max_retries FROM background_jobs WHERE id = ?",
                (job_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return

        job_type, target_id, retry_count, max_retries = row

        # Mark as running
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            await conn.commit()

        # Execute handler
        handler = self._handlers.get(job_type)
        if handler is None:
            logger.warning(f"No handler for job type: {job_type}")
            await self._mark_done(job_id)
            return

        try:
            await handler(target_id)
            await self._mark_done(job_id)
        except Exception as e:
            logger.warning(f"Background job {job_id} ({job_type}) failed: {e}")
            await self._mark_failed(job_id, str(e), retry_count, max_retries)

    async def _mark_done(self, job_id: str) -> None:
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = 'done', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            await conn.commit()

    async def _mark_failed(self, job_id: str, error: str, retry_count: int, max_retries: int) -> None:
        now = _utcnow().isoformat()
        new_retry = retry_count + 1
        new_status = "pending" if new_retry < max_retries else "failed"

        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE background_jobs SET status = ?, retry_count = ?, last_error = ?, updated_at = ? WHERE id = ?",
                (new_status, new_retry, error[:500], now, job_id),
            )
            await conn.commit()

        if new_status == "pending":
            try:
                self._memory_queue.put_nowait(job_id)
            except asyncio.QueueFull:
                pass

    async def list_jobs(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """List background jobs."""
        if status:
            async with self._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    "SELECT id, job_type, target_id, status, retry_count, last_error, created_at FROM background_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
                rows = await cursor.fetchall()
        else:
            async with self._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    "SELECT id, job_type, target_id, status, retry_count, last_error, created_at FROM background_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()

        return [
            {
                "id": r[0], "job_type": r[1], "target_id": r[2],
                "status": r[3], "retry_count": r[4], "last_error": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
