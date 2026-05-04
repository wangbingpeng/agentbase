"""SQLite connection pool — single-write, multi-read."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

from ..exceptions import StorageError

# PRAGMA settings for optimal OLTP performance
_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA cache_size = -64000",        # 64MB
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 268435456",      # 256MB
    "PRAGMA busy_timeout = 5000",        # 5 seconds
]


class ConnectionPool:
    """SQLite connection pool with single-writer, multi-reader concurrency."""

    def __init__(self, db_path: Path, pool_size: int = 5) -> None:
        self._db_path = db_path
        self._pool_size = pool_size
        self._write_lock = asyncio.Lock()
        self._read_pool: asyncio.Queue[aiosqlite.Connection] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        self._read_pool = asyncio.Queue(maxsize=self._pool_size)

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create read connections
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(str(self._db_path))
            await self._apply_pragmas(conn)
            await conn.execute("PRAGMA query_only = ON")
            self._read_pool.put_nowait(conn)

        self._initialized = True

    async def _apply_pragmas(self, conn: aiosqlite.Connection) -> None:
        """Apply PRAGMA settings to a connection."""
        for pragma in _PRAGMAS:
            await conn.execute(pragma)

    @asynccontextmanager
    async def get_read_conn(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a read-only connection from the pool."""
        if self._read_pool is None:
            raise StorageError("Connection pool not initialized")

        conn = await self._read_pool.get()
        try:
            yield conn
        finally:
            await self._read_pool.put(conn)

    @asynccontextmanager
    async def get_write_conn(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a write connection (serialized via write lock)."""
        if not self._initialized:
            raise StorageError("Connection pool not initialized")

        async with self._write_lock:
            conn = await aiosqlite.connect(str(self._db_path))
            try:
                await self._apply_pragmas(conn)
                yield conn
            finally:
                await conn.close()

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._read_pool is None:
            return

        while not self._read_pool.empty():
            try:
                conn = self._read_pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break

        self._initialized = False
