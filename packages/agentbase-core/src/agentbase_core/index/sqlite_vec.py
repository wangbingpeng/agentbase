"""SQLiteVecIndex — vector search using sqlite-vec extension.

Per SPEC §6.3: sqlite-vec based vector index with cosine distance.
sqlite-vec is a required dependency. Vector search is controlled by
config.index.vector_enabled (default False).
"""

from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from ..exceptions import IndexOpError
from ..models.context_entry import ContextEntry
from ..models.query import SearchResult
from .base import AbstractIndex

logger = logging.getLogger(__name__)


class SQLiteVecIndex(AbstractIndex):
    """Vector search index using sqlite-vec extension.

    Creates vec_meta and vec_context tables per SPEC §19.3.
    sqlite-vec is a required dependency; this class is only instantiated
    when config.index.vector_enabled=True.
    """

    def __init__(self, pool: Any, dimensions: int = 1536) -> None:
        self._pool = pool
        self._dimensions = dimensions

    async def initialize(self) -> None:
        """Load sqlite-vec extension and create vector tables."""
        import sqlite_vec

        async with self._pool.get_write_conn() as conn:
            await conn.enable_load_extension(True)
            await conn.load_extension(sqlite_vec.loadable_path())

            # Create vec_meta table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vec_meta (
                    rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id    TEXT NOT NULL UNIQUE,
                    context_type  TEXT NOT NULL,
                    scope         TEXT NOT NULL,
                    owner_id      TEXT,
                    confidence    REAL NOT NULL DEFAULT 1.0
                )
            """)

            # Create vec_context virtual table
            await conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_context USING vec0(
                    embedding float[{self._dimensions}] distance_metric=cosine
                )
            """)

            await conn.commit()
        logger.info(f"sqlite-vec index initialized (dimensions={self._dimensions})")

    async def add(self, entry: ContextEntry) -> None:
        """Add entry to vector index.

        Requires that entry has embedding_hash set and the actual
        embedding vector stored in embedding_cache.  Reads the vector
        from embedding_cache and writes it into vec_context.
        """
        if not entry.embedding_hash:
            return

        # Read the embedding vector from embedding_cache table
        embedding_blob = await self._read_embedding_blob(entry.embedding_hash)
        if embedding_blob is None:
            logger.debug(f"No cached embedding for {entry.id}, skipping vec insert")
            return

        try:
            async with self._pool.get_write_conn() as conn:
                # Load extension for this connection
                import sqlite_vec
                await conn.enable_load_extension(True)
                await conn.load_extension(sqlite_vec.loadable_path())

                # Insert into vec_meta
                cursor = await conn.execute(
                    "INSERT OR IGNORE INTO vec_meta (context_id, context_type, scope, owner_id, confidence) VALUES (?, ?, ?, ?, ?)",
                    (entry.id, entry.context_type.value, entry.scope.value, entry.owner_id, entry.confidence),
                )
                meta_rowid = cursor.lastrowid

                if meta_rowid == 0:
                    # Already exists, get rowid
                    row = await conn.execute(
                        "SELECT rowid FROM vec_meta WHERE context_id = ?",
                        (entry.id,),
                    )
                    result = await row.fetchone()
                    meta_rowid = result[0] if result else None

                if meta_rowid is None:
                    logger.warning(f"Failed to get meta_rowid for {entry.id}")
                    return

                # Insert embedding into vec_context
                await conn.execute(
                    "INSERT OR REPLACE INTO vec_context (rowid, embedding) VALUES (?, ?)",
                    (meta_rowid, embedding_blob),
                )

                await conn.commit()
        except Exception as e:
            logger.warning(f"Vector index add failed for {entry.id}: {e}")

    async def _read_embedding_blob(self, content_hash: str) -> bytes | None:
        """Read embedding vector blob from embedding_cache table."""
        try:
            async with self._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
                    (content_hash,),
                )
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.debug(f"Could not read embedding cache for {content_hash}: {e}")
            return None

    async def search(
        self,
        query: str,
        top_k: int = 10,
        context_type: str | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        query_embedding: list[float] | None = None,
        fts_column: str | None = None,    # unused by vec; for interface compatibility
        include_statuses: list[str] | None = None,  # unused by vec
        date_from: str | None = None,
        date_to: str | None = None,
        speaker: str | None = None,
    ) -> list[SearchResult]:
        """Vector similarity search using cosine distance.

        Optional filters (applied via JOIN with context_entries):
        - date_from / date_to: filter by ce.created_at range (ISO format strings)
        - speaker: filter by role tag or extra.role (case-insensitive match)
        """
        if query_embedding is None:
            return []

        import struct

        blob = struct.pack(f"<{len(query_embedding)}f", *query_embedding)

        conditions = []
        params: list[Any] = [blob, top_k]

        if context_type:
            conditions.append("vm.context_type = ?")
            params.append(context_type)
        if scope:
            conditions.append("vm.scope = ?")
            params.append(scope)
        if owner_id:
            conditions.append("vm.owner_id = ?")
            params.append(owner_id)

        # Time-aware filter: date_from / date_to on ce.created_at
        if date_from:
            conditions.append("ce.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("ce.created_at <= ?")
            params.append(date_to)

        # Speaker filter: match against tags or extra.role
        if speaker:
            conditions.append(
                "(ce.tags LIKE ? OR json_extract(ce.extra, '$.role') = ?)"
            )
            params.append(f"%speaker:{speaker}%")
            params.append(speaker)

        where = " AND ".join(conditions)
        where_clause = f"AND {where}" if where else ""

        sql = f"""
        SELECT v.rowid, v.distance, vm.context_id, vm.context_type, vm.scope, vm.confidence
        FROM vec_context v
        JOIN vec_meta vm ON vm.rowid = v.rowid
        JOIN context_entries ce ON ce.id = vm.context_id
        WHERE v.embedding MATCH ?
          AND k = ?
          {where_clause}
        ORDER BY v.distance
        """

        try:
            async with self._pool.get_read_conn() as conn:
                # Load sqlite-vec extension
                import sqlite_vec
                await conn.enable_load_extension(True)
                await conn.load_extension(sqlite_vec.loadable_path())

                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        results = []
        for row in rows:
            distance = row[1]
            context_id = row[2]
            # Convert cosine distance to similarity score (1 - distance)
            score = max(0.0, 1.0 - distance)
            results.append(
                SearchResult(
                    entry=ContextEntry(id=context_id),  # Placeholder; resolved by HybridIndex
                    score=score,
                    score_breakdown={"vector": score},
                    ranking_stage="vector",
                    matched_by="vector",
                )
            )

        return results

    async def remove(self, entry_id: str) -> None:
        """Remove from vector index."""
        try:
            async with self._pool.get_write_conn() as conn:
                await conn.execute("DELETE FROM vec_meta WHERE context_id = ?", (entry_id,))
                await conn.commit()
        except Exception as e:
            logger.warning(f"Vector index remove failed for {entry_id}: {e}")

    async def count(self) -> int:
        """Count entries in vector index."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM vec_meta")
            row = await cursor.fetchone()
            return row[0] if row else 0
