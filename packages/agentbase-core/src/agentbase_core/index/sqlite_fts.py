"""SQLite FTS5 full-text search index."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import aiosqlite

from ..exceptions import IndexOpError
from ..models.context_entry import ContextEntry, EntryStatus
from ..models.query import SearchResult
from .base import AbstractIndex

logger = logging.getLogger(__name__)


def normalize_query(text: str, tokenizer: str = "auto") -> str:
    """Normalize and sanitize a query string for FTS5 MATCH.

    Uses the configured tokenizer (jieba/char) for CJK word segmentation,
    then filters out FTS5 operators and truncates.

    Tokens are joined with OR logic so that entries matching ANY token
    are returned (broad recall), and BM25 ranking prioritizes entries
    matching MORE tokens.  This is essential for turn-level granularity
    where a single turn is unlikely to contain all query words.

    To avoid FTS5 syntax errors with overly long queries:
    - Single-character tokens are filtered out (noise for English)
    - CJK single characters are kept (meaningful in Chinese/Japanese)
    - Maximum 20 tokens are used
    """
    from .tokenizer import tokenize_query, _is_cjk

    tokens = tokenize_query(text, tokenizer=tokenizer)
    if not tokens:
        return ""

    # Filter out single-char non-CJK tokens (noise), keep CJK singles
    filtered = []
    for t in tokens:
        if len(t) == 1:
            if _is_cjk(t):
                filtered.append(t)  # CJK single chars are meaningful
            # else: skip single-char Latin/digits (noise)
        else:
            filtered.append(t)

    # Limit to 20 tokens to avoid FTS5 syntax issues
    filtered = filtered[:20]

    if not filtered:
        return ""

    # Use OR logic for broad recall; BM25 ranking will prioritize
    # entries with more matching tokens.
    result_text = " OR ".join(filtered)

    # Truncate at 512 chars
    if len(result_text) > 512:
        result_text = result_text[:512]

    return result_text


class SQLiteFTSIndex(AbstractIndex):
    """FTS5 full-text search index backed by SQLite."""

    def __init__(self, pool: Any, tokenizer: str = "auto") -> None:
        """Initialize with a ConnectionPool."""
        self._pool = pool
        self._tokenizer = tokenizer

    async def add(self, entry: ContextEntry) -> None:
        """FTS5 index is auto-updated via triggers; no-op here."""
        # The FTS5 index is kept in sync via SQLite triggers (see migrator.py).
        # This method is a no-op for the FTS5 backend.
        pass

    async def add_batch(self, entries: list[ContextEntry]) -> None:
        """FTS5 batch add — also a no-op since FTS5 is auto-synced via triggers."""
        # FTS5 is kept in sync via triggers on context_entries table.
        # When entries are batch-inserted into context_entries via store.add(),
        # the FTS5 triggers fire automatically for each INSERT.
        pass

    async def search(
        self,
        query: str,
        top_k: int = 10,
        context_type: str | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        fts_column: str | None = None,
        include_statuses: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker: str | None = None,
    ) -> list[SearchResult]:
        """Search using FTS5 BM25 ranking.

        Optional filters:
        - date_from / date_to: filter by ce.created_at range (ISO format strings)
        - speaker: filter by role tag or extra.role (case-insensitive match)
        """
        normalized = normalize_query(query, tokenizer=self._tokenizer)
        if not normalized:
            return []

        # Default: search fts_text column (tokenized) for CJK-aware matching.
        # If fts_column is explicitly set, use that instead.
        if not fts_column:
            fts_column = "fts_text"

        # Build FTS MATCH expression with optional column filter
        if fts_column:
            match_expr = f"{fts_column} : {normalized}"
        else:
            match_expr = normalized

        # Build WHERE conditions for context_entries
        # Per SPEC §4.8: default search returns only status='active'
        params: list[Any] = []
        if include_statuses:
            placeholders = ", ".join("?" for _ in include_statuses)
            conditions = [f"ce.status IN ({placeholders})"]
            params.extend(include_statuses)
        else:
            conditions = ["ce.status = 'active'"]
        params.insert(0, match_expr)  # match_expr is the first param

        if context_type:
            conditions.append("ce.context_type = ?")
            params.append(context_type)
        if scope:
            conditions.append("ce.scope = ?")
            params.append(scope)
        if owner_id:
            conditions.append("ce.owner_id = ?")
            params.append(owner_id)

        # Time-aware filter: date_from / date_to on ce.created_at
        if date_from:
            conditions.append("ce.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("ce.created_at <= ?")
            params.append(date_to)

        # Speaker filter: match against tags (e.g., "speaker:Caroline") or json_extract(extra, '$.role')
        if speaker:
            conditions.append(
                "(ce.tags LIKE ? OR json_extract(ce.extra, '$.role') = ?)"
            )
            params.append(f"%speaker:{speaker}%")
            params.append(speaker)

        where = " AND ".join(conditions)

        sql = f"""
        SELECT ce.rowid, ce.id, ce.context_type, ce.memory_category,
               ce.l0_abstract, ce.l1_overview, ce.l2_full, ce.fts_text,
               ce.scope, ce.owner_id, ce.tags, ce.confidence,
               ce.source, ce.uri, ce.status,
               ce.valid_from, ce.valid_until, ce.superseded_by,
               ce.origin_type, ce.origin_id,
               ce.resource_url, ce.resource_format, ce.resource_size,
               ce.skill_tool_name, ce.skill_api_spec,
               ce.embedding_hash, ce.embedding_source_level,
               ce.embedding_model, ce.embedding_dimensions,
               ce.created_at, ce.updated_at, ce.deleted_at, ce.extra,
               bm25(context_fts) AS rank
        FROM context_fts fts
        JOIN context_entries ce ON ce.rowid = fts.rowid
        WHERE context_fts MATCH ?
          AND {where}
        ORDER BY rank
        LIMIT ?
        """
        params.append(top_k)

        try:
            async with self._pool.get_read_conn() as conn:
                cursor = await conn.execute(sql, params)
                rows = await cursor.fetchall()
        except aiosqlite.OperationalError as e:
            error_msg = str(e)
            logger.warning(f"FTS5 search failed: {e}, query was: {normalized!r}")

            # Graceful degradation: retry with a fully sanitized query
            # that removes all non-alphanumeric/CJK characters.
            # This handles edge cases the tokenizer may have missed.
            if "no such column" in error_msg.lower() or "syntax error" in error_msg.lower():
                sanitized_tokens = []
                for t in normalized.split():
                    # Strip all non-alphanumeric, non-CJK, non-OR characters
                    clean = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", "", t)
                    if clean and clean.upper() not in ("AND", "OR", "NOT", "NEAR"):
                        sanitized_tokens.append(clean)
                if sanitized_tokens:
                    sanitized_query = " OR ".join(sanitized_tokens)
                    logger.info(f"FTS5 retry with sanitized query: {sanitized_query!r}")
                    try:
                        # Rebuild params with sanitized match expression
                        retry_params = [sanitized_query] + params[1:]
                        async with self._pool.get_read_conn() as conn:
                            cursor = await conn.execute(sql, retry_params)
                            rows = await cursor.fetchall()
                        # Retry succeeded — fall through to result processing
                    except aiosqlite.OperationalError as e2:
                        logger.warning(f"FTS5 retry also failed: {e2}")
                        raise IndexOpError(f"FTS5 search failed (even after retry): {e2}") from e2
                else:
                    raise IndexOpError(f"FTS5 search failed: {e}") from e
            else:
                raise IndexOpError(f"FTS5 search failed: {e}") from e

        results = []
        for row in rows:
            # row[-1] is the BM25 rank score (negative, more negative = better match)
            rank_score = row[-1]
            entry = self._row_to_entry(row[:-1])
            # Convert BM25 score: negate it so higher = better match
            results.append(
                SearchResult(
                    entry=entry,
                    score=-rank_score if rank_score else 0.0,
                    score_breakdown={"fts": -rank_score if rank_score else 0.0},
                    ranking_stage="fts",
                    matched_by="fts",
                )
            )

        return results

    async def remove(self, entry_id: str) -> None:
        """FTS5 removal is handled by triggers on DELETE from context_entries."""
        pass

    async def count(self) -> int:
        """Count entries in the FTS index."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM context_fts")
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def rebuild(self) -> int:
        """Rebuild the FTS5 index from scratch.

        Drops and recreates the FTS5 table, then repopulates from context_entries.
        Returns the number of entries reindexed.
        """
        async with self._pool.get_write_conn() as conn:
            # Drop and recreate
            await conn.execute("DROP TABLE IF EXISTS context_fts")
            await conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(
                    l0_abstract, l1_overview, fts_text, tags,
                    content='context_entries', content_rowid='rowid',
                    tokenize='unicode61'
                )"""
            )
            # Rebuild from content table
            await conn.execute("INSERT INTO context_fts(context_fts) VALUES('rebuild')")
            await conn.commit()

        count = await self.count()
        logger.info(f"FTS5 index rebuilt: {count} entries")
        return count

    @staticmethod
    def _row_to_entry(row: tuple) -> ContextEntry:
        """Convert a FTS join result row to ContextEntry (simplified)."""
        from ..store.sqlite_store import _ENTRY_COLUMNS

        col_names = [
            "rowid", "id", "context_type", "memory_category",
            "l0_abstract", "l1_overview", "l2_full", "fts_text",
            "scope", "owner_id", "tags", "confidence",
            "source", "uri", "status",
            "valid_from", "valid_until", "superseded_by",
            "origin_type", "origin_id",
            "resource_url", "resource_format", "resource_size",
            "skill_tool_name", "skill_api_spec",
            "embedding_hash", "embedding_source_level",
            "embedding_model", "embedding_dimensions",
            "created_at", "updated_at", "deleted_at", "extra",
        ]
        col_map = dict(zip(col_names[1:], row[1:]))  # skip rowid
        import json
        from datetime import datetime, timezone

        return ContextEntry(
            id=col_map.get("id", ""),
            context_type=col_map.get("context_type", "memory"),
            memory_category=col_map.get("memory_category"),
            status=col_map.get("status", "active"),
            origin_type=col_map.get("origin_type", "manual"),
            origin_id=col_map.get("origin_id"),
            l0_abstract=col_map.get("l0_abstract", "") or "",
            l1_overview=col_map.get("l1_overview", "") or "",
            l2_full=col_map.get("l2_full", "") or "",
            fts_text=col_map.get("fts_text", "") or "",
            scope=col_map.get("scope", "global"),
            owner_id=col_map.get("owner_id"),
            tags=json.loads(col_map.get("tags", "[]") or "[]"),
            confidence=col_map.get("confidence", 1.0),
            source=col_map.get("source", "unknown") or "unknown",
            uri=col_map.get("uri", "") or "",
            valid_from=col_map.get("valid_from", datetime.now(timezone.utc).isoformat()),
            valid_until=col_map.get("valid_until"),
            superseded_by=col_map.get("superseded_by"),
            resource_url=col_map.get("resource_url"),
            resource_format=col_map.get("resource_format"),
            resource_size=col_map.get("resource_size"),
            skill_tool_name=col_map.get("skill_tool_name"),
            skill_api_spec=json.loads(col_map["skill_api_spec"]) if col_map.get("skill_api_spec") else None,
            embedding_hash=col_map.get("embedding_hash", "") or "",
            embedding_source_level=col_map.get("embedding_source_level", "l1") or "l1",
            embedding_model=col_map.get("embedding_model"),
            embedding_dimensions=col_map.get("embedding_dimensions"),
            created_at=col_map.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=col_map.get("updated_at", datetime.now(timezone.utc).isoformat()),
            deleted_at=col_map.get("deleted_at"),
            extra=json.loads(col_map.get("extra", "{}") or "{}"),
        )
