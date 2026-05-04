"""SQLiteStore — primary storage backend for AgentBase."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from ..exceptions import StorageError, ValidationError
from ..models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    OriginType,
)
from .connection import ConnectionPool
from .migrator import Migrator

logger = logging.getLogger(__name__)

# Column list for INSERT/SELECT (order matters)
_ENTRY_COLUMNS = [
    "id", "context_type", "memory_category", "status", "origin_type", "origin_id",
    "l0_abstract", "l1_overview", "l2_full", "fts_text",
    "embedding_hash", "embedding_source_level", "embedding_model", "embedding_dimensions",
    "scope", "owner_id", "tags", "confidence", "source", "uri",
    "resource_url", "resource_format", "resource_size",
    "skill_tool_name", "skill_api_spec",
    "valid_from", "valid_until", "superseded_by",
    "created_at", "updated_at", "deleted_at", "extra",
]

_INSERT_SQL = f"""
INSERT INTO context_entries ({', '.join(_ENTRY_COLUMNS)})
VALUES ({', '.join('?' * len(_ENTRY_COLUMNS))})
"""

_SELECT_SQL = f"SELECT {', '.join(_ENTRY_COLUMNS)} FROM context_entries"


def _entry_to_row(entry: ContextEntry) -> tuple:
    """Convert a ContextEntry model to a database row tuple."""
    return (
        entry.id,
        entry.context_type.value,
        entry.memory_category.value if entry.memory_category else None,
        entry.status.value,
        entry.origin_type.value,
        entry.origin_id,
        entry.l0_abstract,
        entry.l1_overview,
        entry.l2_full,
        entry.fts_text,
        entry.embedding_hash,
        entry.embedding_source_level,
        entry.embedding_model,
        entry.embedding_dimensions,
        entry.scope.value,
        entry.owner_id,
        json.dumps(entry.tags),
        entry.confidence,
        entry.source,
        entry.uri,
        entry.resource_url,
        entry.resource_format,
        entry.resource_size,
        entry.skill_tool_name,
        json.dumps(entry.skill_api_spec) if entry.skill_api_spec else None,
        entry.valid_from.isoformat(),
        entry.valid_until.isoformat() if entry.valid_until else None,
        entry.superseded_by,
        entry.created_at.isoformat(),
        entry.updated_at.isoformat(),
        entry.deleted_at.isoformat() if entry.deleted_at else None,
        json.dumps(entry.extra),
    )


def _row_to_entry(row: tuple) -> ContextEntry:
    """Convert a database row to a ContextEntry model."""
    col_map = dict(zip(_ENTRY_COLUMNS, row))
    return ContextEntry(
        id=col_map["id"],
        context_type=ContextType(col_map["context_type"]),
        memory_category=MemoryCategory(col_map["memory_category"]) if col_map["memory_category"] else None,
        status=EntryStatus(col_map["status"]),
        origin_type=OriginType(col_map["origin_type"]),
        origin_id=col_map["origin_id"],
        l0_abstract=col_map["l0_abstract"] or "",
        l1_overview=col_map["l1_overview"] or "",
        l2_full=col_map["l2_full"] or "",
        fts_text=col_map.get("fts_text") or "",
        embedding_hash=col_map["embedding_hash"] or "",
        embedding_source_level=col_map["embedding_source_level"] or "l1",
        embedding_model=col_map["embedding_model"],
        embedding_dimensions=col_map["embedding_dimensions"],
        scope=ContextScope(col_map["scope"]),
        owner_id=col_map["owner_id"],
        tags=json.loads(col_map["tags"]) if col_map["tags"] else [],
        confidence=col_map["confidence"],
        source=col_map["source"] or "unknown",
        uri=col_map["uri"] or "",
        resource_url=col_map["resource_url"],
        resource_format=col_map["resource_format"],
        resource_size=col_map["resource_size"],
        skill_tool_name=col_map["skill_tool_name"],
        skill_api_spec=json.loads(col_map["skill_api_spec"]) if col_map["skill_api_spec"] else None,
        valid_from=datetime.fromisoformat(col_map["valid_from"]) if col_map["valid_from"] else datetime.now(timezone.utc),
        valid_until=datetime.fromisoformat(col_map["valid_until"]) if col_map["valid_until"] else None,
        superseded_by=col_map["superseded_by"],
        created_at=datetime.fromisoformat(col_map["created_at"]) if col_map["created_at"] else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(col_map["updated_at"]) if col_map["updated_at"] else datetime.now(timezone.utc),
        deleted_at=datetime.fromisoformat(col_map["deleted_at"]) if col_map["deleted_at"] else None,
        extra=json.loads(col_map["extra"]) if col_map["extra"] else {},
    )


class SQLiteStore:
    """Primary storage backend using SQLite."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        """Run schema migrations."""
        async with self._pool.get_write_conn() as conn:
            migrator = Migrator(conn)
            await migrator.migrate()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def add(self, entry: ContextEntry) -> ContextEntry:
        """Insert a new context entry (within a transaction)."""
        # Validate scope/owner_id constraints
        self._validate_scope_owner(entry)

        # Auto-generate URI if not set
        if not entry.uri:
            entry.uri = entry.generate_uri()

        async with self._pool.get_write_conn() as conn:
            try:
                await conn.execute(_INSERT_SQL, _entry_to_row(entry))

                # Sync tags to context_tags helper table
                await self._sync_tags(conn, entry.id, entry.tags)

                # Sync entity links if entry has origin in graph
                await self._sync_entity_links(conn, entry)

                await conn.commit()
            except aiosqlite.IntegrityError as e:
                raise StorageError(f"Failed to insert entry {entry.id}: {e}") from e

        return entry

    async def get(self, entry_id: str) -> ContextEntry | None:
        """Get a single entry by ID."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                f"{_SELECT_SQL} WHERE id = ?", (entry_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_entry(row)

    async def update(self, entry: ContextEntry) -> None:
        """Update an existing entry."""
        entry.updated_at = datetime.now(timezone.utc)

        async with self._pool.get_write_conn() as conn:
            sets = ", ".join(f"{col} = ?" for col in _ENTRY_COLUMNS[1:])
            values = _entry_to_row(entry)[1:] + (entry.id,)
            await conn.execute(
                f"UPDATE context_entries SET {sets} WHERE id = ?", values
            )
            # Sync tags
            await self._sync_tags(conn, entry.id, entry.tags)
            # Sync entity links
            await self._sync_entity_links(conn, entry)
            await conn.commit()

    async def delete(self, entry_id: str) -> bool:
        """Soft delete an entry (set status=deleted + deleted_at)."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.get_write_conn() as conn:
            cursor = await conn.execute(
                "UPDATE context_entries SET status = 'deleted', deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, entry_id),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def purge(self, entry_id: str) -> bool:
        """Hard delete an entry (physical removal from all tables)."""
        async with self._pool.get_write_conn() as conn:
            # Tags and entity links cascade
            await conn.execute("DELETE FROM context_entries WHERE id = ?", (entry_id,))
            await conn.commit()
            return True

    async def supersede(self, old_id: str, new_entry: ContextEntry) -> None:
        """Supersede an old entry with a new one."""
        now = datetime.now(timezone.utc)
        async with self._pool.get_write_conn() as conn:
            # Mark old entry as superseded
            await conn.execute(
                "UPDATE context_entries SET status = 'superseded', superseded_by = ?, valid_until = ?, updated_at = ? WHERE id = ?",
                (new_entry.id, now.isoformat(), now.isoformat(), old_id),
            )
            # Insert new entry
            await conn.execute(_INSERT_SQL, _entry_to_row(new_entry))
            await self._sync_tags(conn, new_entry.id, new_entry.tags)
            await conn.commit()

    # ------------------------------------------------------------------
    # List / count operations
    # ------------------------------------------------------------------

    async def list_entries(
        self,
        scope: ContextScope | None = None,
        owner_id: str | None = None,
        context_type: ContextType | None = None,
        status: EntryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextEntry]:
        """List entries with optional filters."""
        conditions = []
        params: list[Any] = []

        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope.value)
        if owner_id is not None:
            conditions.append("owner_id = ?")
            params.append(owner_id)
        if context_type is not None:
            conditions.append("context_type = ?")
            params.append(context_type.value)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        else:
            # Default: only active entries
            conditions.append("status = 'active'")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"{_SELECT_SQL} WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [_row_to_entry(row) for row in rows]

    async def count(
        self,
        scope: ContextScope | None = None,
        context_type: ContextType | None = None,
        status: EntryStatus | None = None,
    ) -> int:
        """Count entries with optional filters."""
        conditions = []
        params: list[Any] = []

        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope.value)
        if context_type is not None:
            conditions.append("context_type = ?")
            params.append(context_type.value)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        else:
            conditions.append("status = 'active'")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT COUNT(*) FROM context_entries WHERE {where}"

        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Scope-aware query (visible to a given agent)
    # ------------------------------------------------------------------

    async def list_visible(
        self,
        agent_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextEntry]:
        """List entries visible to a given agent (global + matching scopes)."""
        conditions = ["status = 'active'"]
        params: list[Any] = []

        # Global is always visible
        scope_parts = ["scope = 'global'"]

        if project_id:
            scope_parts.append("(scope = 'project' AND owner_id = ?)")
            params.append(project_id)
        if agent_id:
            scope_parts.append("(scope = 'agent' AND owner_id = ?)")
            params.append(agent_id)
        if session_id:
            scope_parts.append("(scope = 'session' AND owner_id = ?)")
            params.append(session_id)

        conditions.append(f"({' OR '.join(scope_parts)})")
        where = " AND ".join(conditions)

        sql = f"{_SELECT_SQL} WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [_row_to_entry(row) for row in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_scope_owner(entry: ContextEntry) -> None:
        """Validate scope/owner_id constraints per SPEC §4.7."""
        if entry.scope == ContextScope.GLOBAL and entry.owner_id is not None:
            raise ValidationError(
                f"scope=global requires owner_id=NULL, got owner_id={entry.owner_id}"
            )
        if entry.scope != ContextScope.GLOBAL and entry.owner_id is None:
            raise ValidationError(
                f"scope={entry.scope.value} requires owner_id to be set"
            )

    @staticmethod
    async def _sync_tags(
        conn: aiosqlite.Connection, entry_id: str, tags: list[str]
    ) -> None:
        """Sync tags to the context_tags helper table."""
        await conn.execute("DELETE FROM context_tags WHERE context_id = ?", (entry_id,))
        for tag in tags:
            await conn.execute(
                "INSERT OR IGNORE INTO context_tags (context_id, tag) VALUES (?, ?)",
                (entry_id, tag),
            )

    @staticmethod
    async def _sync_entity_links(
        conn: aiosqlite.Connection, entry: ContextEntry
    ) -> None:
        """Sync context_entity_links per SPEC §19.12.

        When an entry has entity-related metadata (origin_id references
        an entity, or tags match entity aliases), create links.
        """
        entry_id = entry.id

        # Link by origin_id if origin is graph-related (EXTRACTED from graph)
        if entry.origin_id and entry.origin_type == OriginType.EXTRACTED:
            await conn.execute(
                "INSERT OR IGNORE INTO context_entity_links (context_id, entity_id) VALUES (?, ?)",
                (entry_id, entry.origin_id),
            )

        # Link by tags that match entity aliases
        for tag in entry.tags:
            cursor = await conn.execute(
                "SELECT entity_id FROM entity_aliases WHERE alias = ?",
                (tag,),
            )
            row = await cursor.fetchone()
            if row:
                await conn.execute(
                    "INSERT OR IGNORE INTO context_entity_links (context_id, entity_id) VALUES (?, ?)",
                    (entry_id, row[0]),
                )
