"""EntityService — entity/relation CRUD, alias, and temporal graph operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from ..models.context_entry import _new_ulid, _utcnow
from ..models.entity import Entity, FactTimeline, Relation
from ..store.connection import ConnectionPool

logger = logging.getLogger(__name__)


class EntityService:
    """Entity and relation management service with alias-based disambiguation."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    async def add_entity(self, entity: Entity) -> Entity:
        """Add or merge an entity."""
        # Check for existing entity by (name, entity_type)
        existing = await self._find_by_name_type(entity.name, entity.entity_type)
        if existing:
            # Merge: update last_seen, increment fact_count
            existing.last_seen = _utcnow()
            existing.fact_count += 1
            existing.updated_at = _utcnow()
            if entity.description and not existing.description:
                existing.description = entity.description
            await self._update_entity(existing)
            return existing

        # Insert new entity
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type, description, first_seen, last_seen, fact_count, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.id, entity.name, entity.entity_type, entity.description,
                    entity.first_seen.isoformat(), entity.last_seen.isoformat(),
                    entity.fact_count, json.dumps(entity.properties),
                    entity.created_at.isoformat(), entity.updated_at.isoformat(),
                ),
            )
            await conn.commit()
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_entity(row)

    async def find_entities(self, name: str, entity_type: str | None = None) -> list[Entity]:
        """Find entities by name (and optionally type)."""
        if entity_type:
            sql = "SELECT * FROM entities WHERE name = ? AND entity_type = ?"
            params = (name, entity_type)
        else:
            sql = "SELECT * FROM entities WHERE name = ?"
            params = (name,)

        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_entity(r) for r in rows]

    async def add_alias(self, entity_id: str, alias: str) -> None:
        """Add an alias for entity disambiguation."""
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                (entity_id, alias),
            )
            await conn.commit()

    async def get_aliases(self, entity_id: str) -> list[str]:
        """Get aliases for an entity."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity_id,)
            )
            rows = await cursor.fetchall()
            return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    async def add_relation(self, relation: Relation) -> Relation:
        """Add a relation between entities."""
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO relations (id, source_id, target_id, predicate, valid_from, valid_until, confidence, evidence_ids, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    relation.id, relation.source_id, relation.target_id, relation.predicate,
                    relation.valid_from.isoformat(),
                    relation.valid_until.isoformat() if relation.valid_until else None,
                    relation.confidence,
                    json.dumps(relation.evidence_ids),
                    json.dumps(relation.properties),
                    relation.created_at.isoformat(), relation.updated_at.isoformat(),
                ),
            )
            await conn.commit()
        return relation

    async def get_current_relations(self, entity_id: str) -> list[Relation]:
        """Get current (valid) relations for an entity."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT * FROM relations WHERE (source_id = ? OR target_id = ?) AND valid_until IS NULL ORDER BY confidence DESC",
                (entity_id, entity_id),
            )
            rows = await cursor.fetchall()
            return [self._row_to_relation(r) for r in rows]

    async def get_relation_history(self, entity_id: str) -> list[Relation]:
        """Get all relations (including historical) for an entity."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT * FROM relations WHERE source_id = ? OR target_id = ? ORDER BY valid_from DESC",
                (entity_id, entity_id),
            )
            rows = await cursor.fetchall()
            return [self._row_to_relation(r) for r in rows]

    # ------------------------------------------------------------------
    # Graph traversal (recursive CTE)
    # ------------------------------------------------------------------

    async def graph_traversal(self, entity_name: str, depth: int = 2) -> list[dict]:
        """Traverse the knowledge graph from an entity using recursive CTE."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                """
                WITH RECURSIVE graph_walk(id, name, entity_type, depth, path) AS (
                    SELECT e.id, e.name, e.entity_type, 0, '/' || e.id || '/'
                    FROM entities e
                    WHERE e.name = ?

                    UNION ALL

                    SELECT
                        CASE WHEN r.source_id = gw.id THEN r.target_id ELSE r.source_id END,
                        CASE WHEN r.source_id = gw.id THEN e2.name ELSE e1.name END,
                        CASE WHEN r.source_id = gw.id THEN e2.entity_type ELSE e1.entity_type END,
                        gw.depth + 1,
                        gw.path || (CASE WHEN r.source_id = gw.id THEN r.target_id ELSE r.source_id END) || '/'
                    FROM graph_walk gw
                    JOIN relations r ON (
                        (r.source_id = gw.id OR r.target_id = gw.id)
                        AND r.valid_until IS NULL
                    )
                    JOIN entities e1 ON e1.id = r.source_id
                    JOIN entities e2 ON e2.id = r.target_id
                    WHERE gw.depth < ?
                      AND gw.path NOT LIKE '%/' || (CASE WHEN r.source_id = gw.id THEN r.target_id ELSE r.source_id END) || '/%'
                )
                SELECT DISTINCT id, name, entity_type, depth, path
                FROM graph_walk
                WHERE depth > 0
                ORDER BY depth, name
                """,
                (entity_name, depth),
            )
            rows = await cursor.fetchall()
            return [
                {"id": r[0], "name": r[1], "entity_type": r[2], "depth": r[3], "path": r[4]}
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Fact timeline
    # ------------------------------------------------------------------

    async def add_fact(self, fact: FactTimeline) -> None:
        """Add a fact to the timeline."""
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO fact_timeline (id, entity_id, fact, valid_at, superseded_by, action, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.id, fact.entity_id, fact.fact, fact.valid_at.isoformat(),
                    fact.superseded_by, fact.action, fact.created_at.isoformat(),
                ),
            )
            await conn.commit()

    async def get_current_facts(self, entity_id: str) -> list[FactTimeline]:
        """Get current (non-superseded) facts for an entity."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT * FROM fact_timeline WHERE entity_id = ? AND superseded_by IS NULL ORDER BY valid_at DESC",
                (entity_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_fact(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_by_name_type(self, name: str, entity_type: str) -> Entity | None:
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT * FROM entities WHERE name = ? AND entity_type = ? LIMIT 1",
                (name, entity_type),
            )
            row = await cursor.fetchone()
            return self._row_to_entity(row) if row else None

    async def _update_entity(self, entity: Entity) -> None:
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE entities SET last_seen = ?, fact_count = ?, description = ?, properties = ?, updated_at = ? WHERE id = ?",
                (
                    entity.last_seen.isoformat(), entity.fact_count, entity.description,
                    json.dumps(entity.properties), entity.updated_at.isoformat(), entity.id,
                ),
            )
            await conn.commit()

    @staticmethod
    def _row_to_entity(row: tuple) -> Entity:
        return Entity(
            id=row[0], name=row[1], entity_type=row[2], description=row[3] or "",
            first_seen=datetime.fromisoformat(row[4]),
            last_seen=datetime.fromisoformat(row[5]),
            fact_count=row[6], properties=json.loads(row[7]) if row[7] else {},
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
        )

    @staticmethod
    def _row_to_relation(row: tuple) -> Relation:
        return Relation(
            id=row[0], source_id=row[1], target_id=row[2], predicate=row[3],
            valid_from=datetime.fromisoformat(row[4]),
            valid_until=datetime.fromisoformat(row[5]) if row[5] else None,
            confidence=row[6],
            evidence_ids=json.loads(row[7]) if row[7] else [],
            properties=json.loads(row[8]) if row[8] else {},
            created_at=datetime.fromisoformat(row[9]),
            updated_at=datetime.fromisoformat(row[10]),
        )

    @staticmethod
    def _row_to_fact(row: tuple) -> FactTimeline:
        return FactTimeline(
            id=row[0], entity_id=row[1], fact=row[2],
            valid_at=datetime.fromisoformat(row[3]),
            superseded_by=row[4], action=row[5] or "created",
            created_at=datetime.fromisoformat(row[6]),
        )
