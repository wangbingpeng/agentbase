"""API: Knowledge graph endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..app import get_db

router = APIRouter()


@router.get("/entities")
async def list_entities(
    name: str = Query("", description="Search by name prefix"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """List entities."""
    db = get_db()
    if name:
        entities = await db.find_entities(name)
    else:
        # Get all entities via direct SQL
        pool = db._engine._pool
        async with pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT id, name, entity_type, description FROM entities ORDER BY name LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return {"entities": [{"id": r[0], "name": r[1], "entity_type": r[2], "description": r[3] or ""} for r in rows]}

    return {
        "entities": [
            {"id": e.id, "name": e.name, "entity_type": e.entity_type, "description": e.description or ""}
            for e in entities
        ]
    }


@router.get("/graph-data")
async def graph_data() -> dict:
    """Get full knowledge graph data (nodes + edges) for visualization."""
    db = get_db()
    pool = db._engine._pool

    nodes = []
    edges = []

    async with pool.get_read_conn() as conn:
        # Fetch all entities
        cursor = await conn.execute(
            "SELECT id, name, entity_type, description FROM entities ORDER BY name"
        )
        entity_rows = await cursor.fetchall()
        for r in entity_rows:
            nodes.append({
                "id": r[0],
                "name": r[1],
                "entity_type": r[2],
                "description": r[3] or "",
            })

        # Fetch all active relations
        cursor = await conn.execute(
            """SELECT r.id, r.source_id, e1.name, r.predicate, r.target_id, e2.name, r.confidence
               FROM relations r
               JOIN entities e1 ON e1.id = r.source_id
               JOIN entities e2 ON e2.id = r.target_id
               WHERE r.valid_until IS NULL
               ORDER BY r.confidence DESC"""
        )
        rel_rows = await cursor.fetchall()
        for r in rel_rows:
            edges.append({
                "id": r[0],
                "source": r[1],
                "source_name": r[2],
                "predicate": r[3],
                "target": r[4],
                "target_name": r[5],
                "confidence": r[6],
            })

    return {"nodes": nodes, "edges": edges}
