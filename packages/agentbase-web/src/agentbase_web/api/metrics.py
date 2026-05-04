"""API: Metrics & distribution endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..app import get_db

router = APIRouter()


@router.get("/metrics")
async def metrics() -> dict:
    """Get context quality metrics."""
    db = get_db()
    return await db.get_metrics()


@router.get("/distribution")
async def distribution() -> dict:
    """Get distribution statistics for visualization."""
    db = get_db()
    pool = db._engine._pool
    result: dict = {}

    async with pool.get_read_conn() as conn:
        # Type distribution
        cursor = await conn.execute(
            "SELECT context_type, COUNT(*) FROM context_entries WHERE status='active' GROUP BY context_type"
        )
        rows = await cursor.fetchall()
        result["by_type"] = [{"name": r[0], "count": r[1]} for r in rows]

        # Scope distribution
        cursor = await conn.execute(
            "SELECT scope, COUNT(*) FROM context_entries WHERE status='active' GROUP BY scope"
        )
        rows = await cursor.fetchall()
        result["by_scope"] = [{"name": r[0], "count": r[1]} for r in rows]

        # Memory category distribution (only for type=memory)
        cursor = await conn.execute(
            "SELECT memory_category, COUNT(*) FROM context_entries WHERE status='active' AND context_type='memory' GROUP BY memory_category"
        )
        rows = await cursor.fetchall()
        result["by_category"] = [{"name": r[0] or "uncategorized", "count": r[1]} for r in rows]

        # Confidence distribution (bucketed into 0.2 intervals)
        cursor = await conn.execute(
            """SELECT
                CASE
                    WHEN confidence >= 0.9 THEN '0.9-1.0'
                    WHEN confidence >= 0.7 THEN '0.7-0.9'
                    WHEN confidence >= 0.5 THEN '0.5-0.7'
                    WHEN confidence >= 0.3 THEN '0.3-0.5'
                    ELSE '0.0-0.3'
                END as bucket,
                COUNT(*)
            FROM context_entries WHERE status='active'
            GROUP BY bucket ORDER BY bucket"""
        )
        rows = await cursor.fetchall()
        result["by_confidence"] = [{"name": r[0], "count": r[1]} for r in rows]

        # Entity count
        cursor = await conn.execute("SELECT COUNT(*) FROM entities")
        row = await cursor.fetchone()
        result["entity_count"] = row[0] if row else 0

        # Session count
        cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
        row = await cursor.fetchone()
        result["session_count"] = row[0] if row else 0

        # Recent entries (last 7 days)
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM context_entries WHERE status='active' AND created_at >= datetime('now', '-7 days')"
        )
        row = await cursor.fetchone()
        result["recent_7d"] = row[0] if row else 0

    return result
