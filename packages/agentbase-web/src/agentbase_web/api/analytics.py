"""API: Timeline, heatmap, recent activity, and health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..app import get_db

router = APIRouter()


@router.get("/timeline")
async def timeline(
    granularity: str = Query("day", pattern="^(day|week|month)$"),
    days: int = Query(90, ge=7, le=365),
) -> dict:
    """Get memory creation timeline data — list of {date, count}."""
    db = get_db()
    pool = db._engine._pool

    if granularity == "day":
        group_expr = "date(created_at)"
    elif granularity == "week":
        group_expr = "strftime('%Y-W%W', created_at)"
    else:
        group_expr = "strftime('%Y-%m', created_at)"

    async with pool.get_read_conn() as conn:
        cursor = await conn.execute(
            f"""SELECT {group_expr} as period, COUNT(*)
                FROM context_entries
                WHERE status='active' AND created_at >= datetime('now', ?)
                GROUP BY period ORDER BY period""",
            (f"-{days} days",),
        )
        rows = await cursor.fetchall()

    points = [{"date": r[0], "count": r[1]} for r in rows]
    return {"granularity": granularity, "days": days, "points": points}


@router.get("/heatmap")
async def heatmap() -> dict:
    """Get memory creation heatmap data — GitHub contribution style.

    Returns {cells: [{date, count, dow}], weeks: int, max_count: int}
    where dow = day-of-week (0=Mon … 6=Sun).
    """
    db = get_db()
    pool = db._engine._pool

    async with pool.get_read_conn() as conn:
        cursor = await conn.execute(
            """SELECT date(created_at) as d, COUNT(*) as c
               FROM context_entries
               WHERE status='active' AND created_at >= datetime('now', '-365 days')
               GROUP BY d ORDER BY d"""
        )
        rows = await cursor.fetchall()

    count_map = {r[0]: r[1] for r in rows}
    max_count = max(count_map.values()) if count_map else 1

    import datetime
    today = datetime.date.today()
    cells = []
    for i in range(364, -1, -1):
        d = today - datetime.timedelta(days=i)
        ds = d.isoformat()
        cells.append({
            "date": ds,
            "count": count_map.get(ds, 0),
            "dow": d.weekday(),  # 0=Mon
        })

    return {"cells": cells, "weeks": 52, "max_count": max_count}


@router.get("/recent-activity")
async def recent_activity(
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Get recent activity feed — latest created/updated entries."""
    db = get_db()
    pool = db._engine._pool

    async with pool.get_read_conn() as conn:
        cursor = await conn.execute(
            """SELECT id, context_type, memory_category, scope, l0_abstract,
                      confidence, created_at, updated_at
               FROM context_entries
               WHERE status='active'
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

    items = []
    for r in rows:
        items.append({
            "id": r[0],
            "type": r[1],
            "category": r[2],
            "scope": r[3],
            "summary": (r[4] or "")[:100],
            "confidence": r[5],
            "created_at": r[6],
            "updated_at": r[7],
        })

    return {"activities": items}


@router.get("/health")
async def health() -> dict:
    """Get memory health metrics — confidence, freshness, coverage."""
    db = get_db()
    pool = db._engine._pool

    result: dict = {}

    async with pool.get_read_conn() as conn:
        # Average confidence
        cursor = await conn.execute(
            "SELECT AVG(confidence) FROM context_entries WHERE status='active'"
        )
        row = await cursor.fetchone()
        result["avg_confidence"] = round(row[0], 3) if row and row[0] else 0

        # Confidence distribution
        cursor = await conn.execute(
            """SELECT
                CASE
                    WHEN confidence >= 0.9 THEN 'high'
                    WHEN confidence >= 0.6 THEN 'medium'
                    ELSE 'low'
                END as level, COUNT(*)
               FROM context_entries WHERE status='active'
               GROUP BY level"""
        )
        rows = await cursor.fetchall()
        result["confidence_levels"] = {r[0]: r[1] for r in rows}

        # Freshness — entries by age bucket
        cursor = await conn.execute(
            """SELECT
                CASE
                    WHEN created_at >= datetime('now', '-1 day') THEN '1d'
                    WHEN created_at >= datetime('now', '-7 days') THEN '7d'
                    WHEN created_at >= datetime('now', '-30 days') THEN '30d'
                    WHEN created_at >= datetime('now', '-90 days') THEN '90d'
                    ELSE 'older'
                END as age_bucket, COUNT(*)
               FROM context_entries WHERE status='active'
               GROUP BY age_bucket ORDER BY age_bucket"""
        )
        rows = await cursor.fetchall()
        result["freshness"] = [{"bucket": r[0], "count": r[1]} for r in rows]

        # Coverage — how many unique tags
        cursor = await conn.execute(
            "SELECT COUNT(DISTINCT tag) FROM context_tags"
        )
        row = await cursor.fetchone()
        result["unique_tags"] = row[0] if row else 0

        # Top tags
        cursor = await conn.execute(
            "SELECT tag, COUNT(*) as c FROM context_tags GROUP BY tag ORDER BY c DESC LIMIT 20"
        )
        rows = await cursor.fetchall()
        result["top_tags"] = [{"tag": r[0], "count": r[1]} for r in rows]

        # Memory type coverage
        cursor = await conn.execute(
            """SELECT context_type, memory_category, COUNT(*)
               FROM context_entries WHERE status='active'
               GROUP BY context_type, memory_category
               ORDER BY context_type, memory_category"""
        )
        rows = await cursor.fetchall()
        coverage = []
        for r in rows:
            coverage.append({
                "type": r[0],
                "category": r[1] or "none",
                "count": r[2],
            })
        result["type_coverage"] = coverage

    return result
