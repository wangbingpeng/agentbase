"""API: Context entries endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from agentbase import ContextScope, ContextType

from ..app import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/entries")
async def list_entries(
    scope: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List context entries with pagination."""
    db = get_db()
    sc = ContextScope(scope) if scope else None
    ct = ContextType(type) if type else None

    try:
        entries = await db.list_entries(scope=sc, context_type=ct, limit=limit, offset=offset)
        total = await db.count(scope=sc, context_type=ct)
    except Exception as e:
        logger.error(f"Failed to list entries: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Database query failed: {e}. Try re-initializing the database with 'agentbase init'."},
        )

    items = []
    for e in entries:
        items.append({
            "id": e.id,
            "type": e.context_type.value,
            "category": e.memory_category.value if e.memory_category else None,
            "scope": e.scope.value,
            "status": e.status.value,
            "confidence": e.confidence,
            "tags": e.tags,
            "l0": e.l0_abstract,
            "l1": e.l1_overview,
            "l2": (e.l2_full or "")[:300],
            "source": e.source,
            "created_at": e.created_at.isoformat() if hasattr(e.created_at, "isoformat") else str(e.created_at),
        })

    return {"entries": items, "total": total, "limit": limit, "offset": offset}


@router.get("/entries/{entry_id}")
async def get_entry(entry_id: str) -> dict:
    """Get a single context entry by ID."""
    db = get_db()
    entry = await db.get(entry_id)
    if entry is None:
        return {"error": "Not found"}
    return {
        "id": entry.id,
        "type": entry.context_type.value,
        "category": entry.memory_category.value if entry.memory_category else None,
        "scope": entry.scope.value,
        "status": entry.status.value,
        "confidence": entry.confidence,
        "tags": entry.tags,
        "l0": entry.l0_abstract,
        "l1": entry.l1_overview,
        "l2": entry.l2_full,
        "source": entry.source,
        "uri": entry.uri,
        "origin_type": entry.origin_type.value if entry.origin_type else None,
        "owner_id": entry.owner_id,
        "created_at": entry.created_at.isoformat() if hasattr(entry.created_at, "isoformat") else str(entry.created_at),
        "updated_at": entry.updated_at.isoformat() if hasattr(entry.updated_at, "isoformat") else str(entry.updated_at),
    }


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    scope: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
) -> dict:
    """Search context entries."""
    db = get_db()
    sc = ContextScope(scope) if scope else None
    ct = ContextType(type) if type else None

    try:
        results = await db.find(query=q, top_k=top_k, context_type=ct, scope=sc)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {e}. Try re-initializing the database."},
        )

    items = []
    for r in results:
        items.append({
            "id": r.entry.id,
            "type": r.entry.context_type.value,
            "category": r.entry.memory_category.value if r.entry.memory_category else None,
            "scope": r.entry.scope.value,
            "score": round(r.score, 4),
            "confidence": r.entry.confidence,
            "tags": r.entry.tags,
            "l0": r.entry.l0_abstract,
            "l1": r.entry.l1_overview,
            "l2": (r.entry.l2_full or "")[:500],
        })

    return {"results": items, "query": q}


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str) -> dict:
    """Soft-delete a context entry."""
    db = get_db()
    ok = await db.delete(entry_id)
    return {"deleted": ok}
