"""API: Session endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..app import get_db

router = APIRouter()


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """List sessions."""
    db = get_db()
    pool = db._engine._pool

    async with pool.get_read_conn() as conn:
        # Get total count
        cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        # Get sessions with message count
        cursor = await conn.execute(
            """SELECT s.id, s.agent_id, s.project, s.status,
                      s.archived_message_count, s.total_tokens_used,
                      s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM session_messages WHERE session_id = s.id) as msg_count
               FROM sessions s
               ORDER BY s.created_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        )
        rows = await cursor.fetchall()

        sessions = []
        for r in rows:
            sessions.append({
                "id": r[0],
                "agent_id": r[1],
                "project": r[2],
                "status": r[3],
                "archived_message_count": r[4],
                "total_tokens_used": r[5],
                "created_at": r[6],
                "updated_at": r[7],
                "message_count": r[8],
            })

    return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get a session with its messages."""
    db = get_db()
    session = await db.get_session(session_id, load_messages=True)
    if session is None:
        return {"error": "Not found"}

    messages = []
    for m in session.messages:
        messages.append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "token_count": m.token_count,
            "created_at": m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at),
        })

    return {
        "id": session.id,
        "agent_id": session.agent_id,
        "project": session.project,
        "status": session.status,
        "archived_summary_l0": session.archived_summary_l0,
        "archived_summary_l1": session.archived_summary_l1,
        "archived_message_count": session.archived_message_count,
        "total_tokens_used": session.total_tokens_used,
        "created_at": session.created_at.isoformat() if hasattr(session.created_at, "isoformat") else str(session.created_at),
        "messages": messages,
    }
