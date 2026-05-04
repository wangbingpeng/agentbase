"""API: Trace endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query

from ..app import get_db

router = APIRouter()


@router.get("/traces")
async def list_traces(
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    """List recent retrieval traces."""
    db = get_db()
    pool = db._engine._pool

    async with pool.get_read_conn() as conn:
        cursor = await conn.execute(
            """SELECT id, query, strategy, total_latency_ms,
                      token_budget_used, token_budget_limit, result_ids, created_at
               FROM retrieval_traces
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        traces = []
        for r in rows:
            result_ids = json.loads(r[6]) if r[6] else []
            traces.append({
                "id": r[0],
                "query": r[1],
                "strategy": r[2],
                "latency_ms": round(r[3], 2) if r[3] else 0,
                "token_budget_used": r[4],
                "token_budget_limit": r[5],
                "result_count": len(result_ids),
                "created_at": r[7],
            })

    return {"traces": traces}


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict:
    """Get a trace with its steps."""
    db = get_db()
    pool = db._engine._pool

    async with pool.get_read_conn() as conn:
        cursor = await conn.execute(
            """SELECT id, query, strategy, steps, result_ids,
                      total_latency_ms, token_budget_used, token_budget_limit, created_at
               FROM retrieval_traces WHERE id = ?""",
            (trace_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return {"error": "Not found"}

        steps = json.loads(row[3]) if row[3] else []
        result_ids = json.loads(row[4]) if row[4] else []

        # Fetch trace_steps if available
        cursor2 = await conn.execute(
            """SELECT trace_id, step_order, step_name, started_at, finished_at,
                      latency_ms, input_count, output_count, model_name, cache_hit, error_code,
                      budget_before, budget_after
               FROM trace_steps WHERE trace_id = ?
               ORDER BY step_order""",
            (trace_id,),
        )
        step_rows = await cursor2.fetchall()

        detailed_steps = []
        for sr in step_rows:
            detailed_steps.append({
                "order": sr[1],
                "name": sr[2],
                "started_at": sr[3],
                "finished_at": sr[4],
                "latency_ms": sr[5],
                "input_count": sr[6],
                "output_count": sr[7],
                "model_name": sr[8],
                "cache_hit": bool(sr[9]) if sr[9] is not None else None,
                "error_code": sr[10],
                "budget_before": sr[11],
                "budget_after": sr[12],
            })

        # Use detailed_steps if available, else fall back to JSON steps
        final_steps = detailed_steps if detailed_steps else steps

        return {
            "id": row[0],
            "query": row[1],
            "strategy": row[2],
            "steps": final_steps,
            "result_ids": result_ids,
            "result_count": len(result_ids),
            "latency_ms": round(row[5], 2) if row[5] else 0,
            "token_budget_used": row[6],
            "token_budget_limit": row[7],
            "created_at": row[8],
        }
