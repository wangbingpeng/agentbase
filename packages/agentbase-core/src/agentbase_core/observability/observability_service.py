"""ObservabilityService — trace collection, metrics, and debug interface."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ..models.context_entry import _new_ulid, _utcnow
from ..models.trace import RetrievalTrace, TraceStep
from ..store.connection import ConnectionPool

logger = logging.getLogger(__name__)


class TraceCollector:
    """Collect and persist retrieval traces."""

    def __init__(self, pool: ConnectionPool, sample_rate: float = 1.0) -> None:
        self._pool = pool
        self._sample_rate = sample_rate
        self._active_traces: dict[str, RetrievalTrace] = {}

    async def start_trace(self, query: str, strategy: str) -> RetrievalTrace:
        """Start a new retrieval trace."""
        import random
        if random.random() > self._sample_rate:
            return RetrievalTrace(query=query, strategy=strategy)

        trace = RetrievalTrace(query=query, strategy=strategy)
        self._active_traces[trace.id] = trace
        return trace

    async def record_step(self, trace: RetrievalTrace, step: TraceStep) -> None:
        """Record a step in the trace."""
        trace.add_step(step)

    async def finish_trace(self, trace: RetrievalTrace) -> None:
        """Finish and persist a trace."""
        if trace.id not in self._active_traces:
            return

        # Persist trace
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO retrieval_traces (id, query, strategy, steps, result_ids, total_latency_ms, token_budget_used, token_budget_limit, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.id, trace.query, trace.strategy,
                    json.dumps([s.model_dump() for s in trace.steps]),
                    json.dumps(trace.result_ids),
                    trace.total_latency_ms,
                    trace.token_budget_used,
                    trace.token_budget_limit,
                    now,
                ),
            )
            # Persist individual steps
            for i, step in enumerate(trace.steps):
                await conn.execute(
                    "INSERT INTO trace_steps (trace_id, step_order, step_name, latency_ms, input_count, output_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (trace.id, i, step.step, step.latency_ms, step.candidates_in, step.candidates_out),
                )
            await conn.commit()

        self._active_traces.pop(trace.id, None)


class ContextMetrics:
    """Compute and report context quality metrics."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    async def get_metrics(self) -> dict[str, Any]:
        """Get current context quality metrics."""
        metrics: dict[str, Any] = {}

        async with self._pool.get_read_conn() as conn:
            # Query count
            cursor = await conn.execute("SELECT COUNT(*) FROM retrieval_traces")
            row = await cursor.fetchone()
            metrics["query_count"] = row[0] if row else 0

            # Average latency
            cursor = await conn.execute("SELECT AVG(total_latency_ms) FROM retrieval_traces")
            row = await cursor.fetchone()
            metrics["avg_latency_ms"] = round(row[0], 2) if row and row[0] else 0.0

            # P50 latency
            cursor = await conn.execute(
                "SELECT total_latency_ms FROM retrieval_traces ORDER BY total_latency_ms LIMIT 1 OFFSET (SELECT COUNT(*) FROM retrieval_traces) / 2"
            )
            row = await cursor.fetchone()
            metrics["p50_latency_ms"] = round(row[0], 2) if row and row[0] else 0.0

            # Background job backlog
            cursor = await conn.execute("SELECT COUNT(*) FROM background_jobs WHERE status = 'pending'")
            row = await cursor.fetchone()
            metrics["pending_jobs"] = row[0] if row else 0

            # Active context count
            cursor = await conn.execute("SELECT COUNT(*) FROM context_entries WHERE status = 'active'")
            row = await cursor.fetchone()
            metrics["active_entries"] = row[0] if row else 0

            # Failed jobs
            cursor = await conn.execute("SELECT COUNT(*) FROM background_jobs WHERE status = 'failed'")
            row = await cursor.fetchone()
            metrics["failed_jobs"] = row[0] if row else 0

        return metrics


class DebugService:
    """Context debugging service — explain queries, diff contexts, trace sessions."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    async def get_trace(self, trace_id: str) -> RetrievalTrace | None:
        """Get a retrieval trace by ID."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT * FROM retrieval_traces WHERE id = ?", (trace_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return RetrievalTrace(
                id=row[0], query=row[1], strategy=row[2],
                steps=[TraceStep(**s) for s in json.loads(row[3])] if row[3] else [],
                result_ids=json.loads(row[4]) if row[4] else [],
                total_latency_ms=row[5],
                token_budget_used=row[6],
                token_budget_limit=row[7],
                created_at=datetime.fromisoformat(row[8]),
            )

    async def list_recent_traces(self, limit: int = 20) -> list[dict]:
        """List recent retrieval traces."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                "SELECT id, query, strategy, total_latency_ms, result_ids, created_at FROM retrieval_traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0], "query": r[1], "strategy": r[2],
                    "latency_ms": r[3], "result_count": len(json.loads(r[4])) if r[4] else 0,
                    "created_at": r[5],
                }
                for r in rows
            ]

    async def explain_query(self, query: str) -> dict:
        """Explain what a retrieval query would do (dry run)."""
        return {
            "query": query,
            "normalized": query.lower().strip(),
            "estimated_strategy": "hybrid",
            "note": "This is a dry-run explanation. Actual results may vary.",
        }

    async def diff_contexts(self, id1: str, id2: str) -> dict:
        """Show the difference between two context entries (per SPEC §13.3)."""
        async with self._pool.get_read_conn() as conn:
            cursor1 = await conn.execute("SELECT * FROM context_entries WHERE id = ?", (id1,))
            row1 = await cursor1.fetchone()
            cursor2 = await conn.execute("SELECT * FROM context_entries WHERE id = ?", (id2,))
            row2 = await cursor2.fetchone()

        if row1 is None or row2 is None:
            return {"error": f"One or both entries not found", "id1_found": row1 is not None, "id2_found": row2 is not None}

        # Compare key fields
        col_names = ["id", "context_type", "status", "scope", "confidence", "tags", "l0_abstract", "l1_overview", "l2_full"]
        diffs = {}
        for i, col in enumerate(col_names):
            v1 = row1[i] if i < len(row1) else None
            v2 = row2[i] if i < len(row2) else None
            if v1 != v2:
                diffs[col] = {"entry1": v1, "entry2": v2}

        return {"id1": id1, "id2": id2, "differences": diffs, "identical": len(diffs) == 0}

    async def trace_session(self, session_id: str) -> list[dict]:
        """Show all retrieval traces for a session (per SPEC §13.3)."""
        # Sessions can be linked to traces via origin_id or through session_memory_links
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute(
                """SELECT rt.id, rt.query, rt.strategy, rt.total_latency_ms, rt.created_at
                   FROM retrieval_traces rt
                   WHERE rt.query LIKE ?
                   ORDER BY rt.created_at DESC""",
                (f"%{session_id}%",),
            )
            rows = await cursor.fetchall()

        return [
            {
                "id": r[0],
                "query": r[1],
                "strategy": r[2],
                "latency_ms": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    async def entity_graph(self, entity_name: str, depth: int = 2) -> dict:
        """Visualize entity relationship graph (per SPEC §13.3)."""
        async with self._pool.get_read_conn() as conn:
            # Find the entity
            cursor = await conn.execute(
                "SELECT id, name, entity_type, description FROM entities WHERE name = ?",
                (entity_name,),
            )
            entity_row = await cursor.fetchone()
            if entity_row is None:
                return {"error": f"Entity not found: {entity_name}"}

            entity_id = entity_row[0]

            # Get current relations
            cursor = await conn.execute(
                """SELECT r.id, r.source_id, e1.name, r.predicate, r.target_id, e2.name, r.confidence, r.valid_from
                   FROM relations r
                   JOIN entities e1 ON e1.id = r.source_id
                   JOIN entities e2 ON e2.id = r.target_id
                   WHERE (r.source_id = ? OR r.target_id = ?) AND r.valid_until IS NULL
                   ORDER BY r.confidence DESC
                   LIMIT 50""",
                (entity_id, entity_id),
            )
            rel_rows = await cursor.fetchall()

        nodes = {entity_id: {"name": entity_row[1], "type": entity_row[2], "description": entity_row[3] or ""}}
        edges = []
        for r in rel_rows:
            rid, sid, sname, pred, tid, tname, conf, vf = r
            if sid not in nodes:
                nodes[sid] = {"name": sname}
            if tid not in nodes:
                nodes[tid] = {"name": tname}
            edges.append({"source": sid, "source_name": sname, "predicate": pred, "target": tid, "target_name": tname, "confidence": conf})

        return {
            "entity": {"id": entity_id, "name": entity_row[1], "type": entity_row[2]},
            "nodes": nodes,
            "edges": edges,
            "depth": depth,
        }
