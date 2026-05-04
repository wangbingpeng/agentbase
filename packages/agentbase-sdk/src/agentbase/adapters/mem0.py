"""Mem0-compatible adapter for AgentBase.

Drop-in replacement for ``mem0.memory.main.Memory`` that delegates all
storage and retrieval to an AgentBase instance.

Usage::

    from agentbase import AgentBase
    from agentbase.adapters import Mem0Adapter

    db = AgentBase(path="./mem.db")
    await db.initialize()

    m = Mem0Adapter(db)
    m.add("I like pizza", user_id="alice")
    results = m.search("food preferences", user_id="alice")
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from .base import BaseAdapter


# Scope constant — when owner_id is set we scope to AGENT level
from agentbase_core.models import ContextScope
_AGENT_SCOPE = ContextScope.AGENT


def _run(coro: Any) -> Any:
    """Run *coro* in a dedicated event loop (safe for sync callers)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We are inside an existing loop — run in a new thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, coro)
            return fut.result()
    return asyncio.run(coro)


class Mem0Adapter(BaseAdapter):
    """Mem0-compatible adapter backed by AgentBase.

    Provides the same public surface as ``mem0.memory.main.Memory``:

    - add(messages, *, user_id, agent_id, metadata, infer, …)
    - search(query, *, filters, top_k, threshold)
    - get_all(*, filters, top_k)
    - update(memory_id, data, metadata)
    - delete(memory_id)
    - delete_all(user_id, agent_id)
    - reset()
    """

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    def add(
        self,
        messages: str | list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """Add memories (mirrors ``mem0.Memory.add``).

        When *infer* is True and *messages* is a list of role/content dicts,
        the conversation is added via ``add_conversation`` + ``commit_session``
        so that AgentBase's LLM extraction pipeline runs.

        When *infer* is False the raw text is stored directly.
        """
        owner_id = user_id or agent_id or "default"

        if isinstance(messages, str):
            return self._add_text(messages, owner_id, metadata, infer)
        # list of dicts
        return self._add_messages(messages, owner_id, metadata, infer)

    # -- internal helpers ------------------------------------------------

    def _add_text(
        self,
        text: str,
        owner_id: str,
        metadata: dict | None,
        infer: bool,
    ) -> dict:
        tags = _metadata_to_tags(metadata)
        if infer:
            entries = _run(self.db.ingest_text(text, owner_id=owner_id, tags=tags, scope=_AGENT_SCOPE))
        else:
            entry = _run(self.db.add_memory(text, owner_id=owner_id, tags=tags, scope=_AGENT_SCOPE))
            entries = [entry]

        results = [_entry_to_mem0_result(e) for e in entries]
        return {"results": results, "id": str(uuid.uuid4())}

    def _add_messages(
        self,
        messages: list[dict[str, str]],
        owner_id: str,
        metadata: dict | None,
        infer: bool,
    ) -> dict:
        turns: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            turns.append({"role": role, "content": content})

        tags = _metadata_to_tags(metadata)
        entries = _run(
            self.db.add_conversation(turns, owner_id=owner_id, tags=tags, scope=_AGENT_SCOPE)
        )

        if infer and entries:
            # Commit the auto-created session so AgentBase can extract facts
            session_id = entries[0].session_id if hasattr(entries[0], "session_id") else None
            if session_id:
                _run(self.db.commit_session(session_id))

        results = [_entry_to_mem0_result(e) for e in entries]
        return {"results": results, "id": str(uuid.uuid4())}

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: dict | None = None,
        top_k: int = 20,
        threshold: float = 0.1,
        rerank: bool = False,
    ) -> dict:
        """Search memories (mirrors ``mem0.Memory.search``)."""
        owner_id = _resolve_owner(filters, user_id, agent_id)
        results = _run(
            self.db.find(query=query, top_k=top_k, owner_id=owner_id)
        )
        mem0_results = []
        for r in results:
            score = r.score if hasattr(r, "score") else 0.0
            if score < threshold:
                continue
            entry = r.entry if hasattr(r, "entry") else r
            mem0_results.append(
                {
                    "id": entry.id,
                    "memory": entry.l2_full if hasattr(entry, "l2_full") else str(entry),
                    "score": score,
                    "metadata": _entry_metadata(entry),
                }
            )
        return {"results": mem0_results}

    # ------------------------------------------------------------------
    # get_all
    # ------------------------------------------------------------------

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: dict | None = None,
        top_k: int = 20,
    ) -> dict:
        """List all memories (mirrors ``mem0.Memory.get_all``)."""
        owner_id = _resolve_owner(filters, user_id, agent_id)
        entries = _run(self.db.list_entries(limit=top_k))
        # Filter by owner_id when specified
        if owner_id:
            entries = [e for e in entries if e.owner_id == owner_id]
        results = [_entry_to_mem0_result(e) for e in entries]
        return {"results": results}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        memory_id: str,
        data: str,
        metadata: dict | None = None,
    ) -> dict:
        """Update a memory by ID (mirrors ``mem0.Memory.update``)."""
        tags = _metadata_to_tags(metadata) if metadata else None
        entry = _run(self.db.update(memory_id, content=data, tags=tags))
        if entry is None:
            return {"results": []}
        return _entry_to_mem0_result(entry)

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> dict:
        """Delete a single memory (mirrors ``mem0.Memory.delete``)."""
        ok = _run(self.db.delete(memory_id))
        return {"deleted": ok}

    def delete_all(self, user_id: str | None = None, agent_id: str | None = None) -> dict:
        """Delete all memories for a user/agent (mirrors ``mem0.Memory.delete_all``)."""
        owner_id = user_id or agent_id
        count = _run(self.db.delete_all(owner_id=owner_id))
        return {"deleted": count}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all stored data (mirrors ``mem0.Memory.reset``)."""
        _run(self.db.delete_all())


# ======================================================================
# Private helpers
# ======================================================================


def _resolve_owner(
    filters: dict | None,
    user_id: str | None,
    agent_id: str | None,
) -> str | None:
    """Derive owner_id from filters or explicit IDs."""
    if user_id:
        return user_id
    if agent_id:
        return agent_id
    if filters:
        return filters.get("user_id") or filters.get("agent_id")
    return None


def _metadata_to_tags(metadata: dict | None) -> list[str] | None:
    """Convert mem0 metadata dict to AgentBase tags list."""
    if not metadata:
        return None
    return [f"{k}:{v}" for k, v in metadata.items()]


def _entry_to_mem0_result(entry: Any) -> dict:
    """Convert an AgentBase ContextEntry to mem0 result dict."""
    return {
        "id": entry.id,
        "memory": entry.l2_full if hasattr(entry, "l2_full") else str(entry),
        "score": getattr(entry, "confidence", 1.0),
        "metadata": _entry_metadata(entry),
    }


def _entry_metadata(entry: Any) -> dict:
    """Extract metadata dict from an entry."""
    meta: dict = {}
    if hasattr(entry, "tags") and entry.tags:
        meta["tags"] = entry.tags
    if hasattr(entry, "owner_id") and entry.owner_id:
        meta["user_id"] = entry.owner_id
    if hasattr(entry, "category") and entry.category:
        meta["category"] = entry.category
    if hasattr(entry, "session_id") and entry.session_id:
        meta["session_id"] = entry.session_id
    return meta
