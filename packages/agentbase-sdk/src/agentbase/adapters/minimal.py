"""Minimal adapter — the smallest useful interface to AgentBase.

Three-method API: ``remember``, ``recall``, ``forget`` — plus async variants.
Ideal for quick integration where only basic store / retrieve / delete
is needed.

Usage::

    from agentbase import AgentBase
    from agentbase.adapters import MinimalAdapter

    db = AgentBase(path="./mem.db")
    await db.initialize()

    mem = MinimalAdapter(db)
    mem.remember("User prefers dark mode", who="alice", tags=["preference"])
    results = mem.recall("theme preferences", who="alice")
    mem.forget(entry_id="...")
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .mem0 import _run


from agentbase_core.models import ContextScope
_AGENT_SCOPE = ContextScope.AGENT


class MinimalAdapter(BaseAdapter):
    """The simplest possible memory interface backed by AgentBase.

    Sync API
    -------
    - ``remember(content, who, tags)`` → store a memory
    - ``recall(query, who, top_k)`` → retrieve matching memories as ``list[str]``
    - ``forget(entry_id)`` → delete a memory

    Async API
    --------
    - ``aremember(content, who, tags)`` → ``await`` store
    - ``arecall(query, who, top_k)`` → ``await`` retrieve
    - ``aforget(entry_id)`` → ``await`` delete
    """

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        who: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Store a memory and return its ID."""
        entry = _run(self.db.add_memory(content, owner_id=who, tags=tags, scope=_AGENT_SCOPE))
        return entry.id

    def recall(
        self,
        query: str,
        *,
        who: str | None = None,
        top_k: int = 10,
    ) -> list[str]:
        """Retrieve matching memories as a list of content strings."""
        results = _run(self.db.find(query=query, top_k=top_k, owner_id=who))
        texts: list[str] = []
        for r in results:
            entry = r.entry if hasattr(r, "entry") else r
            text = getattr(entry, "l2_full", str(entry))
            texts.append(text)
        return texts

    def forget(self, entry_id: str) -> bool:
        """Delete a memory by ID. Returns True if successful."""
        return _run(self.db.delete(entry_id))

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def aremember(
        self,
        content: str,
        *,
        who: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Async version of ``remember``."""
        entry = await self.db.add_memory(content, owner_id=who, tags=tags, scope=_AGENT_SCOPE)
        return entry.id

    async def arecall(
        self,
        query: str,
        *,
        who: str | None = None,
        top_k: int = 10,
    ) -> list[str]:
        """Async version of ``recall``."""
        results = await self.db.find(query=query, top_k=top_k, owner_id=who)
        texts: list[str] = []
        for r in results:
            entry = r.entry if hasattr(r, "entry") else r
            text = getattr(entry, "l2_full", str(entry))
            texts.append(text)
        return texts

    async def aforget(self, entry_id: str) -> bool:
        """Async version of ``forget``."""
        return await self.db.delete(entry_id)
