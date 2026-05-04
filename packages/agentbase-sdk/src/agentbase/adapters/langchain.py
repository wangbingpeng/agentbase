"""LangChain Memory-compatible adapter for AgentBase.

Implements the duck-typed interface of LangChain's ``BaseChatMemory`` so that
AgentBase can be used as a drop-in memory backend for LangChain agents
*without* requiring ``langchain-core`` as a hard dependency.

If ``langchain-core`` is installed, the adapter also inherits from
``BaseChatMemory`` for full type compatibility.

Usage::

    from agentbase import AgentBase
    from agentbase.adapters import LangChainMemoryAdapter

    db = AgentBase(path="./mem.db")
    await db.initialize()

    memory = LangChainMemoryAdapter(db, owner_id="alice")
    memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
    history = memory.load_memory_variables({"input": "Hello"})
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .mem0 import _run


from agentbase_core.models import ContextScope
_AGENT_SCOPE = ContextScope.AGENT


class LangChainMemoryAdapter(BaseAdapter):
    """LangChain-compatible memory backed by AgentBase.

    Key properties / methods that LangChain expects:

    - ``memory_variables`` → ``["history"]``
    - ``save_context(inputs, outputs)`` → store a user-assistant turn
    - ``load_memory_variables(inputs)`` → retrieve relevant history
    - ``clear()`` → delete all entries for *owner_id*

    Parameters
    ----------
    db : AgentBase
        The underlying AgentBase instance.
    owner_id : str
        Scope all memory operations to this owner (typically a user ID).
    top_k : int
        How many results to return from ``load_memory_variables``.
    memory_key : str
        The key used in the returned dict (default ``"history"``).
    """

    def __init__(
        self,
        db: Any,
        *,
        owner_id: str = "default",
        top_k: int = 10,
        memory_key: str = "history",
    ) -> None:
        super().__init__(db)
        self._owner_id = owner_id
        self._top_k = top_k
        self._memory_key = memory_key

    # ------------------------------------------------------------------
    # LangChain interface — properties
    # ------------------------------------------------------------------

    @property
    def memory_variables(self) -> list[str]:
        """Return the list of memory variable names (LangChain convention)."""
        return [self._memory_key]

    # ------------------------------------------------------------------
    # LangChain interface — save_context
    # ------------------------------------------------------------------

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save a user-assistant turn to AgentBase.

        Parameters
        ----------
        inputs : dict
            Typically ``{"input": "user message"}``.
        outputs : dict
            Typically ``{"output": "assistant response"}``.
        """
        user_msg = inputs.get("input", inputs.get("human", ""))
        ai_msg = outputs.get("output", outputs.get("ai", ""))
        if not user_msg and not ai_msg:
            return
        turns: list[dict[str, str]] = []
        if user_msg:
            turns.append({"role": "user", "content": str(user_msg)})
        if ai_msg:
            turns.append({"role": "assistant", "content": str(ai_msg)})
        _run(self.db.add_conversation(turns, owner_id=self._owner_id, scope=_AGENT_SCOPE))

    # ------------------------------------------------------------------
    # LangChain interface — load_memory_variables
    # ------------------------------------------------------------------

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Retrieve relevant memory formatted as a history string.

        The *inputs* dict is used as a query for semantic search.
        """
        query = inputs.get("input", inputs.get("human", ""))
        if not query:
            # Fallback: return recent entries
            entries = _run(self.db.list_entries(limit=self._top_k))
            if self._owner_id:
                entries = [e for e in entries if e.owner_id == self._owner_id]
            return {self._memory_key: _format_entries(entries)}

        results = _run(
            self.db.find(query=query, top_k=self._top_k, owner_id=self._owner_id)
        )
        entries = [r.entry if hasattr(r, "entry") else r for r in results]
        return {self._memory_key: _format_entries(entries)}

    # ------------------------------------------------------------------
    # LangChain interface — clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all memory entries for the configured *owner_id*."""
        _run(self.db.delete_all(owner_id=self._owner_id))


# ======================================================================
# Private helpers
# ======================================================================


def _format_entries(entries: list[Any]) -> str:
    """Format a list of ContextEntry objects into a human-readable string."""
    lines: list[str] = []
    for entry in entries:
        role = "Assistant" if getattr(entry, "context_type", None) == "assistant" else "Human"
        content = getattr(entry, "l2_full", str(entry))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
