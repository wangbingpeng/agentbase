"""LlamaIndex ChatStore adapter for AgentBase.

Implements LlamaIndex's ``BaseChatStore`` interface so that AgentBase
can serve as the persistence backend for LlamaIndex's chat memory.

Requires ``llama-index-core`` to be installed (optional dependency).

Usage::

    from agentbase import AgentBase
    from agentbase.adapters.llamaindex import AgentBaseChatStore
    from llama_index.core.memory import ChatMemoryBuffer

    db = AgentBase(path="./mem.db")
    await db.initialize()

    chat_store = AgentBaseChatStore(db)
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user_alice",
    )
"""

from __future__ import annotations

from typing import Any, List, Optional

from .base import BaseAdapter
from .mem0 import _run


# Scope constant — owner_id requires AGENT scope
from agentbase_core.models import ContextScope
_AGENT_SCOPE = ContextScope.AGENT

# Lazy import for llama_index — only needed at runtime
_ChatMessage: Any = None


def _get_chat_message_class() -> Any:
    """Lazily import ``ChatMessage`` from llama_index."""
    global _ChatMessage
    if _ChatMessage is None:
        try:
            from llama_index.core.llms import ChatMessage
            _ChatMessage = ChatMessage
        except ImportError as exc:
            raise ImportError(
                "llama-index-core is required for AgentBaseChatStore. "
                "Install it with: pip install agentbase-sdk[llamaindex]"
            ) from exc
    return _ChatMessage


class AgentBaseChatStore(BaseAdapter):
    """LlamaIndex ``BaseChatStore`` implementation backed by AgentBase.

    Each LlamaIndex *key* maps to an AgentBase *owner_id*, so that
    different users / conversations are naturally isolated.

    Methods mirror the ``BaseChatStore`` abstract interface:

    - ``set_messages(key, messages)`` — replace all messages for a key
    - ``get_messages(key)`` — retrieve all messages for a key
    - ``add_message(key, message)`` — append a single message
    - ``delete_messages(key)`` — delete all messages for a key
    - ``delete_message(key, idx)`` — delete message at index
    - ``delete_last_message(key)`` — delete the most recent message
    - ``get_keys()`` — list all keys (unique owner_ids)
    """

    # ------------------------------------------------------------------
    # Core implementation
    # ------------------------------------------------------------------

    def set_messages(self, key: str, messages: List[Any]) -> None:
        """Replace all messages for *key* with *messages*."""
        ChatMessage = _get_chat_message_class()

        # Clear existing entries for this key
        _run(self.db.delete_all(owner_id=key))

        # Convert ChatMessage list → AgentBase conversation turns
        turns: list[dict[str, str]] = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            # LlamaIndex MessageRole enum → plain string
            if hasattr(role, "value"):
                role = role.value
            content = getattr(msg, "content", "")
            turns.append({"role": str(role), "content": str(content)})

        if turns:
            _run(self.db.add_conversation(
                turns, owner_id=key, scope=_AGENT_SCOPE,
            ))

    def get_messages(self, key: str) -> List[Any]:
        """Get all messages for *key*, ordered chronologically."""
        ChatMessage = _get_chat_message_class()

        entries = _run(self.db.list_entries(limit=10000))
        # Filter to this key and sort by created_at
        entries = [e for e in entries if e.owner_id == key]
        entries.sort(key=lambda e: getattr(e, "created_at", ""))

        messages: list[Any] = []
        for entry in entries:
            content = getattr(entry, "l2_full", "") or ""
            # Try to recover role from tags or default to "user"
            role = self._infer_role(entry)
            messages.append(ChatMessage(role=role, content=content))
        return messages

    def add_message(self, key: str, message: Any) -> None:
        """Append a single message for *key*."""
        role = getattr(message, "role", "user")
        if hasattr(role, "value"):
            role = role.value
        content = getattr(message, "content", "")
        _run(self.db.add_memory(
            str(content),
            owner_id=key,
            tags=[f"role:{role}"],
            scope=_AGENT_SCOPE,
        ))

    def delete_messages(self, key: str) -> Optional[List[Any]]:
        """Delete all messages for *key*. Returns the deleted messages."""
        messages = self.get_messages(key)
        _run(self.db.delete_all(owner_id=key))
        return messages

    def delete_message(self, key: str, idx: int) -> Optional[Any]:
        """Delete message at *idx* for *key*."""
        messages = self.get_messages(key)
        if idx < 0 or idx >= len(messages):
            return None
        deleted = messages[idx]

        # Rebuild without the deleted message
        remaining = messages[:idx] + messages[idx + 1:]
        self.set_messages(key, remaining)
        return deleted

    def delete_last_message(self, key: str) -> Optional[Any]:
        """Delete the last (most recent) message for *key*."""
        messages = self.get_messages(key)
        if not messages:
            return None
        return self.delete_message(key, len(messages) - 1)

    def get_keys(self) -> List[str]:
        """Get all keys (unique owner_ids with messages)."""
        entries = _run(self.db.list_entries(limit=10000))
        keys: set[str] = set()
        for e in entries:
            if e.owner_id:
                keys.add(e.owner_id)
        return sorted(keys)

    # ------------------------------------------------------------------
    # Async overrides (optional — BaseChatStore provides defaults via
    # asyncio.to_thread, but we can be more efficient by calling AgentBase
    # async API directly)
    # ------------------------------------------------------------------

    async def aset_messages(self, key: str, messages: List[Any]) -> None:
        """Async version of set_messages."""
        await self.db.delete_all(owner_id=key)
        ChatMessage = _get_chat_message_class()
        turns: list[dict[str, str]] = []
        for msg in messages:
            role = getattr(msg, "role", "user")
            if hasattr(role, "value"):
                role = role.value
            content = getattr(msg, "content", "")
            turns.append({"role": str(role), "content": str(content)})
        if turns:
            await self.db.add_conversation(
                turns, owner_id=key, scope=_AGENT_SCOPE,
            )

    async def aget_messages(self, key: str) -> List[Any]:
        """Async version of get_messages."""
        ChatMessage = _get_chat_message_class()
        entries = await self.db.list_entries(limit=10000)
        entries = [e for e in entries if e.owner_id == key]
        entries.sort(key=lambda e: getattr(e, "created_at", ""))
        messages: list[Any] = []
        for entry in entries:
            content = getattr(entry, "l2_full", "") or ""
            role = self._infer_role(entry)
            messages.append(ChatMessage(role=role, content=content))
        return messages

    async def aadd_message(self, key: str, message: Any) -> None:
        """Async version of add_message."""
        role = getattr(message, "role", "user")
        if hasattr(role, "value"):
            role = role.value
        content = getattr(message, "content", "")
        await self.db.add_memory(
            str(content),
            owner_id=key,
            tags=[f"role:{role}"],
            scope=_AGENT_SCOPE,
        )

    async def adelete_messages(self, key: str) -> Optional[List[Any]]:
        """Async version of delete_messages."""
        messages = await self.aget_messages(key)
        await self.db.delete_all(owner_id=key)
        return messages

    async def adelete_last_message(self, key: str) -> Optional[Any]:
        """Async version of delete_last_message."""
        messages = await self.aget_messages(key)
        if not messages:
            return None
        # Rebuild without last message
        remaining = messages[:-1]
        await self.aset_messages(key, remaining)
        return messages[-1]

    async def aget_keys(self) -> List[str]:
        """Async version of get_keys."""
        entries = await self.db.list_entries(limit=10000)
        keys: set[str] = set()
        for e in entries:
            if e.owner_id:
                keys.add(e.owner_id)
        return sorted(keys)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_role(entry: Any) -> str:
        """Infer the message role from an entry's tags or content."""
        tags = getattr(entry, "tags", None) or []
        for tag in tags:
            if tag.startswith("role:"):
                return tag.split(":", 1)[1]
        # Fallback: try context_type hint
        ct = getattr(entry, "context_type", None)
        if ct and hasattr(ct, "value"):
            ct = ct.value
        if ct == "assistant":
            return "assistant"
        return "user"
