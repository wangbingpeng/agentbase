"""OpenAI Assistants API-compatible adapter for AgentBase.

Provides a subset of the OpenAI ``threads`` API surface so that AgentBase
can serve as the memory backend for applications built against the
Assistants API.

This adapter maps:
- OpenAI *Thread* → AgentBase *Session*
- OpenAI *Message* → AgentBase *SessionMessage* (via ``add_message``)
- Retrieval / context injection → AgentBase ``find``

Usage::

    from agentbase import AgentBase
    from agentbase.adapters import OpenAIAssistantAdapter

    db = AgentBase(path="./mem.db")
    await db.initialize()

    oa = OpenAIAssistantAdapter(db)
    thread = oa.create_thread(metadata={"agent_id": "my-agent"})
    oa.create_message(thread_id=thread["id"], role="user", content="Hello")
    messages = oa.list_messages(thread_id=thread["id"])
"""

from __future__ import annotations

import uuid
from typing import Any

from .base import BaseAdapter
from .mem0 import _run


class OpenAIAssistantAdapter(BaseAdapter):
    """OpenAI Assistants API-compatible adapter backed by AgentBase.

    Methods mirror the OpenAI ``beta.threads`` namespace:

    - ``create_thread(metadata)`` → create an AgentBase session
    - ``create_message(thread_id, role, content)`` → add a session message
    - ``list_messages(thread_id, limit)`` → retrieve session messages
    - ``retrieve_context(thread_id, query, top_k)`` → semantic search
    - ``delete_thread(thread_id)`` → delete a thread and its data
    """

    # ------------------------------------------------------------------
    # Thread operations
    # ------------------------------------------------------------------

    def create_thread(self, metadata: dict | None = None) -> dict:
        """Create a new thread (maps to AgentBase session).

        Returns a dict shaped like OpenAI's thread object.
        """
        agent_id = (metadata or {}).get("agent_id", "default")
        project = (metadata or {}).get("project")
        session = _run(self.db.create_session(agent_id=agent_id, project=project))
        return {
            "id": session.id,
            "object": "thread",
            "metadata": metadata or {},
            "created_at": _iso_now(),
        }

    def delete_thread(self, thread_id: str) -> dict:
        """Delete a thread and all associated entries."""
        # Delete all entries associated with this thread's owner
        count = _run(self.db.delete_all(owner_id=thread_id))
        return {
            "id": thread_id,
            "object": "thread.deleted",
            "deleted": count > 0,
        }

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    def create_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        """Add a message to a thread (maps to AgentBase add_message)."""
        msg = _run(
            self.db.add_message(
                session_id=thread_id,
                role=role,
                content=content,
            )
        )
        return {
            "id": msg.id if hasattr(msg, "id") else str(uuid.uuid4()),
            "object": "thread.message",
            "thread_id": thread_id,
            "role": role,
            "content": [{"type": "text", "text": {"value": content}}],
            "metadata": metadata or {},
            "created_at": _iso_now(),
        }

    def list_messages(
        self,
        thread_id: str,
        limit: int = 20,
        order: str = "desc",
    ) -> dict:
        """List messages in a thread (maps to AgentBase get_session)."""
        session = _run(self.db.get_session(thread_id, load_messages=True))
        if session is None:
            return {"object": "list", "data": [], "first_id": None, "last_id": None}

        messages = []
        raw = getattr(session, "messages", []) or []
        for msg in raw:
            messages.append(
                {
                    "id": msg.id if hasattr(msg, "id") else str(uuid.uuid4()),
                    "object": "thread.message",
                    "thread_id": thread_id,
                    "role": getattr(msg, "role", "user"),
                    "content": [
                        {
                            "type": "text",
                            "text": {"value": getattr(msg, "content", "")},
                        }
                    ],
                    "created_at": getattr(msg, "created_at", _iso_now()),
                }
            )
        if order == "desc":
            messages.reverse()
        messages = messages[:limit]
        return {
            "object": "list",
            "data": messages,
            "first_id": messages[0]["id"] if messages else None,
            "last_id": messages[-1]["id"] if messages else None,
        }

    # ------------------------------------------------------------------
    # Retrieval / context operations
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        thread_id: str,
        query: str,
        top_k: int = 10,
    ) -> dict:
        """Retrieve relevant context for a query within a thread scope.

        This is not a standard OpenAI API method but a convenience that
        leverages AgentBase's semantic search within the thread's owner scope.
        """
        results = _run(
            self.db.find(query=query, top_k=top_k, owner_id=thread_id)
        )
        data = []
        for r in results:
            entry = r.entry if hasattr(r, "entry") else r
            data.append(
                {
                    "id": entry.id,
                    "content": getattr(entry, "l2_full", str(entry)),
                    "score": getattr(r, "score", 0.0),
                    "metadata": {
                        "thread_id": thread_id,
                    },
                }
            )
        return {"object": "list", "data": data}


# ======================================================================
# Private helpers
# ======================================================================


def _iso_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
