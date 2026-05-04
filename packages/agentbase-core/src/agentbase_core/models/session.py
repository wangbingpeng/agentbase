"""Session / SessionMessage — conversation management models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .context_entry import _new_ulid, _utcnow


class SessionMessage(BaseModel):
    """A single message within a session."""

    id: str = Field(default_factory=_new_ulid)
    role: str  # user/assistant/tool/system
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    token_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"from_attributes": True}


class Session(BaseModel):
    """A conversation session."""

    id: str = Field(default_factory=_new_ulid)
    agent_id: str = "default"
    project: str | None = None
    status: str = "active"  # active/archived/deleted

    # Messages (aggregated view; persisted in session_messages table)
    messages: list[SessionMessage] = Field(default_factory=list)

    # Compression / archive
    archived_summary_l0: str = ""
    archived_summary_l1: str = ""
    archived_message_count: int = 0

    # Token usage
    total_tokens_used: int = 0

    # Memory extraction results
    extracted_memory_ids: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    committed_at: datetime | None = None

    model_config = {"from_attributes": True}

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_archived(self) -> bool:
        return self.status == "archived"

    def add_message(self, role: str, content: str, **kwargs) -> SessionMessage:
        """Add a message to the session."""
        msg = SessionMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.updated_at = _utcnow()
        return msg
