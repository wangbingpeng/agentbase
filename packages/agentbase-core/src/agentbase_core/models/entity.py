"""Entity / Relation / FactTimeline — temporal knowledge graph models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .context_entry import _new_ulid, _utcnow


class Entity(BaseModel):
    """Knowledge graph entity."""

    id: str = Field(default_factory=_new_ulid)
    name: str
    entity_type: str  # person/project/concept/tool/event/organization
    description: str = ""
    first_seen: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    fact_count: int = 0
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"from_attributes": True}

    def touch(self) -> None:
        """Update last_seen timestamp."""
        self.last_seen = _utcnow()
        self.updated_at = _utcnow()


class Relation(BaseModel):
    """Knowledge graph relation with temporal validity."""

    id: str = Field(default_factory=_new_ulid)
    source_id: str
    target_id: str
    predicate: str  # prefers/works_on/depends_on/contains/belongs_to/uses/...
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_until: datetime | None = None  # None = still valid
    confidence: float = 1.0
    evidence_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"from_attributes": True}

    @property
    def is_current(self) -> bool:
        return self.valid_until is None

    def supersede(self) -> None:
        """Mark this relation as no longer current."""
        self.valid_until = _utcnow()
        self.updated_at = _utcnow()


class FactTimeline(BaseModel):
    """Fact temporal change record."""

    id: str = Field(default_factory=_new_ulid)
    entity_id: str
    fact: str
    valid_at: datetime = Field(default_factory=_utcnow)
    superseded_by: str | None = None
    action: str = "created"  # created/superseded/merged/deleted
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"from_attributes": True}
