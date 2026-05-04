"""ContextEntry — unified context data model for AgentBase."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_ulid() -> str:
    return str(ULID())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContextType(str, Enum):
    """Context type — the primary classification axis."""

    MEMORY = "memory"
    RESOURCE = "resource"
    SKILL = "skill"


class ContextScope(str, Enum):
    """Context scope — determines visibility across agents."""

    GLOBAL = "global"
    AGENT = "agent"
    PROJECT = "project"
    SESSION = "session"


class MemoryCategory(str, Enum):
    """Memory sub-category (only valid when context_type=memory)."""

    PROFILE = "profile"
    PREFERENCE = "preference"
    ENTITY = "entity"
    EVENT = "event"
    CASE = "case"
    PATTERN = "pattern"


class EntryStatus(str, Enum):
    """Entry lifecycle status."""

    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"
    FAILED = "failed"


class OriginType(str, Enum):
    """Origin of the entry."""

    MANUAL = "manual"
    SESSION = "session"
    IMPORT = "import"
    EXTRACTED = "extracted"
    GENERATED = "generated"
    SYNCED = "synced"


# ---------------------------------------------------------------------------
# ContextEntry model
# ---------------------------------------------------------------------------


class ContextEntry(BaseModel):
    """Unified context entry — the core data unit of AgentBase."""

    id: str = Field(default_factory=_new_ulid)
    context_type: ContextType = ContextType.MEMORY
    memory_category: MemoryCategory | None = None

    # Lifecycle
    status: EntryStatus = EntryStatus.PENDING
    origin_type: OriginType = OriginType.MANUAL
    origin_id: str | None = None

    # L0 / L1 / L2 layered content
    l0_abstract: str = ""
    l1_overview: str = ""
    l2_full: str = ""

    # FTS tokenized text (space-separated words for FTS5 indexing)
    # Populated by the ingester before write; keeps l2_full pristine.
    fts_text: str = ""

    # Embedding metadata (actual vector stored separately)
    embedding_hash: str = ""
    embedding_source_level: str = "l1"
    embedding_model: str | None = None
    embedding_dimensions: int | None = None

    # Scope
    scope: ContextScope = ContextScope.GLOBAL
    owner_id: str | None = None

    # Metadata
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "unknown"
    uri: str = ""

    # Resource-specific fields
    resource_url: str | None = None
    resource_format: str | None = None
    resource_size: int | None = None

    # Skill-specific fields
    skill_tool_name: str | None = None
    skill_api_spec: dict | None = None

    # Temporal
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_until: datetime | None = None
    superseded_by: str | None = None

    # Audit
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    deleted_at: datetime | None = None

    # Extension
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}

    def validate_type_constraints(self) -> None:
        """Enforce field constraints per SPEC §4.6.

        - memory: forbid resource_url, resource_format, resource_size, skill_tool_name, skill_api_spec
        - resource: forbid memory_category, skill_tool_name, skill_api_spec
        - skill: forbid memory_category, resource_url, resource_format, resource_size
        """
        from ..exceptions import ValidationError

        if self.context_type == ContextType.MEMORY:
            if self.resource_url is not None:
                raise ValidationError("memory type forbids resource_url")
            if self.resource_format is not None:
                raise ValidationError("memory type forbids resource_format")
            if self.resource_size is not None:
                raise ValidationError("memory type forbids resource_size")
            if self.skill_tool_name is not None:
                raise ValidationError("memory type forbids skill_tool_name")
            if self.skill_api_spec is not None:
                raise ValidationError("memory type forbids skill_api_spec")

        elif self.context_type == ContextType.RESOURCE:
            if self.memory_category is not None:
                raise ValidationError("resource type forbids memory_category")
            if self.skill_tool_name is not None:
                raise ValidationError("resource type forbids skill_tool_name")
            if self.skill_api_spec is not None:
                raise ValidationError("resource type forbids skill_api_spec")

        elif self.context_type == ContextType.SKILL:
            if self.memory_category is not None:
                raise ValidationError("skill type forbids memory_category")
            if self.resource_url is not None:
                raise ValidationError("skill type forbids resource_url")
            if self.resource_format is not None:
                raise ValidationError("skill type forbids resource_format")
            if self.resource_size is not None:
                raise ValidationError("skill type forbids resource_size")

    def generate_uri(self) -> str:
        """Generate ctx:// URI for this entry."""
        category = self.memory_category.value if self.memory_category else "general"
        return f"ctx://{self.context_type.value}/{category}/{self.id}"

    def mark_active(self) -> None:
        """Transition entry to active status."""
        if self.status == EntryStatus.PENDING:
            self.status = EntryStatus.ACTIVE
            self.updated_at = _utcnow()
            if not self.uri:
                self.uri = self.generate_uri()

    def soft_delete(self) -> None:
        """Soft delete this entry."""
        self.status = EntryStatus.DELETED
        self.deleted_at = _utcnow()
        self.updated_at = _utcnow()

    def supersede(self, new_id: str) -> None:
        """Mark this entry as superseded by a newer entry."""
        self.status = EntryStatus.SUPERSEDED
        self.superseded_by = new_id
        self.valid_until = _utcnow()
        self.updated_at = _utcnow()

    def mark_failed(self) -> None:
        """Mark entry processing as failed."""
        self.status = EntryStatus.FAILED
        self.updated_at = _utcnow()

    @property
    def is_active(self) -> bool:
        return self.status == EntryStatus.ACTIVE

    @property
    def is_retrievable(self) -> bool:
        return self.status == EntryStatus.ACTIVE

    @property
    def tags_text(self) -> str:
        """Tags joined as space-separated text (for FTS indexing)."""
        return " ".join(self.tags)

    def get_embedding_input(self) -> str:
        """Return the text to be used for embedding generation based on context_type."""
        if self.context_type == ContextType.MEMORY:
            if self.l1_overview:
                return self.l1_overview
            return self.l2_full[:512]
        elif self.context_type == ContextType.RESOURCE:
            parts = []
            if self.resource_url:
                parts.append(self.resource_url)
            if self.l1_overview:
                parts.append(self.l1_overview)
            elif self.l2_full:
                parts.append(self.l2_full[:256])
            return " ".join(parts)
        elif self.context_type == ContextType.SKILL:
            parts = []
            if self.skill_tool_name:
                parts.append(self.skill_tool_name)
            if self.l1_overview:
                parts.append(self.l1_overview)
            return " ".join(parts)
        return self.l2_full[:512]

    def apply_truncation_fallback(self) -> None:
        """Apply truncation fallback when LLM is unavailable for L0/L1 generation."""
        if not self.l0_abstract:
            self.l0_abstract = self.l2_full[:100] + "..." if len(self.l2_full) > 100 else self.l2_full
        if not self.l1_overview:
            self.l1_overview = self.l2_full[:500] + "..." if len(self.l2_full) > 500 else self.l2_full
