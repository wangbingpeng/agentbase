"""Tests for ContextEntry model and related enums."""

import pytest

from agentbase_core.models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    OriginType,
)


class TestContextType:
    def test_values(self):
        assert ContextType.MEMORY.value == "memory"
        assert ContextType.RESOURCE.value == "resource"
        assert ContextType.SKILL.value == "skill"


class TestContextScope:
    def test_values(self):
        assert ContextScope.GLOBAL.value == "global"
        assert ContextScope.AGENT.value == "agent"
        assert ContextScope.PROJECT.value == "project"
        assert ContextScope.SESSION.value == "session"


class TestEntryStatus:
    def test_lifecycle_order(self):
        assert EntryStatus.PENDING.value == "pending"
        assert EntryStatus.ACTIVE.value == "active"
        assert EntryStatus.SUPERSEDED.value == "superseded"
        assert EntryStatus.DELETED.value == "deleted"


class TestContextEntry:
    def test_default_creation(self):
        entry = ContextEntry(l2_full="test content")
        assert entry.context_type == ContextType.MEMORY
        assert entry.status == EntryStatus.PENDING
        assert entry.scope == ContextScope.GLOBAL
        assert entry.owner_id is None
        assert entry.id  # ULID auto-generated
        assert len(entry.id) == 26  # ULID length

    def test_generate_uri(self):
        entry = ContextEntry(
            l2_full="test",
            context_type=ContextType.MEMORY,
            memory_category=MemoryCategory.PREFERENCE,
        )
        uri = entry.generate_uri()
        assert uri.startswith("ctx://memory/preference/")
        assert entry.id in uri

    def test_mark_active(self):
        entry = ContextEntry(l2_full="test")
        assert entry.status == EntryStatus.PENDING
        entry.mark_active()
        assert entry.status == EntryStatus.ACTIVE
        assert entry.uri  # URI should be set

    def test_soft_delete(self):
        entry = ContextEntry(l2_full="test")
        entry.mark_active()
        entry.soft_delete()
        assert entry.status == EntryStatus.DELETED
        assert entry.deleted_at is not None

    def test_supersede(self):
        entry = ContextEntry(l2_full="old content")
        entry.mark_active()
        entry.supersede("new-id-123")
        assert entry.status == EntryStatus.SUPERSEDED
        assert entry.superseded_by == "new-id-123"
        assert entry.valid_until is not None

    def test_is_active_property(self):
        entry = ContextEntry(l2_full="test")
        assert not entry.is_active
        entry.mark_active()
        assert entry.is_active

    def test_tags_text(self):
        entry = ContextEntry(l2_full="test", tags=["python", "ai"])
        assert entry.tags_text == "python ai"

    def test_get_embedding_input_memory(self):
        entry = ContextEntry(
            l2_full="full content here",
            l1_overview="overview text",
            context_type=ContextType.MEMORY,
        )
        # Should prefer l1_overview
        assert entry.get_embedding_input() == "overview text"

    def test_get_embedding_input_memory_fallback(self):
        entry = ContextEntry(
            l2_full="full content here that is definitely more than 512 characters" * 10,
            context_type=ContextType.MEMORY,
        )
        # Should fallback to l2_full[:512]
        result = entry.get_embedding_input()
        assert len(result) <= 520  # some margin

    def test_apply_truncation_fallback(self):
        entry = ContextEntry(l2_full="x" * 600)
        entry.apply_truncation_fallback()
        assert entry.l0_abstract.endswith("...")
        assert entry.l1_overview.endswith("...")
        assert len(entry.l0_abstract) <= 110  # 100 + "..."
        assert len(entry.l1_overview) <= 510  # 500 + "..."

    def test_scope_validation_global(self):
        """global scope must have owner_id=None."""
        entry = ContextEntry(
            l2_full="test",
            scope=ContextScope.GLOBAL,
            owner_id=None,
        )
        # Should not raise
        from agentbase_core.store.sqlite_store import SQLiteStore
        SQLiteStore._validate_scope_owner(entry)

    def test_scope_validation_global_with_owner_raises(self):
        """global scope with owner_id should raise ValidationError."""
        from agentbase_core.exceptions import ValidationError
        from agentbase_core.store.sqlite_store import SQLiteStore

        entry = ContextEntry(
            l2_full="test",
            scope=ContextScope.GLOBAL,
            owner_id="agent-1",
        )
        with pytest.raises(ValidationError):
            SQLiteStore._validate_scope_owner(entry)

    def test_scope_validation_agent_without_owner_raises(self):
        """agent scope without owner_id should raise ValidationError."""
        from agentbase_core.exceptions import ValidationError
        from agentbase_core.store.sqlite_store import SQLiteStore

        entry = ContextEntry(
            l2_full="test",
            scope=ContextScope.AGENT,
            owner_id=None,
        )
        with pytest.raises(ValidationError):
            SQLiteStore._validate_scope_owner(entry)
