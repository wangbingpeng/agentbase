"""Tests for SQLiteStore — CRUD operations."""

import pytest
import pytest_asyncio

from agentbase_core.exceptions import ValidationError
from agentbase_core.models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    OriginType,
)


@pytest.mark.asyncio
class TestSQLiteStore:
    async def test_add_and_get(self, store):
        entry = ContextEntry(
            l2_full="Test content for storage",
            context_type=ContextType.MEMORY,
            memory_category=MemoryCategory.PREFERENCE,
            tags=["test", "storage"],
            confidence=0.9,
            scope=ContextScope.GLOBAL,
        )
        entry.mark_active()

        result = await store.add(entry)
        assert result.id == entry.id

        fetched = await store.get(entry.id)
        assert fetched is not None
        assert fetched.l2_full == "Test content for storage"
        assert fetched.context_type == ContextType.MEMORY
        assert fetched.memory_category == MemoryCategory.PREFERENCE
        assert fetched.tags == ["test", "storage"]
        assert fetched.confidence == 0.9
        assert fetched.status == EntryStatus.ACTIVE

    async def test_add_with_scope_validation(self, store):
        """Global scope must not have owner_id."""
        entry = ContextEntry(
            l2_full="test",
            scope=ContextScope.GLOBAL,
            owner_id="should-fail",
        )
        with pytest.raises(ValidationError):
            await store.add(entry)

    async def test_add_agent_scope(self, store):
        entry = ContextEntry(
            l2_full="Agent-specific content",
            scope=ContextScope.AGENT,
            owner_id="agent-001",
        )
        entry.mark_active()
        result = await store.add(entry)
        assert result.scope == ContextScope.AGENT
        assert result.owner_id == "agent-001"

    async def test_soft_delete(self, store):
        entry = ContextEntry(l2_full="to be deleted", scope=ContextScope.GLOBAL)
        entry.mark_active()
        await store.add(entry)

        deleted = await store.delete(entry.id)
        assert deleted is True

        fetched = await store.get(entry.id)
        assert fetched is not None
        assert fetched.status == EntryStatus.DELETED
        assert fetched.deleted_at is not None

    async def test_hard_delete(self, store):
        entry = ContextEntry(l2_full="to be purged", scope=ContextScope.GLOBAL)
        entry.mark_active()
        await store.add(entry)

        purged = await store.purge(entry.id)
        assert purged is True

        fetched = await store.get(entry.id)
        assert fetched is None

    async def test_supersede(self, store):
        old = ContextEntry(
            l2_full="old content",
            scope=ContextScope.GLOBAL,
        )
        old.mark_active()
        await store.add(old)

        new = ContextEntry(
            l2_full="new content",
            scope=ContextScope.GLOBAL,
        )
        new.mark_active()
        await store.supersede(old.id, new)

        old_fetched = await store.get(old.id)
        assert old_fetched.status == EntryStatus.SUPERSEDED
        assert old_fetched.superseded_by == new.id

        new_fetched = await store.get(new.id)
        assert new_fetched.l2_full == "new content"

    async def test_update(self, store):
        entry = ContextEntry(
            l2_full="original",
            scope=ContextScope.GLOBAL,
        )
        entry.mark_active()
        await store.add(entry)

        entry.l2_full = "updated"
        await store.update(entry)

        fetched = await store.get(entry.id)
        assert fetched.l2_full == "updated"

    async def test_list_entries(self, store):
        for i in range(5):
            entry = ContextEntry(
                l2_full=f"content {i}",
                scope=ContextScope.GLOBAL,
            )
            entry.mark_active()
            await store.add(entry)

        entries = await store.list_entries(limit=3, offset=0)
        assert len(entries) == 3

        all_entries = await store.list_entries(limit=100)
        assert len(all_entries) >= 5

    async def test_list_entries_filter_by_type(self, store):
        mem = ContextEntry(l2_full="memory", context_type=ContextType.MEMORY, scope=ContextScope.GLOBAL)
        mem.mark_active()
        await store.add(mem)

        res = ContextEntry(l2_full="resource", context_type=ContextType.RESOURCE, scope=ContextScope.GLOBAL, resource_url="https://example.com")
        res.mark_active()
        await store.add(res)

        memories = await store.list_entries(context_type=ContextType.MEMORY)
        assert all(e.context_type == ContextType.MEMORY for e in memories)

    async def test_count(self, store):
        for i in range(3):
            entry = ContextEntry(l2_full=f"content {i}", scope=ContextScope.GLOBAL)
            entry.mark_active()
            await store.add(entry)

        count = await store.count()
        assert count >= 3

    async def test_list_visible(self, store):
        # Global entry
        g = ContextEntry(l2_full="global", scope=ContextScope.GLOBAL)
        g.mark_active()
        await store.add(g)

        # Agent entry
        a = ContextEntry(l2_full="agent", scope=ContextScope.AGENT, owner_id="agent-1")
        a.mark_active()
        await store.add(a)

        # Different agent entry
        a2 = ContextEntry(l2_full="agent2", scope=ContextScope.AGENT, owner_id="agent-2")
        a2.mark_active()
        await store.add(a2)

        # Agent-1 should see global + their own
        visible = await store.list_visible(agent_id="agent-1")
        visible_ids = {e.id for e in visible}
        assert g.id in visible_ids
        assert a.id in visible_ids
        assert a2.id not in visible_ids

    async def test_get_nonexistent(self, store):
        result = await store.get("nonexistent-id")
        assert result is None

    async def test_delete_nonexistent(self, store):
        result = await store.delete("nonexistent-id")
        assert result is False
