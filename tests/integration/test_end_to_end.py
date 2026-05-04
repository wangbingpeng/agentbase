"""End-to-end integration tests."""

import pytest

from agentbase_core.models.context_entry import (
    ContextScope,
    ContextType,
    MemoryCategory,
)


@pytest.mark.asyncio
class TestEndToEnd:
    async def test_full_lifecycle(self, engine):
        """Test the complete lifecycle: add -> search -> get -> delete."""
        # Add
        entry = await engine.add_memory(
            content="User prefers dark mode in IDE",
            category=MemoryCategory.PREFERENCE,
            tags=["ide", "dark-mode"],
            confidence=0.9,
            scope=ContextScope.GLOBAL,
        )
        assert entry.id
        assert entry.status.value == "active"
        assert entry.l0_abstract  # Should have truncation fallback

        # Search
        results = await engine.find("dark mode", top_k=5)
        assert len(results) >= 1
        found = results[0]
        assert "dark mode" in found.entry.l2_full.lower()

        # Get
        fetched = await engine.get(entry.id)
        assert fetched is not None
        assert fetched.id == entry.id

        # Delete
        deleted = await engine.delete(entry.id)
        assert deleted is True

        # Verify deleted
        fetched_after = await engine.get(entry.id)
        assert fetched_after.status.value == "deleted"

    async def test_multiple_types(self, engine):
        """Test adding and searching across different context types."""
        mem = await engine.add_memory(
            content="Project uses FastAPI framework",
            category=MemoryCategory.ENTITY,
            tags=["python", "fastapi"],
        )
        res = await engine.add_resource(
            url="https://fastapi.tiangolo.com/",
            content="FastAPI documentation",
        )
        skill = await engine.add_skill(
            tool_name="web_search",
            description="Search the web for information",
        )

        # Search across all types
        results = await engine.find("FastAPI", top_k=10)
        assert len(results) >= 1

    async def test_scope_isolation(self, engine):
        """Test that scope isolation works correctly."""
        # Global entry
        global_entry = await engine.add_memory(
            content="Global knowledge about Python",
            scope=ContextScope.GLOBAL,
        )

        # Agent entry
        agent_entry = await engine.add_memory(
            content="Agent-specific Python preferences",
            scope=ContextScope.AGENT,
            owner_id="agent-001",
        )

        # Search with scope filter
        results = await engine.find("Python", scope=ContextScope.GLOBAL, top_k=10)
        for r in results:
            assert r.entry.scope == ContextScope.GLOBAL

    async def test_count_and_list(self, engine):
        """Test count and list operations."""
        initial_count = await engine.count()

        for i in range(5):
            await engine.add_memory(
                content=f"Test entry {i}",
                scope=ContextScope.GLOBAL,
            )

        new_count = await engine.count()
        assert new_count >= initial_count + 5

        entries = await engine.list_entries(limit=3)
        assert len(entries) <= 3

    async def test_ingest_text_without_llm(self, engine):
        """Test ingest_text falls back to direct ingest without LLM."""
        entries = await engine.ingest_text(
            text="This is some raw text to ingest",
            scope=ContextScope.GLOBAL,
        )
        assert len(entries) >= 1
        assert entries[0].l2_full == "This is some raw text to ingest"
