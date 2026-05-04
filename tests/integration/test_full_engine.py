"""Integration tests — full engine lifecycle with Entity, Session, Observability."""

import pytest

from agentbase_core.engine import AgentBaseEngine
from agentbase_core.models.config import AgentBaseConfig
from agentbase_core.models.context_entry import ContextScope, ContextType, MemoryCategory
from agentbase_core.models.entity import Entity, FactTimeline, Relation


@pytest.mark.asyncio
class TestEngineEntityIntegration:
    """Test Entity operations through Engine."""

    async def test_entity_lifecycle(self, full_engine):
        engine = full_engine
        # Add entities
        e1 = await engine.add_entity(
            Entity(name="Python", entity_type="technology", description="Programming language")
        )
        e2 = await engine.add_entity(
            Entity(name="FastAPI", entity_type="framework", description="Web framework")
        )
        assert e1.id
        assert e2.id

        # Find
        found = await engine.find_entities("Python")
        assert len(found) >= 1
        assert found[0].name == "Python"

        # Add alias
        await engine.add_alias(e1.id, "Python3")
        from agentbase_core.graph.entity_service import EntityService
        # Get entity with alias via direct service call
        es = engine.entity_service
        aliases = await es.get_aliases(e1.id)
        assert "Python3" in aliases

        # Add relation
        rel = Relation(source_id=e2.id, target_id=e1.id, predicate="built_with")
        await engine.add_relation(rel)

        # Get relations
        rels = await engine.get_current_relations(e2.id)
        assert len(rels) == 1
        assert rels[0].predicate == "built_with"

        # Graph traversal
        traversal = await engine.graph_traversal("Python", depth=2)
        assert len(traversal) >= 1

    async def test_entity_merge_on_duplicate(self, full_engine):
        engine = full_engine
        """Adding same (name, type) should merge, not duplicate."""
        e1 = await engine.add_entity(
            Entity(name="Django", entity_type="framework", description="Web framework")
        )
        e2 = await engine.add_entity(
            Entity(name="Django", entity_type="framework", description="Web framework v2")
        )
        assert e1.id == e2.id
        # fact_count should increment
        assert e2.fact_count >= 1

    async def test_fact_timeline(self, full_engine):
        engine = full_engine
        e = await engine.add_entity(
            Entity(name="ProjectX", entity_type="project", description="A project")
        )
        fact = FactTimeline(entity_id=e.id, fact="Project started in 2024")
        await engine.add_fact(fact)

        facts = await engine.get_current_facts(e.id)
        assert len(facts) == 1
        assert facts[0].fact == "Project started in 2024"


@pytest.mark.asyncio
class TestEngineSessionIntegration:
    """Test Session operations through Engine."""

    async def test_session_lifecycle(self, full_engine):
        engine = full_engine
        # Create session
        session = await engine.create_session(agent_id="test-agent", project="test-project")
        assert session.id
        assert session.agent_id == "test-agent"
        assert session.project == "test-project"

        # Add messages
        await engine.add_message(session.id, "user", "Hello!")
        await engine.add_message(session.id, "assistant", "Hi! How can I help?")
        await engine.add_message(session.id, "user", "Tell me about Python")

        # Get session
        fetched = await engine.get_session(session.id, load_messages=True)
        assert fetched is not None
        assert len(fetched.messages) == 3
        assert fetched.messages[0].role == "user"
        assert fetched.messages[1].role == "assistant"

        # Commit session (without LLM — just archives)
        memories = await engine.commit_session(session.id, mode="archive_only")
        assert isinstance(memories, list)

    async def test_session_without_messages(self, full_engine):
        engine = full_engine
        session = await engine.create_session(agent_id="empty-agent")
        fetched = await engine.get_session(session.id, load_messages=True)
        assert fetched is not None
        assert len(fetched.messages) == 0


@pytest.mark.asyncio
class TestEngineObservabilityIntegration:
    """Test Observability operations through Engine."""

    async def test_metrics(self, full_engine):
        engine = full_engine
        metrics = await engine.get_metrics()
        assert "active_entries" in metrics
        assert "query_count" in metrics
        assert "avg_latency_ms" in metrics
        assert "pending_jobs" in metrics
        assert isinstance(metrics["active_entries"], int)

    async def test_explain_query(self, full_engine):
        engine = full_engine
        result = await engine.explain_query("test query")
        assert "query" in result
        assert result["query"] == "test query"

    async def test_metrics_after_operations(self, full_engine):
        engine = full_engine
        # Add entries and search
        await engine.add_memory("Test content for metrics", tags=["test"])
        metrics_before = await engine.get_metrics()

        results = await engine.find("test")
        assert len(results) >= 1

        metrics_after = await engine.get_metrics()
        # Active entries should increase
        assert metrics_after["active_entries"] >= metrics_before["active_entries"]


@pytest.mark.asyncio
class TestEngineFullWorkflow:
    """End-to-end workflow: add → entity → session → search → metrics."""

    async def test_full_workflow(self, full_engine):
        engine = full_engine
        # 1. Add memories
        m1 = await engine.add_memory(
            "User prefers Python for backend development",
            category=MemoryCategory.PREFERENCE,
            tags=["python", "backend"],
            confidence=0.95,
            scope=ContextScope.GLOBAL,
        )
        m2 = await engine.add_memory(
            "FastAPI is the preferred web framework",
            category=MemoryCategory.PREFERENCE,
            tags=["fastapi", "web"],
            confidence=0.9,
            scope=ContextScope.AGENT,
            owner_id="backend-agent",
        )
        assert m1.id
        assert m2.id

        # 2. Add entities and relations
        python = await engine.add_entity(
            Entity(name="Python", entity_type="technology", description="Programming language")
        )
        fastapi = await engine.add_entity(
            Entity(name="FastAPI", entity_type="framework", description="Web framework for Python")
        )
        await engine.add_relation(
            Relation(source_id=fastapi.id, target_id=python.id, predicate="built_with")
        )

        # 3. Create session and interact
        session = await engine.create_session(agent_id="backend-agent")
        await engine.add_message(session.id, "user", "What framework should I use for Python?")
        await engine.add_message(session.id, "assistant", "FastAPI is a great choice for Python backend.")

        # 4. Search
        results = await engine.find("Python backend", top_k=5)
        assert len(results) >= 1

        # 5. Graph traversal
        traversal = await engine.graph_traversal("Python", depth=2)
        assert len(traversal) >= 1

        # 6. Metrics
        metrics = await engine.get_metrics()
        assert metrics["active_entries"] >= 2

    async def test_scope_isolation_with_entities(self, full_engine):
        engine = full_engine
        """Entities should be accessible regardless of scope."""
        await engine.add_entity(
            Entity(name="GlobalEntity", entity_type="concept")
        )

        found = await engine.find_entities("GlobalEntity")
        assert len(found) >= 1
