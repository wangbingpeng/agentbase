"""Tests for EntityService, SessionService, and ObservabilityService."""

import pytest

from agentbase_core.graph.conflict import ConflictResolution, ConflictResolver
from agentbase_core.graph.entity_service import EntityService
from agentbase_core.models.entity import Entity, Relation
from agentbase_core.models.session import Session, SessionMessage
from agentbase_core.session.session_service import SessionService, SessionCompressor
from agentbase_core.observability.observability_service import ContextMetrics, DebugService


@pytest.mark.asyncio
class TestEntityService:
    async def test_add_and_get_entity(self, engine):
        entity_svc = EntityService(engine._pool)
        entity = Entity(name="Python", entity_type="concept", description="A programming language")
        result = await entity_svc.add_entity(entity)
        assert result.name == "Python"

        fetched = await entity_svc.get_entity(result.id)
        assert fetched is not None
        assert fetched.name == "Python"

    async def test_entity_merge_on_duplicate(self, engine):
        entity_svc = EntityService(engine._pool)
        e1 = Entity(name="Rust", entity_type="concept", description="Systems language")
        r1 = await entity_svc.add_entity(e1)
        assert r1.fact_count == 0

        # Add same name/type again — should merge
        e2 = Entity(name="Rust", entity_type="concept", description="Better desc")
        r2 = await entity_svc.add_entity(e2)
        assert r2.id == r1.id  # Same entity, merged
        assert r2.fact_count == 1

    async def test_add_alias(self, engine):
        entity_svc = EntityService(engine._pool)
        entity = Entity(name="JavaScript", entity_type="concept")
        result = await entity_svc.add_entity(entity)
        await entity_svc.add_alias(result.id, "JS")

        aliases = await entity_svc.get_aliases(result.id)
        assert "JS" in aliases

    async def test_add_relation(self, engine):
        entity_svc = EntityService(engine._pool)
        e1 = await entity_svc.add_entity(Entity(name="User1", entity_type="person"))
        e2 = await entity_svc.add_entity(Entity(name="Python", entity_type="concept"))

        rel = Relation(source_id=e1.id, target_id=e2.id, predicate="uses", confidence=0.9)
        result = await entity_svc.add_relation(rel)
        assert result.predicate == "uses"

    async def test_get_current_relations(self, engine):
        entity_svc = EntityService(engine._pool)
        e1 = await entity_svc.add_entity(Entity(name="User2", entity_type="person"))
        e2 = await entity_svc.add_entity(Entity(name="FastAPI", entity_type="tool"))
        await entity_svc.add_relation(Relation(source_id=e1.id, target_id=e2.id, predicate="works_with"))

        rels = await entity_svc.get_current_relations(e1.id)
        assert len(rels) >= 1

    async def test_graph_traversal(self, engine):
        entity_svc = EntityService(engine._pool)
        e1 = await entity_svc.add_entity(Entity(name="GraphTest_A", entity_type="concept"))
        e2 = await entity_svc.add_entity(Entity(name="GraphTest_B", entity_type="concept"))
        await entity_svc.add_relation(Relation(source_id=e1.id, target_id=e2.id, predicate="related_to"))

        neighbors = await entity_svc.graph_traversal("GraphTest_A", depth=1)
        assert len(neighbors) >= 1


class TestConflictResolver:
    def test_rule_based_replace(self):
        result = ConflictResolver.resolve_by_rule("Python 3.10", "Python 3.10 is the version used")
        assert result == ConflictResolution.REPLACE

    def test_rule_based_skip(self):
        result = ConflictResolver.resolve_by_rule("same text", "same text")
        assert result == ConflictResolution.SKIP

    def test_rule_based_merge(self):
        result = ConflictResolver.resolve_by_rule("Uses Python", "Prefers dark mode")
        assert result == ConflictResolution.MERGE


@pytest.mark.asyncio
class TestSessionService:
    async def test_create_session(self, engine):
        session_svc = SessionService(engine._pool)
        session = await session_svc.create_session(agent_id="test-agent", project="test-project")
        assert session.id
        assert session.agent_id == "test-agent"
        assert session.status == "active"

    async def test_add_message(self, engine):
        session_svc = SessionService(engine._pool)
        session = await session_svc.create_session(agent_id="test-agent")
        msg = await session_svc.add_message(session.id, "user", "Hello!")
        assert msg.content == "Hello!"
        assert msg.role == "user"

    async def test_get_session_with_messages(self, engine):
        session_svc = SessionService(engine._pool)
        session = await session_svc.create_session(agent_id="test-agent")
        await session_svc.add_message(session.id, "user", "Question")
        await session_svc.add_message(session.id, "assistant", "Answer")

        fetched = await session_svc.get_session(session.id, load_messages=True)
        assert fetched is not None
        assert len(fetched.messages) == 2

    async def test_commit_session_without_llm(self, engine):
        session_svc = SessionService(engine._pool, llm=None)
        session = await session_svc.create_session(agent_id="test-agent")
        await session_svc.add_message(session.id, "user", "Question")
        await session_svc.add_message(session.id, "assistant", "Answer")

        # Commit without LLM — should archive but no memory extraction
        extracted = await session_svc.commit_session(session.id, mode="archive_only")
        assert isinstance(extracted, list)

        # Session should be archived
        fetched = await session_svc.get_session(session.id)
        assert fetched.status == "archived"


class TestSessionCompressor:
    def test_split_into_turns(self):
        messages = [
            SessionMessage(role="user", content="Q1"),
            SessionMessage(role="assistant", content="A1"),
            SessionMessage(role="user", content="Q2"),
            SessionMessage(role="assistant", content="A2"),
        ]
        turns = SessionCompressor._split_into_turns(messages)
        assert len(turns) == 2

    def test_format_turns(self):
        turns = [[SessionMessage(role="user", content="Hi")]]
        text = SessionCompressor._format_turns(turns)
        assert "[user]: Hi" in text


@pytest.mark.asyncio
class TestObservability:
    async def test_metrics(self, engine):
        metrics_svc = ContextMetrics(engine._pool)
        metrics = await metrics_svc.get_metrics()
        assert "query_count" in metrics
        assert "active_entries" in metrics
        assert "pending_jobs" in metrics

    async def test_debug_explain(self, engine):
        debug_svc = DebugService(engine._pool)
        result = await debug_svc.explain_query("test query")
        assert result["query"] == "test query"
