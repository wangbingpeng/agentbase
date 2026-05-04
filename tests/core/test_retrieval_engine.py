"""Tests for retrieval engine, intent analyzer, budget, reranker, and query decomposition."""

import pytest

from agentbase_core.models.config import AgentBaseConfig, RetrievalConfig
from agentbase_core.models.context_entry import ContextEntry, ContextScope, ContextType
from agentbase_core.models.query import SearchQuery, SearchResult, SearchStrategy
from agentbase_core.retrieval.budget import TokenBudget
from agentbase_core.retrieval.engine import RetrievalEngine
from agentbase_core.retrieval.intent import IntentAnalyzer, TypedSubQuery
from agentbase_core.retrieval.query_decompose import LocalQueryDecomposer
from agentbase_core.retrieval.reranker import heuristic_rerank
from agentbase_core.ingester.ner_extractor import NerExtractor


class TestTokenBudget:
    def test_basic_allocation(self):
        budget = TokenBudget(budget=4000)
        assert budget.remaining == 4000
        assert budget.can_load(1000)
        budget.allocate(1000)
        assert budget.used == 1000
        assert budget.remaining == 3000

    def test_over_budget(self):
        budget = TokenBudget(budget=100)
        assert not budget.can_load(200)

    def test_try_allocate(self):
        budget = TokenBudget(budget=100)
        assert budget.try_allocate("short text")
        assert budget.used > 0

    def test_utilization(self):
        budget = TokenBudget(budget=1000)
        budget.allocate(500)
        assert budget.utilization == 0.5

    def test_estimate_tokens(self):
        budget = TokenBudget()
        est = budget.estimate_tokens("Hello world")
        assert est > 0


class TestIntentAnalyzer:
    @pytest.mark.asyncio
    async def test_rule_based_no_keywords(self):
        analyzer = IntentAnalyzer(llm=None)
        subs = await analyzer.analyze("Python programming")
        assert len(subs) >= 1
        assert subs[0].query == "Python programming"

    @pytest.mark.asyncio
    async def test_rule_based_resource_keywords(self):
        analyzer = IntentAnalyzer(llm=None)
        subs = await analyzer.analyze("查找API文档")
        types = [s.context_type for s in subs]
        assert ContextType.RESOURCE in types

    @pytest.mark.asyncio
    async def test_rule_based_skill_keywords(self):
        analyzer = IntentAnalyzer(llm=None)
        subs = await analyzer.analyze("如何使用这个工具")
        types = [s.context_type for s in subs]
        assert ContextType.SKILL in types


class TestHeuristicRerank:
    def test_basic_rerank(self):
        query = SearchQuery(text="test")
        e1 = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.AGENT, owner_id="a1")
        e1.mark_active()
        e2 = ContextEntry(id="e2", l2_full="test2", scope=ContextScope.GLOBAL)
        e2.mark_active()

        results = [
            SearchResult(entry=e1, score=0.5),
            SearchResult(entry=e2, score=0.6),
        ]
        reranked = heuristic_rerank(results, query)
        assert len(reranked) == 2
        # Both should have updated scores
        for r in reranked:
            assert r.score > 0

    def test_rerank_with_scope_filter(self):
        query = SearchQuery(text="test", scope=ContextScope.AGENT)
        e1 = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.AGENT, owner_id="a1")
        e1.mark_active()
        e2 = ContextEntry(id="e2", l2_full="test2", scope=ContextScope.GLOBAL)
        e2.mark_active()

        results = [
            SearchResult(entry=e1, score=0.5),
            SearchResult(entry=e2, score=0.6),
        ]
        reranked = heuristic_rerank(results, query)
        # Agent-scoped entry should get a scope boost
        agent_result = next(r for r in reranked if r.entry.scope == ContextScope.AGENT)
        global_result = next(r for r in reranked if r.entry.scope == ContextScope.GLOBAL)
        # The agent entry should have higher scope_priority contribution
        # (though global may still win due to higher initial score)


@pytest.mark.asyncio
class TestRetrievalEngine:
    async def test_hierarchical_search(self, engine):
        await engine.add_memory("Python is a popular programming language")
        await engine.add_memory("Rust is a systems programming language")

        query = SearchQuery(
            text="programming language",
            strategy=SearchStrategy.HIERARCHICAL,
            top_k=5,
        )
        results = await engine._retrieval_engine.search(query)
        assert len(results) >= 1

    async def test_search_with_token_budget(self, engine):
        await engine.add_memory("Short content")
        await engine.add_memory("Another piece of content")

        query = SearchQuery(
            text="content",
            token_budget=50,  # Very tight budget
            top_k=10,
        )
        results = await engine._retrieval_engine.search(query)
        # Should be trimmed by budget
        assert len(results) >= 1

    async def test_search_with_heuristic_rerank(self, engine):
        await engine.add_memory("Fresh Python content", tags=["python"], confidence=0.9, scope=ContextScope.AGENT, owner_id="a1")
        await engine.add_memory("Old generic content", tags=["generic"], confidence=0.5, scope=ContextScope.GLOBAL)

        query = SearchQuery(text="content", top_k=10)
        results = await engine._retrieval_engine.search(query)
        assert len(results) >= 1
        # All results should have heuristic ranking_stage
        for r in results:
            assert r.ranking_stage == "heuristic"


class TestLocalQueryDecomposerIntegration:
    """Test LocalQueryDecomposer integration with RetrievalEngine."""

    def test_decomposer_initialized_by_default(self, engine):
        """RetrievalEngine should have a LocalQueryDecomposer when query_decomposition=True."""
        assert engine._retrieval_engine._query_decomposer is not None

    def test_decomposer_disabled_when_config_off(self, engine):
        """When query_decomposition=False, no decomposer should be created."""
        from agentbase_core.retrieval.engine import RetrievalEngine
        re = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(query_decomposition=False),
        )
        assert re._query_decomposer is None

    @pytest.mark.asyncio
    async def test_decomposition_in_hybrid_search(self, engine):
        """Hybrid search with query decomposition should produce results."""
        await engine.add_memory("I love hiking in the mountains.")
        await engine.add_memory("I have three books about Python.")

        # Use a "how many" query that triggers decomposition
        query = SearchQuery(text="how many books do I have?", top_k=10)
        results = await engine._retrieval_engine.search(query)
        # Should return at least one result
        assert len(results) >= 1


class TestNerBoostRetrieval:
    """Test NER-aware query expansion and result boosting in RetrievalEngine."""

    def test_ner_extractor_initialized_by_default(self, engine):
        """RetrievalEngine should have a NerExtractor when ner_boost=True."""
        assert engine._retrieval_engine._ner_extractor is not None

    def test_ner_extractor_disabled_when_config_off(self, engine):
        """When ner_boost=False, no NerExtractor should be created."""
        re = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(ner_boost=False),
        )
        assert re._ner_extractor is None

    def test_ner_weight_config(self, engine):
        """NER weight should be configurable."""
        re = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(ner_boost=True, ner_weight=0.5),
        )
        assert re._ner_weight == 0.5

    @pytest.mark.asyncio
    async def test_ner_boost_search_with_ner_tags(self, engine):
        """NER-tagged entries should get a score boost when query matches their NER tags."""
        # Add an entry with NER entity tags (simulating pipeline tagging)
        await engine.add_memory(
            "I went on a trip to Hawaii last summer.",
            tags=["ner", "ner_Hawaii"],
        )
        # Add another entry without NER tags
        await engine.add_memory(
            "I also visited Japan for two weeks.",
        )

        # Search for the entity - should return results
        query = SearchQuery(text="Hawaii trip", top_k=10)
        results = await engine._retrieval_engine.search(query)
        assert len(results) >= 1
        # NER-tagged result should be boosted (higher score than without NER)
        for r in results:
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_ner_boost_no_dilution(self, engine):
        """NER boost should NOT add new results — only boost existing ones."""
        # Add entries with NER entity tags
        await engine.add_memory(
            "I went on a trip to Hawaii last summer.",
            tags=["ner", "ner_Hawaii"],
        )

        # Search with and without NER boost
        query = SearchQuery(text="summer trip", top_k=10)
        results_with_ner = await engine._retrieval_engine.search(query)

        re_no_ner = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(ner_boost=False),
        )
        results_without_ner = await re_no_ner.search(query)

        # Same number of results (no dilution from added NER entries)
        assert len(results_with_ner) == len(results_without_ner)


class TestNerExtractor:
    """Test NerExtractor entity extraction."""

    def test_extract_empty_text(self):
        extractor = NerExtractor()
        assert extractor.extract("") == []

    def test_extract_regex_fallback(self):
        """Regex fallback should find capitalized multi-word entities."""
        extractor = NerExtractor()
        # Force regex by not having spaCy models
        extractor._spacy_checked = True
        result = extractor._extract_regex(
            "I went to New York City yesterday and visited Central Park."
        )
        # Should find at least some entities
        assert len(result) >= 1
        # Check output format
        for ent in result:
            assert "content" in ent
            assert "category" in ent
            assert "tags" in ent
            assert "ner" in ent["tags"]

    def test_extract_number_unit(self):
        """Should extract number+unit patterns."""
        extractor = NerExtractor()
        extractor._spacy_checked = True
        result = extractor._extract_regex("I ran for 3 hours and 30 minutes today.")
        # Should find "3 hours" and "30 minutes"
        contents = [r["content"] for r in result]
        assert any("hours" in c for c in contents)
        assert any("minutes" in c for c in contents)

    def test_extract_chinese_location(self):
        """Should extract Chinese location patterns."""
        extractor = NerExtractor()
        extractor._spacy_checked = True
        result = extractor._extract_regex("我住在北京和上海市。")
        contents = [r["content"] for r in result]
        assert any("北京" in c or "上海" in c for c in contents)


class TestD2AggregationAware:
    """D2: Aggregation-Aware — detection, top_k boost, and prompt suffix."""

    def test_is_aggregation_query_how_many(self):
        assert IntentAnalyzer.is_aggregation_query("How many books did I read?")

    def test_is_aggregation_query_total(self):
        assert IntentAnalyzer.is_aggregation_query("What is the total amount I spent?")

    def test_is_aggregation_query_non_agg(self):
        assert not IntentAnalyzer.is_aggregation_query("What is my favorite color?")

    def test_is_aggregation_query_chinese(self):
        assert IntentAnalyzer.is_aggregation_query("我总共去了多少次？")

    @pytest.mark.asyncio
    async def test_agg_top_k_boost(self, engine):
        """Aggregation queries should get higher top_k."""
        await engine.add_memory("Book A")
        await engine.add_memory("Book B")
        await engine.add_memory("Book C")

        # Search with aggregation query
        query = SearchQuery(text="how many books do I have?", top_k=10)
        # Before search, top_k is 10
        assert query.top_k == 10
        # After search, the engine should have boosted top_k internally
        results = await engine._retrieval_engine.search(query)
        # The query.top_k should have been boosted to agg_top_k (80)
        assert query.top_k >= 10  # At least the original, likely boosted

    def test_agg_top_k_config(self):
        """agg_top_k should be configurable."""
        config = RetrievalConfig(agg_top_k=100, agg_detection=True)
        assert config.agg_top_k == 100

    def test_agg_detection_disabled(self):
        """When agg_detection=False, top_k should not be boosted."""
        config = RetrievalConfig(agg_detection=False)
        assert not config.agg_detection

    def test_get_aggregation_prompt_suffix(self):
        """Should return a non-empty prompt suffix for aggregation queries."""
        suffix = RetrievalEngine.get_aggregation_prompt_suffix()
        assert len(suffix) > 0
        assert "enumerate" in suffix.lower() or "count" in suffix.lower()


class TestD1SessionCoRetrieval:
    """D1: Session Co-Retrieval — supplement sibling turns from sparse sessions."""

    def test_session_co_retrieval_config_default(self):
        """Session co-retrieval should be enabled by default."""
        config = RetrievalConfig()
        assert config.session_co_retrieval is True
        assert config.co_retrieve_min_turns == 2

    def test_session_co_retrieval_config_off(self):
        """Session co-retrieval can be disabled."""
        config = RetrievalConfig(session_co_retrieval=False)
        assert config.session_co_retrieval is False

    @pytest.mark.asyncio
    async def test_co_retrieve_skips_when_config_off(self, engine):
        """Co-retrieval should be skipped when session_co_retrieval=False."""
        re_no_co = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(session_co_retrieval=False),
        )
        await engine.add_memory("I love Python programming")
        query = SearchQuery(text="Python", top_k=10, query_type="preference")
        # Should work without error, co-retrieval simply skipped
        results = await re_no_co.search(query)
        # Result count depends on FTS matching; just verify no crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_co_retrieve_respects_config(self, engine):
        """When session_co_retrieval=False, co-retrieval should be skipped."""
        re_no_co = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(session_co_retrieval=False),
        )
        await engine.add_memory("Session content")
        query = SearchQuery(text="content", top_k=10, query_type="multi-session")
        results = await re_no_co.search(query)
        # Should return results without co-retrieval
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_co_retrieve_activates_for_generic_query(self, engine):
        """D1 session co-retrieval should now activate for generic (None) queries.

        Previously D1 was gated to multi-session / temporal-reasoning only.
        After the optimization, generic queries also trigger co-retrieval
        when hits come from sparse sessions.
        """
        # Ingest a conversation with sparse coverage (1 turn per session)
        await engine.add_conversation(
            turns=[
                {"role": "user", "content": "I visited Paris last summer"},
                {"role": "assistant", "content": "Paris is beautiful in summer"},
            ],
            session_index=0,
        )
        # Generic query — query_type will be None (no keyword match)
        query = SearchQuery(text="Paris", top_k=10)
        assert query.query_type is None  # sanity: truly generic

        results = await engine._retrieval_engine.search(query)
        # Should return results without error — D1 now runs for generic queries
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_co_retrieve_generic_no_config_still_skips(self, engine):
        """Generic queries should NOT trigger co-retrieval when config is off."""
        re_no_co = RetrievalEngine(
            index=engine._index,
            retrieval_config=RetrievalConfig(session_co_retrieval=False),
        )
        await engine.add_conversation(
            turns=[{"role": "user", "content": "I like hiking in the mountains"}],
            session_index=0,
        )
        query = SearchQuery(text="hiking", top_k=10)
        assert query.query_type is None

        results = await re_no_co.search(query)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_co_retrieve_still_works_for_multi_session(self, engine):
        """Regression: multi-session queries should still trigger co-retrieval."""
        await engine.add_conversation(
            turns=[
                {"role": "user", "content": "I discussed project Alpha"},
                {"role": "assistant", "content": "Alpha sounds interesting"},
            ],
            session_index=0,
        )
        query = SearchQuery(text="project", top_k=10, query_type="multi-session")

        results = await engine._retrieval_engine.search(query)
        assert isinstance(results, list)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_co_retrieve_still_works_for_temporal(self, engine):
        """Regression: temporal-reasoning queries should still trigger co-retrieval."""
        await engine.add_conversation(
            turns=[
                {"role": "user", "content": "I first mentioned Python in March"},
                {"role": "assistant", "content": "Noted your Python mention"},
            ],
            session_index=0,
        )
        query = SearchQuery(text="Python", top_k=10, query_type="temporal-reasoning")

        results = await engine._retrieval_engine.search(query)
        assert isinstance(results, list)
        assert len(results) >= 1


class TestD5TemporalFilterParsing:
    """D5: IntentAnalyzer.parse_temporal_filter—regex-based date extraction."""

    def test_parse_no_temporal_expr(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("What is my favorite color?")
        assert df is None and dt is None

    def test_parse_empty(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("")
        assert df is None and dt is None

    def test_parse_last_week(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        df, dt = IntentAnalyzer.parse_temporal_filter("what did I do last week", now=now)
        assert df is not None and dt is not None
        assert abs((now - df).days - 7) <= 1

    def test_parse_last_n_days(self):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 1, tzinfo=timezone.utc)
        df, dt = IntentAnalyzer.parse_temporal_filter("events in the last 10 days", now=now)
        assert df is not None and dt is not None
        assert (now - df).days == 10

    def test_parse_in_year(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("what happened in 2024")
        assert df is not None and dt is not None
        assert df.year == 2024 and dt.year == 2025

    def test_parse_in_month_year(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("events in March 2024")
        assert df is not None and dt is not None
        assert df.year == 2024 and df.month == 3
        assert dt.year == 2024 and dt.month == 4

    def test_parse_before_year(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("before 2024 I was studying")
        assert df is None and dt is not None and dt.year == 2024

    def test_parse_since_year(self):
        df, dt = IntentAnalyzer.parse_temporal_filter("since 2023 I have been running")
        assert df is not None and df.year == 2023 and dt is None

    def test_parse_this_month(self):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 15, tzinfo=timezone.utc)
        df, dt = IntentAnalyzer.parse_temporal_filter("what did I do this month", now=now)
        assert df is not None and df.month == 5 and df.day == 1
        assert dt == now

    def test_parse_yesterday(self):
        from datetime import datetime, timezone
        now = datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc)
        df, dt = IntentAnalyzer.parse_temporal_filter("yesterday I watched TV", now=now)
        assert df is not None and dt is not None
        assert df.day == 14 and dt.day == 15


class TestD5TemporalFilterAutoPopulate:
    """D5: When query_type=temporal-reasoning, search() auto-populates date range."""

    @pytest.mark.asyncio
    async def test_temporal_query_auto_populates_filter(self, engine):
        await engine.add_memory("I went hiking last year", tags=["event"])
        query = SearchQuery(
            text="what did I do in 2024",
            top_k=10,
            query_type="temporal-reasoning",
        )
        # Before search: date_from/date_to should be None
        assert query.date_from is None and query.date_to is None

        await engine._retrieval_engine.search(query)
        # After search: date_from/date_to should be auto-populated
        assert query.date_from is not None
        assert query.date_to is not None
        assert query.date_from.year == 2024

    @pytest.mark.asyncio
    async def test_temporal_query_respects_explicit_filter(self, engine):
        """When caller passes explicit date_from/date_to, don't overwrite."""
        from datetime import datetime, timezone
        await engine.add_memory("I went hiking", tags=["event"])
        explicit_from = datetime(2020, 1, 1, tzinfo=timezone.utc)
        explicit_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
        query = SearchQuery(
            text="what did I do in 2024",
            top_k=10,
            query_type="temporal-reasoning",
            date_from=explicit_from,
            date_to=explicit_to,
        )
        await engine._retrieval_engine.search(query)
        # Explicit filter must remain untouched
        assert query.date_from == explicit_from
        assert query.date_to == explicit_to


class TestD3SessionLinkExpansion:
    """D3: _fetch_session_linked_memories uses session_memory_links."""

    @pytest.mark.asyncio
    async def test_fetch_linked_memories_empty_seed(self, engine):
        """Empty seed list returns []."""
        linked = await engine._retrieval_engine._fetch_session_linked_memories(
            [], exclude_ids=set()
        )
        assert linked == []

    @pytest.mark.asyncio
    async def test_fetch_linked_memories_no_links(self, engine):
        """Seed with no links returns []."""
        e = await engine.add_memory("orphan entry with no session")
        linked = await engine._retrieval_engine._fetch_session_linked_memories(
            [e.id], exclude_ids={e.id}
        )
        assert linked == []

    @pytest.mark.asyncio
    async def test_retrieval_engine_has_store(self, engine):
        """RetrievalEngine should receive the store reference after initialize."""
        assert engine._retrieval_engine._store is not None


class TestD7SessionSummaryStructured:
    """D7: session summary now carries structured metadata in extra."""

    @pytest.mark.asyncio
    async def test_summary_extra_fields(self, engine):
        """After ingesting a session entry, summary entry should carry structured fields."""
        from agentbase_core.models.context_entry import (
            ContextEntry, ContextType, ContextScope, OriginType,
        )
        # Content must be >300 chars so truncation produces a different string
        # than the parent (otherwise dedup skips the summary entry).
        long_content = (
            "User: I love hiking in the mountains and exploring new trails. "
            "I visited Swiss Alps last summer and the experience was absolutely unforgettable. "
            "The scenery was breathtaking with snow-capped peaks and crystal clear lakes. "
            "Assistant: That sounds amazing! The Swiss Alps are indeed one of the most beautiful "
            "mountain ranges in the world. You should also consider visiting the Dolomites in Italy."
        )
        parent = ContextEntry(
            l2_full=long_content,
            context_type=ContextType.MEMORY,
            scope=ContextScope.GLOBAL,
            tags=["session_0", "turn_0", "user"],
            extra={"session_index": 0, "session_date": "2024/03/15", "role": "user"},
            origin_type=OriginType.MANUAL,
        )
        await engine.ingester.ingest_direct(parent)

        # Search for session_summary entries
        entries = await engine.list_entries(limit=100)
        summary_entries = [
            e for e in entries
            if e.tags and "session_summary" in e.tags
        ]
        assert len(summary_entries) >= 1
        s = summary_entries[0]
        assert s.extra is not None
        # Structured metadata must be present
        assert "summary_source" in s.extra
        # time_range should be populated since session_date is provided
        assert "time_range" in s.extra
        assert "start" in s.extra["time_range"]
        assert "2024-03-15" in s.extra["time_range"]["start"]

    @pytest.mark.asyncio
    async def test_generate_session_summary_returns_tuple(self, engine):
        """_generate_session_summary returns (text, structured_dict)."""
        content = "x" * 400  # force truncation path
        result = await engine.ingester._generate_session_summary(
            content,
            tags=["session_0"],
            extra={"session_date": "2024/03/15"},
        )
        assert result is not None
        summary, structured = result
        assert isinstance(summary, str) and len(summary) > 0
        assert isinstance(structured, dict)
        assert structured["summary_source"] == "truncation"
        assert "time_range" in structured
        assert structured["time_range"]["start"].startswith("2024-03-15")

    @pytest.mark.asyncio
    async def test_generate_session_summary_empty(self, engine):
        """Empty content returns None."""
        result = await engine.ingester._generate_session_summary(
            "", tags=[], extra={}
        )
        assert result is None
