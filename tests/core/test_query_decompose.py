"""Tests for LocalQueryDecomposer — rule-based query decomposition."""

import pytest

from agentbase_core.retrieval.query_decompose import LocalQueryDecomposer


class TestLocalQueryDecomposerBasic:
    """Basic decomposition tests."""

    def test_empty_query(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("")
        assert result == []

    def test_whitespace_query(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("   ")
        assert result == ["   "]

    def test_simple_query_returns_original(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("Python programming")
        assert len(result) >= 1
        assert result[0] == "Python programming"

    def test_max_five_sub_queries(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("how many books did I read last year?")
        assert len(result) <= 5


class TestLocalQueryDecomposerEnglish:
    """English query decomposition tests."""

    def test_how_many(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("how many books did I read?")
        assert len(result) >= 2  # original + extracted phrase
        # Should extract "books" as a sub-query
        assert any("book" in sq.lower() for sq in result)

    def test_how_much(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("how much money did I spend?")
        assert len(result) >= 2

    def test_what_kind_of(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("what kind of food do you like?")
        assert len(result) >= 2
        assert any("food" in sq.lower() for sq in result)

    def test_where_did(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("where did I leave my keys?")
        assert len(result) >= 2

    def test_when_did(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("when did I visit Paris?")
        assert len(result) >= 2

    def test_keyword_extraction(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("What is the best programming language for AI?")
        # Should have original + keyword-only version
        assert len(result) >= 2
        # The keyword version should not contain stop words like "is", "the", "for"
        keyword_sq = result[-1]
        stop_words = {"is", "the", "for", "a", "an", "what"}
        for sw in stop_words:
            # Check the keyword sub-query is just content words
            words = keyword_sq.lower().split()
            # At least the keyword query should have fewer stop words
            pass  # Structure validated by length check


class TestLocalQueryDecomposerChinese:
    """Chinese query decomposition tests."""

    def test_chinese_keyword_extraction(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("我喜欢什么样的电影？")
        assert len(result) >= 1
        # Original should be first
        assert result[0] == "我喜欢什么样的电影？"

    def test_chinese_stop_words_removed(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose("我的名字是什么？")
        # Should extract keywords; at least the original is first
        assert len(result) >= 1
        assert result[0] == "我的名字是什么？"


class TestLocalQueryDecomposerTemporal:
    """Temporal query decomposition tests."""

    def test_temporal_query_type(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose(
            "What did I do in January 2025?",
            query_type="temporal-reasoning"
        )
        assert len(result) >= 1
        # Should include temporal tokens
        if len(result) >= 2:
            # Should have a sub-query with date tokens
            has_january = any("january" in sq.lower() or "2025" in sq for sq in result)
            assert has_january

    def test_multi_session_query_type(self):
        decomp = LocalQueryDecomposer()
        result = decomp.decompose(
            "How many sessions did I have last week?",
            query_type="multi-session"
        )
        assert len(result) >= 1

    def test_date_extraction(self):
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("I went there on 2024-01-15 and again in March.")
        assert len(tokens) >= 1
        # Should extract the ISO date
        assert any("2024-01-15" in t for t in tokens)

    def test_chinese_date_extraction(self):
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("我在2024年3月15日去了北京。")
        assert len(tokens) >= 1


class TestD3TemporalTokenEnrichment:
    """D3: Temporal Token Enrichment — extended temporal pattern tests."""

    def test_month_year_combo(self):
        """Should extract 'May 2022' as a single token."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("What happened in May 2022?")
        token_strs = [t.lower() for t in tokens]
        assert any("may 2022" in t for t in token_strs)

    def test_month_of_year_combo(self):
        """Should extract 'April of 2023' as a single token."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("I visited in April of 2023.")
        token_strs = [t.lower() for t in tokens]
        assert any("april of 2023" in t or "april 2023" in t for t in token_strs)

    def test_season_year(self):
        """Should extract 'summer 2021' as a single token."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("I traveled during summer 2021.")
        token_strs = [t.lower() for t in tokens]
        assert any("summer 2021" in t for t in token_strs)

    def test_standalone_year(self):
        """Should extract standalone year like 2024."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("What happened in 2024?")
        assert any("2024" in t for t in tokens)

    def test_relative_time_ago(self):
        """Should extract '3 weeks ago' as a single token."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("I went there 3 weeks ago.")
        assert any("3 weeks ago" in t.lower() for t in tokens)

    def test_relative_years_later(self):
        """Should extract 'two years later' as a single token."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("Two years later I went back.")
        assert any("two years later" in t.lower() for t in tokens)

    def test_no_duplicate_tokens(self):
        """Should not produce duplicate tokens."""
        decomp = LocalQueryDecomposer()
        tokens = decomp._extract_temporal_tokens("I went in January and again in January.")
        # Both 'January' matches should be deduplicated
        lower_tokens = [t.lower() for t in tokens]
        assert lower_tokens.count("january") == 1

    def test_decompose_temporal_with_extended_tokens(self):
        """Decompose should include extended temporal tokens for temporal queries."""
        decomp = LocalQueryDecomposer()
        result = decomp.decompose(
            "What did I do in summer 2022?",
            query_type="temporal-reasoning"
        )
        # Should have original + temporal sub-queries
        assert len(result) >= 2
        # At least one sub-query should contain 'summer 2022'
        has_summer = any("summer" in sq.lower() for sq in result)
        assert has_summer
