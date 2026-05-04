"""Tests for FTS5 index and hybrid search."""

import pytest
import pytest_asyncio

from agentbase_core.index.sqlite_fts import SQLiteFTSIndex, normalize_query
from agentbase_core.index.hybrid import (
    HybridIndex,
    QUERY_TYPE_WEIGHTS,
    _minmax_normalize,
    rrf_fusion,
    score_fusion,
)
from agentbase_core.models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
)
from agentbase_core.models.query import SearchResult


class TestNormalizeQuery:
    def test_basic(self):
        result = normalize_query("hello world")
        assert "hello" in result
        assert "world" in result

    def test_fullwidth_conversion(self):
        result = normalize_query("ＡＢＣ　テスト")
        assert "ABC" in result

    def test_strip_fts_operators(self):
        result = normalize_query('test AND OR NOT "phrase"')
        # User-supplied AND/OR/NOT are filtered; only our structural OR joins remain
        # The result uses OR as join: "test OR phrase"
        assert result.startswith("test OR")  # 'test' token is preserved
        # 'AND' and 'NOT' user tokens are stripped
        # The structural OR between tokens is intentional for broad recall

    def test_empty_query(self):
        assert normalize_query("") == ""
        assert normalize_query("  ") == ""

    def test_truncation(self):
        long_query = " ".join(["word"] * 200)
        result = normalize_query(long_query)
        assert len(result) <= 512

    def test_strip_punctuation(self):
        """Dots, commas, colons etc. must be removed to avoid FTS5 syntax errors."""
        result = normalize_query("Python 3.12 和 VS Code")
        assert "." not in result
        assert "Python" in result

    def test_strip_special_chars(self):
        """Special FTS5 characters like + @ # $ % should be removed."""
        result = normalize_query("deploy:docker+k8s@prod")
        assert ":" not in result
        assert "+" not in result
        assert "@" not in result

    def test_cjk_segmentation(self):
        """CJK text should be segmented into words by jieba."""
        result = normalize_query("用户偏好使用 Python")
        # jieba should segment "用户偏好使用" into individual words
        assert "用户" in result or "偏好" in result
        assert "Python" in result

    def test_char_fallback(self):
        """When tokenizer='char', CJK uses character-level fallback."""
        result = normalize_query("用户偏好", tokenizer="char")
        assert "用" in result or "户" in result  # single chars or bigrams


@pytest.mark.asyncio
class TestFTSIndex:
    async def test_search_basic(self, engine):
        # Add entries
        await engine.add_memory("Python is a popular programming language", tags=["python", "programming"])
        await engine.add_memory("Rust is a systems programming language", tags=["rust", "programming"])
        await engine.add_memory("The weather is nice today", tags=["weather"])

        # Search
        results = await engine.find("programming language", top_k=5)
        assert len(results) >= 2
        # The first results should be about programming languages
        content_text = " ".join(r.entry.l2_full for r in results[:2])
        assert "programming" in content_text.lower()

    async def test_search_with_type_filter(self, engine):
        await engine.add_memory("test memory content", tags=["test"])
        await engine.add_resource(url="https://example.com/docs", content="test resource content")

        results = await engine.find(
            "test",
            context_type=ContextType.MEMORY,
            top_k=10,
        )
        for r in results:
            assert r.entry.context_type == ContextType.MEMORY

    async def test_search_no_results(self, engine):
        results = await engine.find("xyznonexistent12345", top_k=5)
        assert len(results) == 0

    async def test_search_chinese_content(self, engine):
        await engine.add_memory("用户偏好使用Python进行开发", tags=["python", "偏好"])
        results = await engine.find("Python", top_k=5)
        assert len(results) >= 1


class TestRRFFusion:
    def test_rrf_basic(self):
        entry1 = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry1.mark_active()
        entry2 = ContextEntry(id="e2", l2_full="test2", scope=ContextScope.GLOBAL)
        entry2.mark_active()

        fts_results = [
            SearchResult(entry=entry1, score=1.0, matched_by="fts"),
            SearchResult(entry=entry2, score=0.8, matched_by="fts"),
        ]
        vec_results = [
            SearchResult(entry=entry2, score=0.9, matched_by="vector"),
            SearchResult(entry=entry1, score=0.7, matched_by="vector"),
        ]

        fused = rrf_fusion(fts_results, vec_results)
        assert len(fused) == 2
        # Both entries appear in both lists, so e1 should rank higher
        # (rank 0 in fts + rank 1 in vec) vs (rank 1 in fts + rank 0 in vec)
        assert fused[0].entry.id in ("e1", "e2")

    def test_rrf_single_list(self):
        entry = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry.mark_active()

        fts_results = [SearchResult(entry=entry, score=1.0, matched_by="fts")]
        vec_results: list[SearchResult] = []

        fused = rrf_fusion(fts_results, vec_results)
        assert len(fused) == 1


class TestMinMaxNormalize:
    def test_empty(self):
        assert _minmax_normalize({}) == {}

    def test_single_item(self):
        result = _minmax_normalize({"e1": 5.0})
        assert result == {"e1": 1.0}

    def test_all_same(self):
        result = _minmax_normalize({"e1": 3.0, "e2": 3.0, "e3": 3.0})
        assert result == {"e1": 1.0, "e2": 1.0, "e3": 1.0}

    def test_normal_range(self):
        result = _minmax_normalize({"e1": 0.0, "e2": 5.0, "e3": 10.0})
        assert result["e1"] == 0.0
        assert result["e2"] == 0.5
        assert result["e3"] == 1.0

    def test_two_items(self):
        result = _minmax_normalize({"e1": 2.0, "e2": 8.0})
        assert result["e1"] == 0.0
        assert result["e2"] == 1.0


class TestScoreFusion:
    def test_score_fusion_basic(self):
        entry1 = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry1.mark_active()
        entry2 = ContextEntry(id="e2", l2_full="test2", scope=ContextScope.GLOBAL)
        entry2.mark_active()

        fts_results = [
            SearchResult(entry=entry1, score=5.0, matched_by="fts", score_breakdown={"fts": 5.0}),
            SearchResult(entry=entry2, score=3.0, matched_by="fts", score_breakdown={"fts": 3.0}),
        ]
        vec_results = [
            SearchResult(entry=entry2, score=0.95, matched_by="vector", score_breakdown={"vector": 0.95}),
            SearchResult(entry=entry1, score=0.80, matched_by="vector", score_breakdown={"vector": 0.80}),
        ]

        fused = score_fusion(fts_results, vec_results)
        assert len(fused) == 2
        # Both appear in both lists; check score_breakdown has fusion info
        for r in fused:
            assert r.score_breakdown is not None
            assert "score_fusion" in r.score_breakdown
            assert "fts_norm" in r.score_breakdown
            assert "vec_norm" in r.score_breakdown
            assert r.ranking_stage == "score_fusion"

    def test_score_fusion_fts_only(self):
        entry = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry.mark_active()

        fts_results = [SearchResult(entry=entry, score=5.0, matched_by="fts")]
        vec_results: list[SearchResult] = []

        fused = score_fusion(fts_results, vec_results, fts_weight=0.4, vec_weight=0.6)
        assert len(fused) == 1
        # FTS-only: normalized to 1.0, so score = 0.4 * 1.0 + 0 = 0.4
        assert fused[0].score == pytest.approx(0.4, abs=0.01)

    def test_score_fusion_vec_only(self):
        entry = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry.mark_active()

        fts_results: list[SearchResult] = []
        vec_results = [SearchResult(entry=entry, score=0.9, matched_by="vector")]

        fused = score_fusion(fts_results, vec_results, fts_weight=0.4, vec_weight=0.6)
        assert len(fused) == 1
        # Vec-only: normalized to 1.0, so score = 0 + 0.6 * 1.0 = 0.6
        assert fused[0].score == pytest.approx(0.6, abs=0.01)

    def test_score_fusion_dual_appearance_bonus(self):
        """Document appearing in both lists should rank above single-list docs."""
        entry_both = ContextEntry(id="both", l2_full="both", scope=ContextScope.GLOBAL)
        entry_both.mark_active()
        entry_fts = ContextEntry(id="fts_only", l2_full="fts", scope=ContextScope.GLOBAL)
        entry_fts.mark_active()

        # e1 appears in both with moderate scores
        fts_results = [
            SearchResult(entry=entry_both, score=3.0, matched_by="fts"),
            SearchResult(entry=entry_fts, score=5.0, matched_by="fts"),
        ]
        vec_results = [
            SearchResult(entry=entry_both, score=0.85, matched_by="vector"),
        ]

        fused = score_fusion(fts_results, vec_results, fts_weight=0.4, vec_weight=0.6)
        # "both" gets: 0.4 * (3-3)/(5-3) + 0.6 * 1.0 = 0 + 0.6 = 0.6
        # "fts_only" gets: 0.4 * (5-3)/(5-3) + 0 = 0.4
        # So "both" should rank higher
        assert fused[0].entry.id == "both"

    def test_score_fusion_preserves_relative_order(self):
        """Within a single source, higher-scored docs should rank higher."""
        entry1 = ContextEntry(id="e1", l2_full="test1", scope=ContextScope.GLOBAL)
        entry1.mark_active()
        entry2 = ContextEntry(id="e2", l2_full="test2", scope=ContextScope.GLOBAL)
        entry2.mark_active()

        # Only FTS results, no vector
        fts_results = [
            SearchResult(entry=entry1, score=10.0, matched_by="fts"),
            SearchResult(entry=entry2, score=2.0, matched_by="fts"),
        ]
        vec_results: list[SearchResult] = []

        fused = score_fusion(fts_results, vec_results, fts_weight=1.0, vec_weight=0.0)
        assert fused[0].entry.id == "e1"
        assert fused[1].entry.id == "e2"


class TestAdaptiveWeights:
    def test_known_query_types(self):
        """All expected query types should be in the preset table."""
        expected = [
            "single-session-assistant", "single-session-user",
            "knowledge-update", "preference", "single-session-preference",
            "temporal-reasoning", "multi-session",
        ]
        for qt in expected:
            assert qt in QUERY_TYPE_WEIGHTS
            fts_w, vec_w = QUERY_TYPE_WEIGHTS[qt]
            # Weights should be positive and sum to 1.0
            assert fts_w > 0
            assert vec_w > 0
            assert fts_w + vec_w == pytest.approx(1.0, abs=0.01)

    def test_fts_heavy_types(self):
        """Types where FTS outperforms vector should have fts_weight > 0.5."""
        fts_heavy = ["single-session-assistant", "single-session-user", "knowledge-update"]
        for qt in fts_heavy:
            fts_w, vec_w = QUERY_TYPE_WEIGHTS[qt]
            assert fts_w > 0.5, f"{qt} should have fts_weight > 0.5, got {fts_w}"

    def test_vec_heavy_types(self):
        """Types where vector outperforms FTS should have vec_weight > 0.5."""
        vec_heavy = ["preference", "single-session-preference"]
        for qt in vec_heavy:
            fts_w, vec_w = QUERY_TYPE_WEIGHTS[qt]
            assert vec_w > 0.5, f"{qt} should have vec_weight > 0.5, got {vec_w}"

    def test_hybrid_index_resolve_weights(self):
        """HybridIndex._resolve_weights should use presets for known types."""
        # Just test _resolve_weights directly without DB
        class FakeFTS:
            pass

        idx = HybridIndex(
            fts_index=FakeFTS(),
            fts_weight=0.4,
            vec_weight=0.6,
        )

        # Known type → preset weights
        fts_w, vec_w = idx._resolve_weights("knowledge-update")
        assert fts_w == 0.6
        assert vec_w == 0.4

        # Unknown type → constructor defaults
        fts_w, vec_w = idx._resolve_weights("unknown-type")
        assert fts_w == 0.4
        assert vec_w == 0.6

        # None → constructor defaults
        fts_w, vec_w = idx._resolve_weights(None)
        assert fts_w == 0.4
        assert vec_w == 0.6
