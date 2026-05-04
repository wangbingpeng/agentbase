"""Tests for LocalFactExtractor — bilingual regex-based fact extraction."""

import pytest

from agentbase_core.ingester.local_fact import LocalFactExtractor


class TestLocalFactExtractorEnglish:
    """Tests for English-language fact extraction."""

    def test_preference_love(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I love hiking in the mountains.")
        assert len(facts) >= 1
        assert facts[0]["category"] == "preference"
        assert "hiking" in facts[0]["content"].lower()

    def test_preference_hate(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I hate spicy food.")
        assert len(facts) >= 1
        assert facts[0]["category"] == "preference"

    def test_preference_favorite(self):
        ext = LocalFactExtractor()
        facts = ext.extract("My favorite color is blue.")
        assert len(facts) >= 1
        pref_facts = [f for f in facts if f["category"] == "preference"]
        assert len(pref_facts) >= 1

    def test_entity_work(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I work at Google.")
        assert len(facts) >= 1
        entity_facts = [f for f in facts if f["category"] == "entity"]
        assert len(entity_facts) >= 1

    def test_entity_live(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I live in San Francisco.")
        assert len(facts) >= 1
        entity_facts = [f for f in facts if f["category"] == "entity"]
        assert len(entity_facts) >= 1

    def test_entity_name(self):
        ext = LocalFactExtractor()
        facts = ext.extract("My name is Alice.")
        assert len(facts) >= 1

    def test_event_went(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I went to Paris last summer.")
        assert len(facts) >= 1
        event_facts = [f for f in facts if f["category"] == "event"]
        assert len(event_facts) >= 1

    def test_event_started(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I started learning Python.")
        assert len(facts) >= 1

    def test_event_switched(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I switched to a new laptop.")
        assert len(facts) >= 1

    def test_empty_text(self):
        ext = LocalFactExtractor()
        assert ext.extract("") == []
        assert ext.extract("   ") == []

    def test_no_facts(self):
        ext = LocalFactExtractor()
        facts = ext.extract("The weather is nice today.")
        # May or may not match patterns; should at least not crash
        assert isinstance(facts, list)

    def test_max_facts_limit(self):
        ext = LocalFactExtractor(max_facts=2)
        text = "I love hiking. I hate spicy food. I work at Google. I live in NYC."
        facts = ext.extract(text)
        assert len(facts) <= 2


class TestLocalFactExtractorChinese:
    """Tests for Chinese-language fact extraction."""

    def test_preference_love(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我喜欢打篮球。")
        assert len(facts) >= 1
        assert facts[0]["category"] == "preference"

    def test_preference_hate(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我讨厌下雨天。")
        assert len(facts) >= 1
        assert facts[0]["category"] == "preference"

    def test_entity_work(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我在阿里巴巴工作。")
        assert len(facts) >= 1
        entity_facts = [f for f in facts if f["category"] == "entity"]
        assert len(entity_facts) >= 1

    def test_entity_live(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我住在北京。")
        assert len(facts) >= 1
        entity_facts = [f for f in facts if f["category"] == "entity"]
        assert len(entity_facts) >= 1

    def test_entity_name(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我的名字叫张三。")
        assert len(facts) >= 1

    def test_event_went(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我去了上海旅游。")
        assert len(facts) >= 1
        event_facts = [f for f in facts if f["category"] == "event"]
        assert len(event_facts) >= 1

    def test_event_joined(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我参加了马拉松比赛。")
        assert len(facts) >= 1

    def test_event_moved(self):
        ext = LocalFactExtractor()
        facts = ext.extract("我搬到了深圳。")
        assert len(facts) >= 1


class TestLocalFactExtractorOutputFormat:
    """Test the output format is compatible with MemoryExtractor."""

    def test_output_has_required_keys(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I love hiking.")
        assert len(facts) >= 1
        fact = facts[0]
        assert "category" in fact
        assert "content" in fact
        assert "tags" in fact
        assert "confidence" in fact

    def test_confidence_range(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I work at Google. I love Python.")
        for fact in facts:
            assert 0 <= fact["confidence"] <= 1

    def test_tags_include_category(self):
        ext = LocalFactExtractor()
        facts = ext.extract("I love hiking.")
        assert len(facts) >= 1
        assert facts[0]["category"] in facts[0]["tags"]

    def test_deduplication(self):
        ext = LocalFactExtractor()
        # Same pattern repeated should be deduplicated
        facts = ext.extract("I love hiking. I love hiking.")
        # Should not have duplicate content-prefix entries
        contents = [f["content"].lower()[:80] for f in facts]
        assert len(contents) == len(set(contents))
