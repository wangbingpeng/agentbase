"""Tests for NerExtractor — bilingual NER entity extraction."""

import pytest

from agentbase_core.ingester.ner_extractor import NerExtractor


class TestNerExtractorRegexFallback:
    """Tests for regex-based NER extraction (the guaranteed path when spaCy is absent)."""

    def test_english_capitalized_phrase(self):
        ext = NerExtractor()
        # Force regex fallback by clearing any loaded spaCy model
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex("I visited San Francisco last week with John Smith.")
        # Should find at least "San Francisco"
        assert len(entities) >= 1
        contents = [e["content"] for e in entities]
        assert any("San Francisco" in c for c in contents)

    def test_english_quoted_name(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex('My dog is named "Buddy".')
        assert len(entities) >= 1
        contents = [e["content"] for e in entities]
        assert any("Buddy" in c for c in contents)

    def test_english_number_unit(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex("I ran for 3 hours yesterday.")
        assert len(entities) >= 1
        contents = [e["content"] for e in entities]
        assert any("3 hours" in c for c in contents)

    def test_chinese_location(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex("我去了北京市和上海市旅游。")
        assert len(entities) >= 1
        contents = [e["content"] for e in entities]
        # Should detect at least one of the cities
        has_location = any("市" in c or "省" in c for c in contents)
        assert has_location

    def test_chinese_org(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex("我在清华大学和北京大学都读过书。")
        assert len(entities) >= 1
        contents = [e["content"] for e in entities]
        has_org = any("大学" in c for c in contents)
        assert has_org

    def test_empty_text(self):
        ext = NerExtractor()
        assert ext.extract("") == []
        assert ext.extract("   ") == []

    def test_max_entities_limit(self):
        ext = NerExtractor(max_entities=2)
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext.extract(
            "I visited San Francisco and New York and Los Angeles and Chicago."
        )
        assert len(entities) <= 2


class TestNerExtractorOutputFormat:
    """Test output format is compatible with LocalFactExtractor."""

    def test_output_has_required_keys(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext.extract("I visited San Francisco.")
        assert len(entities) >= 1
        ent = entities[0]
        assert "category" in ent
        assert "content" in ent
        assert "tags" in ent
        assert "confidence" in ent

    def test_tags_include_ner(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext.extract("I visited San Francisco.")
        assert len(entities) >= 1
        assert "ner" in entities[0]["tags"]

    def test_confidence_range(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext.extract("I visited San Francisco and worked at Google.")
        for ent in entities:
            assert 0 <= ent["confidence"] <= 1

    def test_deduplication(self):
        ext = NerExtractor()
        ext._spacy_checked = True
        ext._spacy_nlp_en = None
        ext._spacy_nlp_zh = None
        entities = ext._extract_regex("San Francisco is great. San Francisco is beautiful.")
        # Should not have duplicates
        contents = [e["content"].lower() for e in entities]
        assert len(contents) == len(set(contents))
