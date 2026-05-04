"""NerExtractor — bilingual NER entity extraction (spaCy + regex fallback)."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex-based NER patterns (fallback when spaCy is unavailable)
# ---------------------------------------------------------------------------

# English: capitalized consecutive words (not sentence-initial position)
_RE_EN_CAPS = re.compile(
    r"(?<![.!?]\s)"          # not after sentence-ending punctuation + space
    r"(?<!^)"                # not at start of line
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # 2+ capitalized words
)

# English: quoted names
_RE_EN_QUOTED = re.compile(r'["\']([A-Z][a-z]+(?:\s+\w+)?)["\']')

# English: number + unit patterns
_RE_EN_NUM_UNIT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:hours?|days?|weeks?|months?|years?|minutes?|"
    r"dollars?|cents?|miles?|km|kg|lbs?|GB|MB|TB|Mbps|Hz)\b",
    re.IGNORECASE,
)

# Chinese: location/org patterns
_RE_ZH_LOCATION = re.compile(
    r"([\u4e00-\u9fff]{2,6}(?:市|省|区|县|镇|村|街|路|州|国|岛|山|湖|河|海|湾))"
)
_RE_ZH_ORG = re.compile(
    r"([\u4e00-\u9fff]{2,8}(?:公司|大学|学院|医院|银行|集团|机构|部门|中心|研究所|实验室))"
)

# NER entity type mapping
_SPACY_TO_CATEGORY = {
    "PERSON": "entity",
    "ORG": "entity",
    "GPE": "entity",      # Geo-Political Entity
    "LOC": "entity",
    "PRODUCT": "entity",
    "EVENT": "event",
    "DATE": "event",
    "NORP": "entity",     # Nationalities/Religious/Political groups
    "FAC": "entity",      # Facility
    "WORK_OF_ART": "entity",
}


class NerExtractor:
    """Bilingual NER entity extraction — spaCy preferred, regex fallback.

    Degradation chain: spaCy (en) -> spaCy (zh) -> regex -> skip

    Output format matches ``LocalFactExtractor``:
    ``[{"category": "entity", "content": "...", "tags": ["ner"], "confidence": 0.8}]``
    """

    def __init__(self, max_entities: int = 10) -> None:
        self._max_entities = max_entities
        self._spacy_nlp_en: Any = None
        self._spacy_nlp_zh: Any = None
        self._spacy_checked = False

    def _try_load_spacy(self) -> None:
        """Lazily attempt to load spaCy models."""
        if self._spacy_checked:
            return
        self._spacy_checked = True

        try:
            import spacy  # type: ignore
        except ImportError:
            logger.debug("spaCy not installed, NER will use regex fallback")
            return

        # Try English model first
        for model_name in ("en_core_web_sm", "en_core_web_md"):
            try:
                self._spacy_nlp_en = spacy.load(model_name)
                logger.info(f"Loaded spaCy NER model: {model_name}")
                break
            except OSError:
                continue

        # Try Chinese model
        for model_name in ("zh_core_web_sm", "zh_core_web_md"):
            try:
                self._spacy_nlp_zh = spacy.load(model_name)
                logger.info(f"Loaded spaCy NER model: {model_name}")
                break
            except OSError:
                continue

        if self._spacy_nlp_en is None and self._spacy_nlp_zh is None:
            logger.debug("No spaCy NER models available, will use regex fallback")

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from *text*.

        Returns
        -------
        list[dict]
            Each dict has keys: ``category``, ``content``, ``tags``, ``confidence``.
        """
        if not text or not text.strip():
            return []

        self._try_load_spacy()

        # Try spaCy first
        entities = self._extract_spacy(text)
        if entities:
            return entities[: self._max_entities]

        # Fallback to regex
        entities = self._extract_regex(text)
        return entities[: self._max_entities]

    def _extract_spacy(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using spaCy NER."""
        entities: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Detect language heuristically: if >30% CJK characters, use zh model
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        is_chinese = cjk_count / max(len(text), 1) > 0.3

        nlp = self._spacy_nlp_zh if (is_chinese and self._spacy_nlp_zh) else self._spacy_nlp_en

        if nlp is None:
            return []

        try:
            doc = nlp(text[:5000])  # Truncate to avoid excessive processing
        except Exception as e:
            logger.debug(f"spaCy NER failed: {e}")
            return []

        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) < 2:
                continue

            key = entity_text.lower()
            if key in seen:
                continue
            seen.add(key)

            category = _SPACY_TO_CATEGORY.get(ent.label_, "entity")
            entities.append({
                "category": category,
                "content": entity_text,
                "tags": ["ner", ent.label_.lower()],
                "confidence": 0.85,
            })

        return entities

    def _extract_regex(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using regex patterns (fallback)."""
        entities: list[dict[str, Any]] = []
        seen: set[str] = set()

        # English: capitalized phrases
        for m in _RE_EN_CAPS.finditer(text):
            entity = m.group(1).strip()
            key = entity.lower()
            if key not in seen and len(entity) > 3:
                seen.add(key)
                entities.append({
                    "category": "entity",
                    "content": entity,
                    "tags": ["ner", "regex"],
                    "confidence": 0.6,
                })

        # English: quoted names
        for m in _RE_EN_QUOTED.finditer(text):
            entity = m.group(1).strip()
            key = entity.lower()
            if key not in seen and len(entity) > 2:
                seen.add(key)
                entities.append({
                    "category": "entity",
                    "content": entity,
                    "tags": ["ner", "regex"],
                    "confidence": 0.7,
                })

        # English: number + unit
        for m in _RE_EN_NUM_UNIT.finditer(text):
            entity = m.group(0).strip()
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "category": "event",
                    "content": entity,
                    "tags": ["ner", "regex", "quantity"],
                    "confidence": 0.7,
                })

        # Chinese: locations
        for m in _RE_ZH_LOCATION.finditer(text):
            entity = m.group(1).strip()
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "category": "entity",
                    "content": entity,
                    "tags": ["ner", "regex", "location"],
                    "confidence": 0.7,
                })

        # Chinese: organizations
        for m in _RE_ZH_ORG.finditer(text):
            entity = m.group(1).strip()
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                entities.append({
                    "category": "entity",
                    "content": entity,
                    "tags": ["ner", "regex", "org"],
                    "confidence": 0.7,
                })

        return entities
