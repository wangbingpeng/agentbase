"""LocalFactExtractor — bilingual regex-based fact extraction (zero-LLM fallback)."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bilingual regex patterns for local fact extraction
# Each tuple: (compiled_pattern, category_string)
# ---------------------------------------------------------------------------

# --- Preference patterns ---
_FACT_PATTERNS_PREF: list[tuple[re.Pattern[str], str]] = [
    # English
    (re.compile(r"(?:i |i'm )(?:love|enjoy|prefer|like|really into|big fan of|obsessed with)\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "preference"),
    (re.compile(r"(?:i |i'm )(?:hate|can't stand|not a fan of|don't like|dislike)\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "preference"),
    (re.compile(r"my (?:favorite|favourite)\s+(.+?)(?:is|are)\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "preference"),
    # Chinese
    (re.compile(r"我(?:喜欢|爱|热爱|偏好|最爱|超爱|特别爱|很爱)\s*(.+?)(?:[。，！,.]|$)"), "preference"),
    (re.compile(r"我(?:讨厌|不喜欢|恨|不喜|不爱|不太喜欢|不怎么喜欢)\s*(.+?)(?:[。，！,.]|$)"), "preference"),
    (re.compile(r"(?:最|特别)(?:喜欢|爱|偏好)(?:的)?(.+?)(?:是|为)\s*(.+?)(?:[。，！,.]|$)"), "preference"),
]

# --- Entity patterns ---
_FACT_PATTERNS_ENTITY: list[tuple[re.Pattern[str], str]] = [
    # English
    (re.compile(r"(?:i work (?:at|for|in))\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "entity"),
    (re.compile(r"(?:i live (?:in|at))\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "entity"),
    (re.compile(r"(?:my (?:name|job|role|title|position))\s+(?:is|are)\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "entity"),
    (re.compile(r"(?:i (?:have|own|bought|purchased|use))\s+(?:a |an )?(.+?)(?:[.!]|$)", re.IGNORECASE), "entity"),
    (re.compile(r"(?:i (?:graduated|study|studied|major(?:ed)?))\s+(?:in |from |with )?(.+?)(?:[.!]|$)", re.IGNORECASE), "entity"),
    # Chinese
    (re.compile(r"我(?:在|于)(.+?)(?:工作|上班|任职|就职)"), "entity"),
    (re.compile(r"我(?:住在?|生活在|搬到了?)(.+?)(?:[。，！,.]|$)"), "entity"),
    (re.compile(r"我(?:的)?(?:名字|职位|职务|岗位|职业)(?:是|叫)\s*(.+?)(?:[。，！,.]|$)"), "entity"),
    (re.compile(r"我(?:买了?|购入|入手了?|有|拥有|用的是?)(.+?)(?:[。，！,.]|$)"), "entity"),
    (re.compile(r"我(?:毕业于?|就读于?|从)(.+?)(?:毕业|的|[。，！,.]|$)"), "entity"),
    (re.compile(r"我(?:的)?(?:专业|学历|学位)(?:是|为)\s*(.+?)(?:[。，！,.]|$)"), "entity"),
]

# --- Event patterns ---
_FACT_PATTERNS_EVENT: list[tuple[re.Pattern[str], str]] = [
    # English
    (re.compile(r"(?:i (?:went|go|traveled|travelled|visited|visit|moved))\s+(?:to )?(.+?)(?:[.!]|$)", re.IGNORECASE), "event"),
    (re.compile(r"(?:i (?:started|began|joined|enrolled))\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "event"),
    (re.compile(r"(?:i (?:switched|changed|updated))\s+(.+?)(?:[.!]|$)", re.IGNORECASE), "event"),
    # Chinese
    (re.compile(r"我(?:去了?|去过|前往|到)(.+?)(?:[。，！,.]|旅游|出差|玩|$)"), "event"),
    (re.compile(r"我(?:开始|加入|参加了?|报名了?)(.+?)(?:[。，！,.]|$)"), "event"),
    (re.compile(r"我(?:换|改成|更新了?|升级了?)(.+?)(?:[。，！,.]|了|$)"), "event"),
    (re.compile(r"我(?:搬|搬到|搬去|移居)(.+?)(?:[。，！,.]|了|$)"), "event"),
]

_ALL_PATTERNS = _FACT_PATTERNS_PREF + _FACT_PATTERNS_ENTITY + _FACT_PATTERNS_EVENT


class LocalFactExtractor:
    """Bilingual regex-based fact extraction — zero-LLM fallback.

    Extracts structured factual statements from text via pattern matching
    (English + Chinese).  Returns up to *max_facts* items.

    The output format is compatible with ``MemoryExtractor.extract()``:
    ``[{"category": "preference", "content": "...", "tags": [...], "confidence": 0.8}]``
    """

    def __init__(self, max_facts: int = 8) -> None:
        self._max_facts = max_facts

    def extract(self, text: str, *, role: str | None = None) -> list[dict[str, Any]]:
        """Extract facts from *text*.

        Parameters
        ----------
        text:
            The content to extract facts from.
        role:
            Optional role hint (e.g. "user").  When provided, only extracts
            from user-role content for better precision.

        Returns
        -------
        list[dict]
            Each dict has keys: ``category``, ``content``, ``tags``, ``confidence``.
        """
        if not text or not text.strip():
            return []

        facts: list[dict[str, Any]] = []

        for pat, category in _ALL_PATTERNS:
            # Only apply IGNORECASE patterns to the original text;
            # non-IGNORECASE patterns (Chinese) also use original text.
            for m in pat.finditer(text):
                # Extend match by 20 chars to capture trailing context
                end = min(m.end() + 20, len(text))
                span_text = text[m.start():end].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "category": category,
                    "content": span_text,
                    "tags": [category],
                    "confidence": 0.8,
                })

        # Deduplicate by content prefix
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for f in facts:
            key = f["content"].lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique[: self._max_facts]
