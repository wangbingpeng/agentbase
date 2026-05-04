"""LocalQueryDecomposer — rule-based query decomposition (zero-LLM fallback)."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words for query decomposition
# ---------------------------------------------------------------------------

_EN_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "about", "up", "it", "its", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "this", "that", "these", "those", "what",
    "which", "who", "whom", "whose", "also",
})

_ZH_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "他", "她", "它", "们",
    "吗", "呢", "吧", "啊", "把", "被", "让", "给", "从", "向", "对",
    "什么", "怎么", "哪", "那个", "这个", "那些", "这些", "还有",
    "可以", "能", "是否", "因为", "所以", "但是", "而且", "或者",
})

# Pattern to extract key noun phrases from questions
_Q_PATTERNS = [
    # "how many X" → extract X
    re.compile(r"how\s+many\s+(.+?)(?:\s+did|\s+have|\s+are|\s+were|\s+do|\s+does|\?|$)", re.IGNORECASE),
    # "how much X" → extract X
    re.compile(r"how\s+much\s+(.+?)(?:\s+did|\s+have|\s+are|\s+were|\s+do|\s+does|\?|$)", re.IGNORECASE),
    # "how long" → extract what follows (e.g. "how long is my commute" → "my commute")
    re.compile(r"how\s+long\s+(?:is|are|was|were|did|have|has)\s+(.+?)(?:\?|$)", re.IGNORECASE),
    # "what kind/type of X" → extract X
    re.compile(r"what\s+(?:kind|type)\s+of\s+(.+?)(?:\s+did|\s+is|\s+are|\?|$)", re.IGNORECASE),
    # "where did I X" → extract X
    re.compile(r"where\s+did\s+(?:i|you|he|she|they)\s+(.+?)(?:\?|$)", re.IGNORECASE),
    # "when did I X" → extract X
    re.compile(r"when\s+did\s+(?:i|you|he|she|they)\s+(.+?)(?:\?|$)", re.IGNORECASE),
]


class LocalQueryDecomposer:
    """Rule-based query decomposition — zero-LLM fallback.

    Decomposes a natural-language question into 2-3 sub-queries that
    improve recall by targeting different aspects of the question.

    Degradation: when ``IntentAnalyzer`` (LLM) is unavailable, this
    class provides a reasonable approximation via heuristic rules.
    """

    def decompose(self, query: str, *, query_type: str | None = None) -> list[str]:
        """Decompose *query* into sub-queries.

        Parameters
        ----------
        query:
            The original query string.
        query_type:
            Optional query type hint (e.g. "multi-session",
            "temporal-reasoning", "single-session-user").

        Returns
        -------
        list[str]
            1-3 sub-queries.  Always includes the original query as the
            first element.
        """
        if not query or not query.strip():
            return [query] if query else []

        sub_queries: list[str] = [query.strip()]

        # Strategy 1: Extract key noun phrases via patterns
        for pat in _Q_PATTERNS:
            m = pat.search(query)
            if m:
                # Only extract if there's a captured group
                try:
                    phrase = m.group(1).strip().rstrip("?!.,;：").strip()
                except IndexError:
                    continue
                if phrase and phrase.lower() not in {q.lower() for q in sub_queries}:
                    sub_queries.append(phrase)

        # Strategy 2: Remove stop words to create keyword-only sub-query
        keywords = self._extract_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query.lower() != query.lower() and keyword_query.lower() not in {q.lower() for q in sub_queries}:
                sub_queries.append(keyword_query)

        # Strategy 3: For temporal queries, extract date-related tokens
        if query_type in ("temporal-reasoning", "multi-session"):
            date_tokens = self._extract_temporal_tokens(query)
            if date_tokens:
                date_query = " ".join(date_tokens)
                if date_query.lower() not in {q.lower() for q in sub_queries}:
                    sub_queries.append(date_query)

        return sub_queries[:5]  # Cap at 5 sub-queries (was 3; D3 temporal tokens may add 2-3 more)

    def _extract_keywords(self, query: str) -> list[str]:
        """Remove stop words and extract content-bearing keywords."""
        # Tokenize by whitespace and punctuation
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)

        keywords = []
        for token in tokens:
            if token.lower() in _EN_STOP_WORDS:
                continue
            if token in _ZH_STOP_WORDS:
                continue
            if len(token) <= 1:
                continue
            keywords.append(token)

        return keywords

    # --- Extended temporal patterns (D3: Temporal Token Enrichment) ---
    _EXTENDED_TEMPORAL_PATTERNS: list[re.Pattern] = [
        # "May 2022", "April of 2023"
        re.compile(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
            r"(?:\s+of)?\s+\d{4}\b", re.IGNORECASE
        ),
        # "summer 2021", "winter of 2020"
        re.compile(
            r"\b(?:spring|summer|fall|autumn|winter)(?:\s+of)?\s+\d{4}\b", re.IGNORECASE
        ),
        # Standalone year 2019-2029
        re.compile(r"\b(20[12]\d)\b"),
        # "3 weeks ago", "two years later"
        re.compile(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:day|week|month|year)s?\s+(?:ago|later|before|after)\b", re.IGNORECASE
        ),
        # "last month", "next week"
        re.compile(
            r"\b(?:last|next|this)\s+(?:week|month|year|weekend)\b", re.IGNORECASE
        ),
    ]

    def _extract_temporal_tokens(self, query: str) -> list[str]:
        """Extract date/time-related tokens from a temporal query.

        D3 Temporal Token Enrichment: extends basic date patterns with
        month-year combos ("May 2022"), season-year ("summer 2021"),
        standalone years (2024), relative expressions ("3 weeks ago"),
        and recent-adjacent ("last month").
        """
        temporal_patterns = [
            # English dates
            re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.IGNORECASE),
            re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
            re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
            re.compile(r"\b(?:last|this|next|previous)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
            # Chinese dates
            re.compile(r"[\u4e00-\u9fff]*\d{4}[\u5e74]\d{1,2}[\u6708]\d{1,2}[\u65e5]?"),
            re.compile(r"(?:\u4e0a|\u8fd9|\u4e0b)(?:\u5468|\u4e2a|\u6708)(?:\u521d|\u672b|\u4e2d)?"),
        ]

        tokens: list[str] = []
        seen_lower: set[str] = set()

        # Basic patterns
        for pat in temporal_patterns:
            for m in pat.finditer(query):
                tok = m.group(0)
                if tok.lower() not in seen_lower:
                    tokens.append(tok)
                    seen_lower.add(tok.lower())

        # D3: Extended temporal patterns
        for pat in self._EXTENDED_TEMPORAL_PATTERNS:
            for m in pat.finditer(query):
                tok = m.group(0).strip()
                if tok and tok.lower() not in seen_lower:
                    tokens.append(tok)
                    seen_lower.add(tok.lower())

        return tokens
