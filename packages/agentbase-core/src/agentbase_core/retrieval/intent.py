"""IntentAnalyzer — query intent analysis for typed sub-queries."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..llm.base import AbstractLLM
from ..models.context_entry import ContextType

logger = logging.getLogger(__name__)


class TypedSubQuery:
    """A typed sub-query produced by intent analysis."""

    def __init__(
        self,
        query: str,
        context_type: ContextType = ContextType.MEMORY,
        category: str | None = None,
    ) -> None:
        self.query = query
        self.context_type = context_type
        self.category = category

    def __repr__(self) -> str:
        return f"TypedSubQuery(query={self.query!r}, type={self.context_type.value}, category={self.category})"


_INTENT_SYSTEM_PROMPT = """你是一个检索意图分析器。分析用户的查询，将其拆解为 0-5 个类型化子查询。

每个子查询包含：
- type: memory / resource / skill
- category: profile/preference/entity/event/case/pattern（仅 memory 类型）
- query: 优化后的检索词

如果查询不需要拆解，返回单个子查询。
返回 JSON 数组。"""


class IntentAnalyzer:
    """Analyze search query intent and decompose into typed sub-queries.

    When LLM is available, uses it for semantic decomposition.
    Falls back to simple rule-based analysis when LLM is unavailable.

    Also detects high-level query types (temporal-reasoning, multi-session,
    knowledge-update, preference, aggregation) which can influence
    retrieval strategy.
    """

    # D2: Aggregation-aware detection patterns
    _AGG_PATTERNS = [
        "how many", "how much", "how often", "how long",
        "total", "in total", "all the", "every",
        "how many times", "how many days", "how many hours",
        "how many weeks", "how many months",
        "总共", "一共", "多少", "每个", "所有", "全部",
    ]

    def __init__(self, llm: AbstractLLM | None = None) -> None:
        self._llm = llm

    @staticmethod
    def detect_query_type(query: str) -> str | None:
        """Detect high-level query type from the query text using rules.

        Returns one of: "temporal-reasoning", "knowledge-update",
        "multi-session", "preference", or None (generic).
        """
        q_lower = query.lower()

        # Temporal reasoning: date comparisons, before/after, first/last, when
        temporal_kw = [
            "first time", "last time", "most recent", "earliest",
            "before", "after", "prior to", "since", "until",
            "how many times", "how often", "when did", "when was",
            "how many days", "how many weeks", "how many months", "how long ago",
            "how long did", "how long were", "how long was",
            "days passed", "weeks passed", "months passed",
            "days between", "weeks between", "months between",
            "第一次", "最后一次", "最近", "最早", "最晚",
            "之前", "之后", "什么时候", "多久",
            "most recently", "oldest",
        ]
        if any(kw in q_lower for kw in temporal_kw):
            return "temporal-reasoning"

        # Preference: likes, dislikes, favorites, suggestions, recommendations
        # NOTE: checked BEFORE knowledge-update, because preference queries
        # like "I need a new X, any advice?" contain "new" but are really
        # asking for suggestions/recommendations, not tracking updates.
        pref_kw = [
            "prefer", "favorite", "favourite", "like", "dislike",
            "suggest", "recommend", "any tips", "any advice", "any ideas",
            "what should i", "can you recommend", "can you suggest",
            "do you have any tips", "do you have any advice", "do you have any ideas",
            "what kind of", "what type of", "what sort of",
            "which ", "what would be a good", "what's a good",
            "what is the best", "what are good", "give me some",
            "i need a", "looking for a", "in the market for",
            "喜欢", "偏爱", "最爱", "偏好", "不喜", "建议", "推荐",
            "什么好", "哪个好", "哪种",
        ]
        if any(kw in q_lower for kw in pref_kw):
            return "preference"

        # Knowledge update: current/latest value of some attribute
        # NOTE: checked AFTER preference, because "I need a new X" queries
        # are preference-seeking, not update-tracking.
        update_kw = [
            "current ", "latest ", "new ", "updated ", "changed ",
            "now", "currently", "nowadays",
            "现在的", "最新的", "当前的", "更换", "更新",
        ]
        if any(kw in q_lower for kw in update_kw):
            return "knowledge-update"

        # Multi-session: aggregation across sessions
        # NOTE: checked AFTER preference, because "what should I" / "which"
        # queries are more likely preference than multi-session
        multi_kw = [
            "all the ", "every ", "each ", "total ", "overall",
            "how many", "how much", "how often", "how long",
            "所有的", "全部", "每个", "总共", "一共", "多少",
        ]
        if any(kw in q_lower for kw in multi_kw):
            return "multi-session"

        return None

    @staticmethod
    def is_aggregation_query(query: str) -> bool:
        """D2: Detect whether the query is an aggregation-type question.

        Aggregation queries ask about totals, counts, or exhaustive
        enumeration across multiple entries/sessions. They require
        broader recall (higher top_k) and aggregation-aware prompting.

        Examples:
        - "How many weddings have I attended?"
        - "What is the total amount I spent on books?"
        - "List all the trips I took this year."
        """
        q_lower = query.lower()
        return any(p in q_lower for p in IntentAnalyzer._AGG_PATTERNS)

    # --------------------------------------------------------------
    # D5: Temporal filter parsing
    # --------------------------------------------------------------

    _MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
        "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
        "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    @staticmethod
    def parse_temporal_filter(
        query: str, now: datetime | None = None
    ) -> tuple[datetime | None, datetime | None]:
        """D5: Parse temporal expressions from query text into a date range.

        Returns ``(date_from, date_to)`` where either side may be ``None``
        (no bound). Supports common patterns:

        - Relative: "yesterday", "last week", "last month", "last year",
          "this week", "this month", "this year", "last N days/weeks/months"
        - Absolute month: "in March", "in March 2024", "in 2024"
        - Directional: "before 2024", "after 2024", "since January",
          "prior to March 2024", "until June"

        Unrecognized expressions return ``(None, None)`` — callers
        should fall back to unfiltered retrieval.
        """
        if not query:
            return None, None

        q = query.lower().strip()
        if now is None:
            now = datetime.now(timezone.utc)

        # --- Relative: last N days/weeks/months ---
        m = re.search(r"last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)", q)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("day"):
                return now - timedelta(days=n), now
            if unit.startswith("week"):
                return now - timedelta(weeks=n), now
            if unit.startswith("month"):
                return now - timedelta(days=30 * n), now
            if unit.startswith("year"):
                return now - timedelta(days=365 * n), now

        # --- Relative single units ---
        if "yesterday" in q:
            y = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return y, y + timedelta(days=1)
        if "last week" in q or "past week" in q:
            return now - timedelta(weeks=1), now
        if "last month" in q or "past month" in q:
            return now - timedelta(days=30), now
        if "last year" in q or "past year" in q:
            return now - timedelta(days=365), now
        if "this week" in q:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now
        if "this month" in q:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start, now
        if "this year" in q:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return start, now

        # --- Absolute month with optional year: "in March 2024", "in March" ---
        month_pat = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
        m = re.search(month_pat + r"\s*(\d{4})", q)
        if m:
            month = IntentAnalyzer._MONTH_MAP[m.group(1)]
            year = int(m.group(2))
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = datetime(
                year + (1 if month == 12 else 0),
                1 if month == 12 else month + 1,
                1,
                tzinfo=timezone.utc,
            )
            return start, end

        # --- Year only: "in 2024", "before 2024", "after 2024", "since 2024" ---
        m = re.search(r"\b(before|prior to|until)\s+(\d{4})\b", q)
        if m:
            year = int(m.group(2))
            return None, datetime(year, 1, 1, tzinfo=timezone.utc)

        m = re.search(r"\b(after|since|from)\s+(\d{4})\b", q)
        if m:
            year = int(m.group(2))
            return datetime(year, 1, 1, tzinfo=timezone.utc), None

        m = re.search(r"\bin\s+(\d{4})\b", q)
        if m:
            year = int(m.group(1))
            return (
                datetime(year, 1, 1, tzinfo=timezone.utc),
                datetime(year + 1, 1, 1, tzinfo=timezone.utc),
            )

        # --- Directional with month: "before March", "since January" ---
        m = re.search(r"\b(before|prior to|until)\s+" + month_pat, q)
        if m:
            month = IntentAnalyzer._MONTH_MAP[m.group(2)]
            year = now.year
            return None, datetime(year, month, 1, tzinfo=timezone.utc)

        m = re.search(r"\b(after|since|from)\s+" + month_pat, q)
        if m:
            month = IntentAnalyzer._MONTH_MAP[m.group(2)]
            year = now.year
            # Heuristic: if the referenced month is in the future relative
            # to now, fall back to the previous year's occurrence.
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if start > now:
                start = datetime(year - 1, month, 1, tzinfo=timezone.utc)
            return start, None

        # --- Lone month: "in March" (no year) — use current year ---
        m = re.search(r"\bin\s+" + month_pat, q)
        if m:
            month = IntentAnalyzer._MONTH_MAP[m.group(1)]
            year = now.year
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if start > now:
                start = datetime(year - 1, month, 1, tzinfo=timezone.utc)
                end = datetime(year, month, 1, tzinfo=timezone.utc) if month < 12 else datetime(year, 12, 31, tzinfo=timezone.utc)
            else:
                end = datetime(
                    year + (1 if month == 12 else 0),
                    1 if month == 12 else month + 1,
                    1,
                    tzinfo=timezone.utc,
                )
            return start, end

        return None, None

    async def analyze(self, query: str) -> list[TypedSubQuery]:
        """Analyze query intent and return typed sub-queries."""
        if self._llm is not None:
            try:
                return await self._analyze_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM intent analysis failed, falling back to rules: {e}")

        return self._analyze_with_rules(query)

    async def _analyze_with_llm(self, query: str) -> list[TypedSubQuery]:
        """Use LLM to analyze query intent."""
        prompt = f"""Analyze the following search query and decompose it into typed sub-queries.

Query: {query}

Return a JSON array of objects with keys: type, category, query
- type: one of "memory", "resource", "skill"
- category: one of "profile", "preference", "entity", "event", "case", "pattern" (only for memory type)
- query: optimized search terms"""

        import json
        result = await self._llm.complete_json(
            prompt=prompt,
            system=_INTENT_SYSTEM_PROMPT,
        )

        if not isinstance(result, list):
            result = [result]

        sub_queries = []
        for item in result[:5]:
            if not isinstance(item, dict):
                continue
            type_str = item.get("type", "memory")
            try:
                ctx_type = ContextType(type_str)
            except ValueError:
                ctx_type = ContextType.MEMORY
            sub_queries.append(TypedSubQuery(
                query=item.get("query", query),
                context_type=ctx_type,
                category=item.get("category"),
            ))

        return sub_queries if sub_queries else [TypedSubQuery(query=query)]

    @staticmethod
    def _analyze_with_rules(query: str) -> list[TypedSubQuery]:
        """Simple rule-based intent analysis (no LLM required)."""
        sub_queries = [TypedSubQuery(query=query)]

        # Heuristic: detect type-specific keywords
        q_lower = query.lower()

        # Resource-related keywords
        resource_keywords = ["文档", "document", "api", "规范", "代码", "code", "文件", "file", "链接", "url"]
        if any(kw in q_lower for kw in resource_keywords):
            sub_queries.append(TypedSubQuery(
                query=query,
                context_type=ContextType.RESOURCE,
            ))

        # Skill-related keywords
        skill_keywords = ["工具", "tool", "技能", "skill", "怎么", "如何", "how to", "命令", "command"]
        if any(kw in q_lower for kw in skill_keywords):
            sub_queries.append(TypedSubQuery(
                query=query,
                context_type=ContextType.SKILL,
            ))

        return sub_queries
