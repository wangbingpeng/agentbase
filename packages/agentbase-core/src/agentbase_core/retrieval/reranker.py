"""Reranker — LLM and heuristic reranking of search results."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from ..llm.base import AbstractLLM
from ..models.context_entry import ContextEntry, ContextScope, ContextType
from ..models.query import SearchQuery, SearchResult

logger = logging.getLogger(__name__)


def heuristic_rerank(
    results: list[SearchResult],
    query: SearchQuery,
    alpha: float = 0.6,
    beta: float = 0.15,
    gamma: float = 0.1,
    delta: float = 0.1,
    epsilon: float = 0.05,
) -> list[SearchResult]:
    """Post-fusion heuristic reranking.

    final_score = alpha*rrf + beta*freshness + gamma*confidence
                  + delta*scope_priority + epsilon*type_match
    """
    for result in results:
        freshness = _compute_freshness(result.entry)
        confidence = result.entry.confidence
        scope_prio = _scope_priority(result.entry, query)
        type_match = _type_match(result.entry, query)

        result.score = (
            alpha * result.score
            + beta * freshness
            + gamma * confidence
            + delta * scope_prio
            + epsilon * type_match
        )

    return sorted(results, key=lambda r: r.score, reverse=True)


def _compute_freshness(entry: ContextEntry) -> float:
    """Compute freshness score — newer entries score higher.

    Uses exponential decay with a half-life of 7 days.
    """
    now = datetime.now(timezone.utc)
    age_seconds = (now - entry.created_at).total_seconds()
    half_life = 7 * 24 * 3600  # 7 days
    return math.exp(-0.693 * age_seconds / half_life) if age_seconds > 0 else 1.0


def _scope_priority(entry: ContextEntry, query: SearchQuery) -> float:
    """Compute scope priority — more specific scope matches score higher.

    Default priority: session > agent > project > global
    """
    priority_map = {
        ContextScope.SESSION: 1.0,
        ContextScope.AGENT: 0.8,
        ContextScope.PROJECT: 0.6,
        ContextScope.GLOBAL: 0.4,
    }
    base = priority_map.get(entry.scope, 0.4)

    # Boost if scope matches the query's scope filter
    if query.scope and entry.scope == query.scope:
        base += 0.2

    return min(base, 1.0)


def _type_match(entry: ContextEntry, query: SearchQuery) -> float:
    """Compute type match score."""
    if query.context_type and entry.context_type == query.context_type:
        return 1.0
    if query.context_type is None:
        return 0.5  # no type filter = neutral
    return 0.0


class LLMReranker:
    """LLM-based reranker for search results."""

    _RERANK_PROMPT = """Rank the following search results by relevance to the query.

Query: {query}

Results:
{results}

Return a JSON array of result indices (0-based) in order of relevance, most relevant first.
Only return the indices, e.g. [2, 0, 1]"""

    def __init__(self, llm: AbstractLLM) -> None:
        self._llm = llm

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results using LLM judgment."""
        if not results:
            return results

        top_k = top_k or len(results)

        # Build result descriptions
        result_descs = []
        for i, r in enumerate(results):
            desc = f"[{i}] {r.entry.l0_abstract or r.entry.l2_full[:200]}"
            result_descs.append(desc)

        prompt = self._RERANK_PROMPT.format(
            query=query,
            results="\n".join(result_descs),
        )

        try:
            import json
            ranking = await self._llm.complete_json(
                prompt=prompt,
                system="You are a search result reranker. Rank results by relevance.",
            )
        except Exception as e:
            logger.warning(f"LLM rerank failed, returning original order: {e}")
            return results[:top_k]

        if not isinstance(ranking, list):
            return results[:top_k]

        # Reorder results based on LLM ranking
        reranked = []
        seen = set()
        for idx in ranking:
            if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                reranked.append(results[idx])
                seen.add(idx)

        # Append any results not in the LLM ranking
        for i, r in enumerate(results):
            if i not in seen:
                reranked.append(r)

        # Update ranking_stage
        for r in reranked:
            r.ranking_stage = "llm"

        return reranked[:top_k]
