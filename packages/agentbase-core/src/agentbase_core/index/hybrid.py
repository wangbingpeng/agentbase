"""HybridIndex — FTS5 + (optional) vector search with score fusion."""

from __future__ import annotations

import logging
from typing import Any

from ..models.context_entry import ContextEntry
from ..models.query import SearchResult
from .base import AbstractIndex
from .sqlite_fts import SQLiteFTSIndex

logger = logging.getLogger(__name__)

# ── Adaptive weight presets per query type ──────────────────────────────
# Derived from LongMemEval benchmark analysis:
#   - single-session-assistant: FTS much better than vector (-10.71%) → boost FTS
#   - single-session-user: FTS better (-4.29%) → boost FTS
#   - knowledge-update: FTS better (-3.85%) → boost FTS
#   - single-session-preference: Vector better (+10.00%) → boost vector
#   - temporal-reasoning: Same → balanced
#   - multi-session: Slightly better with vector (+0.75%) → default
QUERY_TYPE_WEIGHTS: dict[str, tuple[float, float]] = {
    "single-session-assistant": (0.7, 0.3),
    "single-session-user": (0.6, 0.4),
    "knowledge-update": (0.6, 0.4),
    "preference": (0.3, 0.7),
    "single-session-preference": (0.3, 0.7),
    "temporal-reasoning": (0.5, 0.5),
    "multi-session": (0.4, 0.6),
}


def _minmax_normalize(raw_scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1] range.

    - Empty dict → empty dict
    - Single item → 1.0
    - All same value → 1.0
    - Otherwise: (score - min) / (max - min)
    """
    if not raw_scores:
        return {}
    if len(raw_scores) == 1:
        return {eid: 1.0 for eid in raw_scores}

    values = list(raw_scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {eid: 1.0 for eid in raw_scores}

    range_val = max_val - min_val
    return {eid: (s - min_val) / range_val for eid, s in raw_scores.items()}


def score_fusion(
    fts_results: list[SearchResult],
    vec_results: list[SearchResult],
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
) -> list[SearchResult]:
    """Score-based fusion — merge FTS and vector search using normalized scores.

    Unlike RRF which only uses rank positions, score fusion preserves
    actual relevance information from both retrieval systems:

    1. Min-max normalize FTS BM25 scores to [0, 1]
    2. Min-max normalize vector cosine scores to [0, 1]
    3. Combine: final_score = fts_weight * norm_fts + vec_weight * norm_vec

    Documents appearing in both result sets naturally receive higher
    combined scores because both terms contribute.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    # Collect raw scores per source
    fts_raw: dict[str, float] = {}
    vec_raw: dict[str, float] = {}

    for r in fts_results:
        eid = r.entry.id
        fts_raw[eid] = r.score
        if eid not in result_map:
            result_map[eid] = r

    for r in vec_results:
        eid = r.entry.id
        vec_raw[eid] = r.score
        if eid not in result_map:
            result_map[eid] = r

    # Min-max normalize each source independently
    fts_norm = _minmax_normalize(fts_raw)
    vec_norm = _minmax_normalize(vec_raw)

    # Combine with weights; missing source → 0 contribution
    all_ids = set(fts_raw.keys()) | set(vec_raw.keys())
    for eid in all_ids:
        fts_s = fts_norm.get(eid, 0.0)
        vec_s = vec_norm.get(eid, 0.0)
        scores[eid] = fts_weight * fts_s + vec_weight * vec_s

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for eid in sorted_ids:
        r = result_map[eid]
        results.append(
            SearchResult(
                entry=r.entry,
                score=scores[eid],
                score_breakdown={
                    **(r.score_breakdown or {}),
                    "score_fusion": scores[eid],
                    "fts_norm": fts_norm.get(eid, 0.0),
                    "vec_norm": vec_norm.get(eid, 0.0),
                },
                ranking_stage="score_fusion",
                matched_by=r.matched_by if r.matched_by != "hybrid" else "hybrid",
            )
        )
    return results


def rrf_fusion(
    fts_results: list[SearchResult],
    vec_results: list[SearchResult],
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion — legacy rank-based merge (kept for backward compat).

    Prefer ``score_fusion`` which preserves actual relevance scores.

    RRF_score(d) = Σ weight_i / (k + rank_i(d))
    """
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, result in enumerate(fts_results):
        eid = result.entry.id
        scores[eid] = scores.get(eid, 0.0) + fts_weight / (rrf_k + rank + 1)
        if eid not in result_map:
            result_map[eid] = result

    for rank, result in enumerate(vec_results):
        eid = result.entry.id
        scores[eid] = scores.get(eid, 0.0) + vec_weight / (rrf_k + rank + 1)
        if eid not in result_map:
            result_map[eid] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for eid in sorted_ids:
        r = result_map[eid]
        results.append(
            SearchResult(
                entry=r.entry,
                score=scores[eid],
                score_breakdown={
                    **(r.score_breakdown or {}),
                    "rrf": scores[eid],
                },
                ranking_stage="rrf",
                matched_by=r.matched_by if r.matched_by != "hybrid" else "hybrid",
            )
        )
    return results


class HybridIndex(AbstractIndex):
    """Hybrid index combining FTS5 and optional vector search with score fusion."""

    def __init__(
        self,
        fts_index: SQLiteFTSIndex,
        vec_index: AbstractIndex | None = None,
        fts_weight: float = 0.4,
        vec_weight: float = 0.6,
        rrf_k: int = 60,
        embedder: Any | None = None,
        fusion_method: str = "score",
    ) -> None:
        self._fts = fts_index
        self._vec = vec_index
        self._fts_weight = fts_weight
        self._vec_weight = vec_weight
        self._rrf_k = rrf_k
        self._embedder = embedder
        self._fusion_method = fusion_method  # "score" or "rrf"

    @property
    def vec_available(self) -> bool:
        return self._vec is not None

    async def add(self, entry: ContextEntry) -> None:
        """Index entry in both FTS and vector indexes."""
        # FTS is auto-synced via triggers; vector needs explicit add
        if self._vec is not None:
            try:
                await self._vec.add(entry)
            except Exception as e:
                logger.warning(f"Vector index add failed for {entry.id}: {e}")

    async def add_batch(self, entries: list[ContextEntry]) -> None:
        """Batch-index entries in both FTS and vector indexes.

        FTS entries are inserted in bulk via a single transaction.
        Vector entries are added individually (no batch API available).
        """
        # Batch add to FTS index
        try:
            await self._fts.add_batch(entries)
        except Exception as e:
            logger.warning(f"FTS batch add failed: {e}")
            # Fallback: add one by one
            for entry in entries:
                try:
                    await self._fts.add(entry)
                except Exception:
                    pass

        # Vector index: add one by one (no batch API)
        if self._vec is not None:
            for entry in entries:
                try:
                    await self._vec.add(entry)
                except Exception as e:
                    logger.warning(f"Vector index add failed for {entry.id}: {e}")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        context_type: str | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        fts_column: str | None = None,
        include_statuses: list[str] | None = None,
        query_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        speaker: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: FTS + vector + score fusion.

        Falls back to FTS-only if vector index is unavailable.
        When ``query_type`` is provided, adaptive weights are looked up
        from ``QUERY_TYPE_WEIGHTS`` and override the constructor defaults.

        Optional filters:
        - date_from / date_to: filter by ce.created_at range (ISO format strings)
        - speaker: filter by role tag or extra.role (case-insensitive match)
        """
        # Resolve adaptive weights for this query type
        fts_w, vec_w = self._resolve_weights(query_type)
        fts_results: list[SearchResult] = []
        vec_results: list[SearchResult] = []

        # Always attempt FTS search
        fts_failed = False
        try:
            fts_results = await self._fts.search(
                query=query,
                top_k=top_k * 3,  # Over-fetch for better RRF
                context_type=context_type,
                scope=scope,
                owner_id=owner_id,
                fts_column=fts_column,
                include_statuses=include_statuses,
                date_from=date_from,
                date_to=date_to,
                speaker=speaker,
            )
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
            fts_failed = True
            # Don't return empty — continue to check vector results

        # Attempt vector search if available
        # When FTS failed, increase vector top_k to compensate for reduced recall
        vec_top_k = top_k * 5 if fts_failed else top_k * 3
        if self._vec is not None:
            try:
                # Generate query embedding for vector search
                query_embedding = None
                if self._embedder is not None:
                    try:
                        query_embedding = await self._embedder.embed(query)
                    except Exception as e:
                        logger.warning(f"Query embedding failed: {e}")

                if query_embedding is not None:
                    vec_results = await self._vec.search(
                        query=query,
                        top_k=vec_top_k,
                        context_type=context_type,
                        scope=scope,
                        owner_id=owner_id,
                        query_embedding=query_embedding,
                        date_from=date_from,
                        date_to=date_to,
                        speaker=speaker,
                    )
                else:
                    logger.debug("No query embedding available, skipping vector search")
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to FTS only: {e}")

        # If no vector results, return FTS results directly
        if not vec_results:
            # Mark results as FTS-only with degrade_reason per SPEC §7.7
            for r in fts_results[:top_k]:
                r.matched_by = "fts"
                if self._vec is None:
                    r.degrade_reason = "vec_unavailable"
                else:
                    r.degrade_reason = "embedding_failed"
            return fts_results[:top_k]

        # Score fusion (or legacy RRF)
        if self._fusion_method == "rrf":
            fused = rrf_fusion(
                fts_results,
                vec_results,
                fts_weight=fts_w,
                vec_weight=vec_w,
                rrf_k=self._rrf_k,
            )
        else:
            fused = score_fusion(
                fts_results,
                vec_results,
                fts_weight=fts_w,
                vec_weight=vec_w,
            )

        return fused[:top_k]

    def _resolve_weights(self, query_type: str | None) -> tuple[float, float]:
        """Return (fts_weight, vec_weight) adapted to *query_type*.

        Falls back to constructor defaults when *query_type* is None or
        not in the preset table.
        """
        if query_type and query_type in QUERY_TYPE_WEIGHTS:
            return QUERY_TYPE_WEIGHTS[query_type]
        return (self._fts_weight, self._vec_weight)

    async def remove(self, entry_id: str) -> None:
        """Remove from vector index (FTS handled by triggers)."""
        if self._vec is not None:
            try:
                await self._vec.remove(entry_id)
            except Exception as e:
                logger.warning(f"Vector index remove failed for {entry_id}: {e}")

    async def count(self) -> int:
        """Return count from FTS index."""
        return await self._fts.count()
