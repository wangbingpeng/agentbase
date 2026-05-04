"""RetrievalEngine — hierarchical progressive retrieval engine."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from ..index.hybrid import HybridIndex
from ..llm.base import AbstractLLM
from ..models.context_entry import ContextEntry, ContextScope, ContextType, EntryStatus
from ..models.query import SearchQuery, SearchResult, SearchStrategy
from ..models.trace import RetrievalTrace, TraceStep
from ..models.config import RetrievalConfig
from ..store.sqlite_store import SQLiteStore
from .budget import TokenBudget
from .intent import IntentAnalyzer
from .query_decompose import LocalQueryDecomposer
from .reranker import LLMReranker, heuristic_rerank
from ..ingester.ner_extractor import NerExtractor

logger = logging.getLogger(__name__)


def _determine_load_level(query: SearchQuery, results: list[SearchResult]) -> str:
    """Deterministically determine load level based on query and results.

    Per SPEC §7.8: load_level=auto must be deterministic.
    Rules (in order of priority):
    1. top_k > 20 → l0
    2. token_budget < 1000 → l0
    3. strategy=hierarchical → l0/l1 (progressive)
    4. context_type=resource && l2_full long → l1
    5. context_type=memory && l2_full short → l2
    6. default → l1
    """
    if query.load_level != "auto":
        return query.load_level

    top_k = query.top_k
    token_budget = query.token_budget

    # Rule 1: Large result set → only L0
    if top_k > 20:
        return "l0"

    # Rule 2: Tight budget → only L0
    if token_budget is not None and token_budget < 1000:
        return "l0"

    # Rule 3: Hierarchical strategy → start with L0/L1
    if query.strategy == SearchStrategy.HIERARCHICAL:
        return "l1"

    # Rule 4 & 5: Inspect results for resource/memory type
    if results:
        first = results[0].entry
        if first.context_type == ContextType.RESOURCE and len(first.l2_full) > 500:
            return "l1"
        if first.context_type == ContextType.MEMORY and len(first.l2_full) < 200:
            return "l2"

    # Rule 6: Default → L1
    return "l1"


class RetrievalEngine:
    """Hierarchical progressive retrieval engine.

    Flow (default path, no LLM):
    1. Query normalize
    2. Scope/type/tag filter
    3. FTS + vector search (dual recall)
    4. RRF fusion
    5. Heuristic rerank (freshness/confidence/scope/type)
    6. Budget trim
    7. Return

    Flow (LLM enhanced, explicit opt-in):
    1. Intent decomposition
    2. Per-sub-query search
    3. Merge results
    4. LLM rerank (optional)
    5. Budget trim
    6. Return
    """

    def __init__(
        self,
        index: HybridIndex,
        llm: AbstractLLM | None = None,
        fts_weight: float = 0.4,
        vec_weight: float = 0.6,
        rrf_k: int = 60,
        retrieval_config: RetrievalConfig | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._index = index
        self._llm = llm
        self._intent_analyzer = IntentAnalyzer(llm=llm) if llm else None
        self._llm_reranker = LLMReranker(llm) if llm else None
        self._fts_weight = fts_weight
        self._vec_weight = vec_weight
        self._rrf_k = rrf_k
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._query_decomposer = LocalQueryDecomposer() if self._retrieval_config.query_decomposition else None
        self._ner_extractor = NerExtractor() if self._retrieval_config.ner_boost else None
        self._ner_weight = self._retrieval_config.ner_weight
        # Optional store reference — enables session_memory_links lookups for
        # D3 (session-link based co-retrieval) and D4 (two-stage completion).
        self._store = store

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute a search query through the retrieval pipeline."""
        start_time = time.monotonic()
        trace = RetrievalTrace(
            query=query.text,
            strategy=query.strategy,
        )

        # Auto-detect query type if not explicitly provided (safety net —
        # engine.find() already runs detection, but search() can be called directly)
        query_type = query.query_type
        if query_type is None:
            query_type = IntentAnalyzer.detect_query_type(query.text)

        # Persist detected type back to the query so sub-methods can use it
        if query.query_type is None and query_type is not None:
            query.query_type = query_type

        # D5: Temporal-aware time filter auto-population
        # When the query is temporal-reasoning and the caller did not
        # provide explicit date_from/date_to, parse time expressions from
        # the query text and populate the filter automatically.
        if query_type == "temporal-reasoning" and query.date_from is None and query.date_to is None:
            try:
                date_from, date_to = IntentAnalyzer.parse_temporal_filter(query.text)
                if date_from is not None or date_to is not None:
                    query.date_from = date_from
                    query.date_to = date_to
                    logger.debug(
                        f"D5: Temporal filter auto-populated from query: "
                        f"date_from={date_from}, date_to={date_to}"
                    )
            except Exception as e:
                logger.debug(f"Temporal filter parsing failed: {e}")

        # D2: Aggregation-aware top_k boost
        # Aggregation queries ("how many", "total", "how much") need
        # exhaustive recall across all relevant entries.
        is_agg = IntentAnalyzer.is_aggregation_query(query.text)
        if is_agg and self._retrieval_config.agg_detection and query.top_k < self._retrieval_config.agg_top_k:
            original_top_k = query.top_k
            query.top_k = self._retrieval_config.agg_top_k
            logger.debug(
                f"D2: Aggregation query detected, top_k boosted "
                f"{original_top_k} → {query.top_k}"
            )

        strategy = query.strategy

        if strategy == SearchStrategy.HIERARCHICAL:
            results = await self._hierarchical_search(query, trace)
        elif strategy == SearchStrategy.FTS:
            results = await self._fts_only_search(query, trace)
        elif strategy == SearchStrategy.VECTOR:
            results = await self._vector_only_search(query, trace)
        else:
            # HYBRID (default)
            results = await self._hybrid_search(query, trace)

        # --- Query-type-aware post-processing ---
        if results and query_type:
            results = await self._apply_query_type_strategy(results, query, query_type)

        # --- D1: Session Co-Retrieval ---
        if results and self._retrieval_config.session_co_retrieval:
            try:
                results = await self._session_co_retrieve(results, query)
            except Exception as e:
                logger.debug(f"Session co-retrieval failed: {e}")

        # Apply heuristic rerank (default path, always runs)
        if results:
            results = heuristic_rerank(results, query)
            for r in results:
                r.ranking_stage = "heuristic"

        # Optional LLM rerank
        if self._llm_reranker and query.strategy == SearchStrategy.HIERARCHICAL and results:
            try:
                results = await self._llm_reranker.rerank(query.text, results, top_k=query.top_k)
            except Exception as e:
                logger.warning(f"LLM rerank failed: {e}")

        # Determine load level and apply
        load_level = _determine_load_level(query, results)
        results = self._apply_load_level(results, load_level)

        # Apply token budget
        if query.token_budget is not None:
            results = self._apply_budget(results, query.token_budget)

        # Final trim to top_k
        results = results[: query.top_k]

        # Update trace
        elapsed = (time.monotonic() - start_time) * 1000
        trace.finish(
            result_ids=[r.entry.id for r in results],
            total_latency_ms=elapsed,
        )

        # Attach trace if requested
        if query.include_trace:
            for r in results:
                r.trace = trace

        return results

    async def _apply_query_type_strategy(
        self,
        results: list[SearchResult],
        query: SearchQuery,
        query_type: str,
    ) -> list[SearchResult]:
        """Apply query-type-specific post-processing to search results.

        Strategies:
        - temporal-reasoning: boost results with older created_at (more context),
          sort by recency within relevant session groups
        - knowledge-update: strongly boost most recent entries, suppress older ones
          for the same entity/topic
        - multi-session: ensure results span multiple sessions (deduplicate by
          session_index in extra, keep most relevant per session)
        - preference: boost entries tagged 'user' role and preference-related categories
        """
        if query_type == "temporal-reasoning":
            return self._temporal_strategy(results, query)
        elif query_type == "knowledge-update":
            return self._knowledge_update_strategy(results, query)
        elif query_type == "multi-session":
            return await self._multi_session_strategy(results, query)
        elif query_type == "preference":
            return self._preference_strategy(results, query)
        return results

    @staticmethod
    def _temporal_strategy(
        results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Temporal reasoning: enhance results with date-aware scoring.

        Strategy:
        1. Entries with session_date metadata get a boost (structured temporal info)
        2. User-uttered entries get a moderate boost (they contain event descriptions)
        3. Sort by session date to build a timeline, then by relevance within groups
        4. Results that match more query tokens get extra weighting
        5. For multi-date scenarios, ensure representation from each date
        """
        from datetime import datetime, timezone

        # First pass: apply score adjustments
        for r in results:
            extra = r.entry.extra or {}
            tags = r.entry.tags or []

            # Entries with session_date metadata get a strong relevance signal
            if "session_date" in extra:
                r.score *= 1.3  # boosted from 1.2 for better temporal recall

            # User-uttered entries often describe events with temporal cues
            if "user" in tags or extra.get("role") == "user":
                r.score *= 1.2  # boosted from 1.15

        # T4: Ensure temporal diversity — if results span multiple sessions,
        # ensure at least 2 results per session (up to a limit)
        session_groups: dict[str, list[SearchResult]] = {}
        for r in results:
            extra = r.entry.extra or {}
            session_key = extra.get("session_date", "unknown")
            if session_key not in session_groups:
                session_groups[session_key] = []
            session_groups[session_key].append(r)

        if len(session_groups) > 1:
            # Interleave: ensure each session has representation
            # Sort groups by score of top result, then interleave
            for group in session_groups.values():
                group.sort(key=lambda r: r.score, reverse=True)

            interleaved = []
            max_per_session = max(3, query.top_k // len(session_groups))
            for session_key in sorted(session_groups.keys()):
                interleaved.extend(session_groups[session_key][:max_per_session])

            # Sort interleaved results by score
            interleaved.sort(key=lambda r: r.score, reverse=True)

            # Merge: interleave first, then fill with remaining sorted by score
            interleaved_ids = {r.entry.id for r in interleaved}
            remaining = [r for r in results if r.entry.id not in interleaved_ids]
            remaining.sort(key=lambda r: r.score, reverse=True)
            results = interleaved + remaining
        else:
            results.sort(key=lambda r: r.score, reverse=True)

        return results

    def _knowledge_update_strategy(
        self,
        results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Knowledge update: strongly prioritize most recent entries."""
        from datetime import datetime, timezone

        # Use configured half-life (default 14 days), not hardcoded value
        half_life = self._retrieval_config.knowledge_update_half_life_days

        # Group by session and keep only the latest session's entries on top
        # Strong recency boost
        now = datetime.now(timezone.utc)
        for r in results:
            age_days = (now - r.entry.created_at).total_seconds() / 86400
            # Exponential recency boost: newer = much higher
            recency = math.exp(-0.693 * age_days / half_life) if age_days > 0 else 2.0
            r.score *= (0.5 + 0.5 * recency)  # blend original score with recency

        results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate by topic: if entries from different sessions cover the
        # same entity, keep only the most recent
        seen_topics: dict[str, SearchResult] = {}
        deduped = []
        for r in results:
            # Use first 50 chars of content as a rough topic signature
            topic_key = r.entry.l2_full[:50].strip().lower() if r.entry.l2_full else r.entry.id
            if topic_key not in seen_topics:
                seen_topics[topic_key] = r
                deduped.append(r)

        return deduped if deduped else results

    async def _multi_session_strategy(
        self,
        results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Multi-session: ensure diversity across sessions.

        Strategy:
        1. Group results by session_index
        2. Interleave results from different sessions (fair representation)
        3. Prioritize session_summary and fact entries within each session
        4. Session completion: if the number of covered sessions < 50% of
           total sessions (inferred from max session_index), pull the top
           result from each uncovered session into the final list
        """
        # Group by session_index, keep top results from each session
        session_groups: dict[int, list[SearchResult]] = {}
        summary_entries: dict[int, SearchResult] = {}  # session_summary per session
        fact_entries: dict[int, list[SearchResult]] = {}  # facts per session
        for r in results:
            extra = r.entry.extra or {}
            tags = r.entry.tags or []
            session_idx = extra.get("session_index", 0)
            if session_idx not in session_groups:
                session_groups[session_idx] = []
            session_groups[session_idx].append(r)

            # Track session summary entries separately for prioritization
            if "session_summary" in tags:
                summary_entries[session_idx] = r

            # Track fact entries separately for prioritization
            if "fact" in tags:
                if session_idx not in fact_entries:
                    fact_entries[session_idx] = []
                fact_entries[session_idx].append(r)

        # --- Session completion: ensure broad coverage ---
        # Even if FTS only returned results from one session,
        # we should try to pull entries from other sessions.
        if self._store:
            try:
                # Query DB for the true max session_index
                async with self._store._pool.get_read_conn() as conn:
                    cursor = await conn.execute(
                        "SELECT MAX(CAST(json_extract(extra, '$.session_index') AS INTEGER)) "
                        "FROM context_entries WHERE status = 'active'"
                    )
                    row = await cursor.fetchone()
                    max_idx = row[0] if row and row[0] is not None else 0
            except Exception:
                max_idx = max(session_groups.keys()) if session_groups else 0

            total_sessions = max_idx + 1
            covered_count = len(session_groups)

            # If coverage < 60%, batch-fetch entries from uncovered sessions
            if covered_count < total_sessions * 0.6:
                uncovered_ids = set(range(total_sessions)) - set(session_groups.keys())
                existing_ids = {r.entry.id for r in results}

                # Limit to top 15 uncovered sessions to avoid pulling too much data
                for si in sorted(uncovered_ids)[:15]:
                    try:
                        sibling_entries = await self._fetch_entries_by_session_index(si)
                        for entry in sibling_entries:
                            if entry.id not in existing_ids:
                                existing_ids.add(entry.id)
                                results.append(SearchResult(
                                    entry=entry,
                                    score=0.3,
                                    matched_by="session-completion",
                                ))
                    except Exception:
                        continue

                # Rebuild session_groups after adding completed sessions
                session_groups.clear()
                summary_entries.clear()
                fact_entries.clear()
                for r in results:
                    extra = r.entry.extra or {}
                    tags = r.entry.tags or []
                    session_idx = extra.get("session_index", 0)
                    if session_idx not in session_groups:
                        session_groups[session_idx] = []
                    session_groups[session_idx].append(r)
                    if "session_summary" in tags:
                        summary_entries[session_idx] = r
                    if "fact" in tags:
                        if session_idx not in fact_entries:
                            fact_entries[session_idx] = []
                        fact_entries[session_idx].append(r)

        # If still only one session found after completion, return as-is
        if len(session_groups) <= 1:
            return results

        # Sort each group by score (descending)
        for group in session_groups.values():
            group.sort(key=lambda r: r.score, reverse=True)

        # Interleave results from different sessions
        interleaved = []
        max_per_session = max(2, query.top_k // max(len(session_groups), 1))
        for session_idx in sorted(session_groups.keys()):
            interleaved.extend(session_groups[session_idx][:max_per_session])

        # For covered sessions with summary entries, boost them
        # so they appear earlier in the interleaved list
        for session_idx, summary_r in summary_entries.items():
            if session_idx in session_groups:
                group = session_groups[session_idx]
                if summary_r in group:
                    group.remove(summary_r)
                    group.insert(0, summary_r)

        # For covered sessions with fact entries, move them after summary
        for session_idx, facts in fact_entries.items():
            if session_idx in session_groups:
                group = session_groups[session_idx]
                for fact_r in facts:
                    if fact_r in group:
                        group.remove(fact_r)
                        insert_pos = 1 if session_idx in summary_entries else 0
                        group.insert(insert_pos, fact_r)

        return interleaved if interleaved else results

    async def _fetch_session_linked_memories(
        self,
        seed_context_ids: list[str],
        exclude_ids: set[str],
    ) -> list[ContextEntry]:
        """D3: Given seed context_ids, return extracted memories linked to the
        same sessions via session_memory_links.

        The seeds are context_ids already present in the result set. We first
        resolve their session_ids, then fetch sibling context_ids from the
        same sessions, filtering out anything in ``exclude_ids``.
        """
        if self._store is None or not seed_context_ids:
            return []

        # Resolve session_ids for seed entries, then fetch sibling context_ids
        placeholders = ",".join("?" for _ in seed_context_ids)
        try:
            async with self._store._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    f"SELECT DISTINCT session_id FROM session_memory_links "
                    f"WHERE context_id IN ({placeholders})",
                    seed_context_ids,
                )
                sid_rows = await cursor.fetchall()
                session_ids = [r[0] for r in sid_rows]
                if not session_ids:
                    return []

                sid_placeholders = ",".join("?" for _ in session_ids)
                cursor = await conn.execute(
                    f"SELECT DISTINCT context_id FROM session_memory_links "
                    f"WHERE session_id IN ({sid_placeholders})",
                    session_ids,
                )
                ctx_rows = await cursor.fetchall()
                linked_ids = [r[0] for r in ctx_rows if r[0] not in exclude_ids]
                if not linked_ids:
                    return []
        except Exception as e:
            logger.debug(f"session_memory_links query failed: {e}")
            return []

        entries: list[ContextEntry] = []
        for cid in linked_ids:
            try:
                entry = await self._store.get(cid)
                if entry is not None and entry.status == EntryStatus.ACTIVE:
                    entries.append(entry)
            except Exception:
                continue
        return entries

    @staticmethod
    def get_aggregation_prompt_suffix() -> str:
        """D2: Return a prompt suffix for aggregation-aware answer generation.

        This suffix should be appended to the answer prompt when the
        query is detected as aggregation-type. It instructs the LLM
        to enumerate each instance methodically before reporting a total.
        """
        return """
AGGREGATION INSTRUCTION:
- This question requires counting or totaling across multiple entries.
- You MUST enumerate EACH instance individually before reporting a total.
- Method:
  1. Go through each session/entry in date order
  2. Find every relevant mention
  3. List each one explicitly
  4. Count them all
  5. Report the final total
- Double-check your count against all the history provided.
- If you cannot find enough information, say so explicitly."""

    async def _fetch_entries_by_session_index(
        self, session_index: int
    ) -> list[ContextEntry]:
        """Fetch all active entries for a given session_index from the store.

        Uses a direct SQLite query on the extra JSON column instead of
        index.search(), avoiding vector embedding API calls entirely.
        """
        if not self._store:
            return []

        try:
            async with self._store._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    "SELECT id FROM context_entries "
                    "WHERE status = ? AND extra LIKE ?",
                    ("active", f'%"session_index": {session_index}%'),
                )
                rows = await cursor.fetchall()
                entry_ids = [r[0] for r in rows]
        except Exception as e:
            logger.debug(f"_fetch_entries_by_session_index query failed: {e}")
            return []

        entries: list[ContextEntry] = []
        for eid in entry_ids:
            try:
                entry = await self._store.get(eid)
                if entry is not None:
                    entries.append(entry)
            except Exception:
                continue
        return entries

    async def _session_co_retrieve(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        """D1 Session Co-Retrieval: supplement sibling turns from same sessions.

        When a search result comes from a specific session, other turns
        from the same session may provide essential context (e.g., the
        question is about an entity mentioned in turn 3, but turn 1
        provides the date/setting needed to answer correctly).

        Strategy:
        1. Identify sessions that have hits but with sparse coverage
           (fewer than ``co_retrieve_min_turns`` turns in results).
        2. For those sessions, search the index with the session tag
           to pull in missing sibling turns.
        3. Supplement results with co-retrieved turns, marked with
           a slightly lower score than the original hits.
        4. Respect top_k limit — never exceed query.top_k * 1.5.

        Co-retrieval activates for all query types when session_co_retrieval
        is enabled — even generic queries benefit from sibling-turn context.
        """
        if not self._retrieval_config.session_co_retrieval:
            return results

        # Session co-retrieval is useful for all query types, not just
        # multi-session / temporal-reasoning — even generic queries benefit
        # from sibling-turn context when hits come from sparse sessions.

        co_retrieve_min = self._retrieval_config.co_retrieve_min_turns
        max_total = int(query.top_k * 1.5)

        # Group existing results by session
        session_hits: dict[int, list[SearchResult]] = {}
        for r in results:
            extra = r.entry.extra or {}
            si = extra.get("session_index")
            if si is not None:
                if si not in session_hits:
                    session_hits[si] = []
                session_hits[si].append(r)

        if not session_hits:
            return results

        # Identify sessions with sparse coverage
        sparse_sessions: list[int] = []
        for si, hits in session_hits.items():
            # Count only turn entries (not summary/fact)
            turn_count = sum(
                1 for h in hits
                if not any(t in (h.entry.tags or []) for t in ("session_summary", "fact"))
            )
            if turn_count < co_retrieve_min:
                sparse_sessions.append(si)

        if not sparse_sessions:
            return results

        # Cap the number of sparse sessions to avoid excessive API calls.
        # Only supplement the most relevant (first-found) sessions.
        max_sparse = 10
        if len(sparse_sessions) > max_sparse:
            sparse_sessions = sparse_sessions[:max_sparse]

        # Supplement: search for sibling turns in sparse sessions.
        # Use FTS-only search (no vector embedding) via the store's
        # session_memory_links table for speed — avoids N serial
        # index.search() calls each triggering an embedding API round-trip.
        existing_ids = {r.entry.id for r in results}
        supplemented: list[SearchResult] = []

        for si in sparse_sessions:
            try:
                # Fast path: direct SQLite query by session_index in extra JSON.
                # This avoids index.search() which triggers vector embedding API.
                sibling_entries = await self._fetch_entries_by_session_index(si)
                for entry in sibling_entries:
                    if entry.id in existing_ids:
                        continue
                    sr = SearchResult(
                        entry=entry,
                        score=0.5,  # Co-retrieved turns get lower score
                        matched_by="co-retrieval",
                    )
                    supplemented.append(sr)
                    existing_ids.add(entry.id)
            except Exception as e:
                logger.debug(f"Session co-retrieval failed for session {si}: {e}")
                continue

        if not supplemented:
            return results

        # Merge: original results first, then supplemented
        merged = list(results) + supplemented
        # Sort by score (original results have higher scores naturally)
        merged.sort(key=lambda r: r.score, reverse=True)

        # Respect max_total limit
        return merged[:max_total]

    @staticmethod
    def _preference_strategy(
        results: list[SearchResult], query: SearchQuery
    ) -> list[SearchResult]:
        """Preference queries: boost user-uttered entries with preference signals.

        Strategy:
        1. Strongly boost user-uttered entries (they contain preference statements)
        2. Boost preference-category entries
        3. Boost entries that contain preference-indicating content
        4. Include assistant entries that reference user preferences
        5. P2: Also boost entries related to user's mentioned entities/products
           (implicit preference indicators)
        """
        # Keywords that often indicate preference-related content
        pref_indicators = [
            "love", "hate", "enjoy", "favorite", "favourite", "like",
            "prefer", "really into", "big fan", "not a fan", "obsessed",
            "can't stand", "always", "never", "usually", "typically",
            "my go-to", "i tend to", "i tend not to",
            # P2: Broader preference indicators
            "bought", "purchased", "own", "using", "use",
            "switched", "tried", "looking for", "need a",
            "want", "looking to", "considering", "decided",
            "good", "great", "bad", "terrible", "amazing",
            "best", "worst", "better", "worse",
            "recommend", "suggest", "advice",
        ]

        for r in results:
            tags = r.entry.tags or []
            extra = r.entry.extra or {}
            content = (r.entry.l2_full or "").lower()

            # Boost user role entries — they contain preference statements
            if "user" in tags or extra.get("role") == "user":
                r.score *= 1.8  # boosted from 1.5 — user utterances are critical

            # Boost preference-category entries
            if r.entry.memory_category and r.entry.memory_category.value == "preference":
                r.score *= 2.0  # boosted from 1.8

            # Boost entries containing preference indicators
            matched_indicators = sum(1 for indicator in pref_indicators if indicator in content)
            if matched_indicators > 0:
                # More indicators = stronger signal
                r.score *= (1.0 + 0.3 * min(matched_indicators, 3))  # capped at 1.9x

            # P2: Boost event-category entries from user turns (they often describe
            # activities/choices that reveal implicit preferences)
            if (r.entry.memory_category and r.entry.memory_category.value == "event"
                    and ("user" in tags or extra.get("role") == "user")):
                r.score *= 1.4

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _get_include_statuses(self, query: SearchQuery) -> list[str] | None:
        """Convert SearchQuery.include_statuses to a list of status strings."""
        if query.include_statuses is not None:
            return [s.value for s in query.include_statuses]
        return None  # None means active-only (default)

    @staticmethod
    def _get_temporal_filter(query: SearchQuery) -> tuple[str | None, str | None, str | None]:
        """Extract date_from/date_to/speaker from query as ISO-format strings.

        date_from/date_to are datetime objects in SearchQuery but the index
        expects ISO-format strings for SQLite comparison with ce.created_at.
        """
        return (
            query.date_from.isoformat() if query.date_from else None,
            query.date_to.isoformat() if query.date_to else None,
            query.speaker,
        )

    async def _hybrid_search(self, query: SearchQuery, trace: RetrievalTrace) -> list[SearchResult]:
        """Standard hybrid search: FTS + vector + RRF."""
        step = TraceStep(step="hybrid_search", input_summary=query.text)
        step_start = time.monotonic()

        all_results: list[SearchResult] = []

        if self._intent_analyzer and query.strategy == SearchStrategy.HYBRID:
            # Intent-aware search (LLM): decompose query, search per sub-query
            sub_queries = await self._intent_analyzer.analyze(query.text)
            date_from, date_to, speaker = self._get_temporal_filter(query)
            for sq in sub_queries:
                results = await self._index.search(
                    query=sq.query,
                    top_k=query.top_k * 2,
                    context_type=sq.context_type.value,
                    scope=query.scope.value if query.scope else None,
                    owner_id=query.owner_id,
                    include_statuses=self._get_include_statuses(query),
                    query_type=query.query_type,
                    date_from=date_from,
                    date_to=date_to,
                    speaker=speaker,
                )
                all_results.extend(results)

            # Deduplicate by entry ID
            seen = set()
            unique = []
            for r in all_results:
                if r.entry.id not in seen:
                    seen.add(r.entry.id)
                    unique.append(r)
            all_results = unique
        elif self._query_decomposer is not None:
            # Local rule-based query decomposition (zero-LLM fallback)
            sub_query_texts = self._query_decomposer.decompose(
                query.text, query_type=query.query_type
            )
            date_from, date_to, speaker = self._get_temporal_filter(query)
            for sq_text in sub_query_texts:
                results = await self._index.search(
                    query=sq_text,
                    top_k=query.top_k * 2,
                    context_type=query.context_type.value if query.context_type else None,
                    scope=query.scope.value if query.scope else None,
                    owner_id=query.owner_id,
                    include_statuses=self._get_include_statuses(query),
                    query_type=query.query_type,
                    date_from=date_from,
                    date_to=date_to,
                    speaker=speaker,
                )
                all_results.extend(results)

            # Deduplicate by entry ID
            seen = set()
            unique = []
            for r in all_results:
                if r.entry.id not in seen:
                    seen.add(r.entry.id)
                    unique.append(r)
            all_results = unique
        else:
            # Simple hybrid search (no decomposition)
            date_from, date_to, speaker = self._get_temporal_filter(query)
            all_results = await self._index.search(
                query=query.text,
                top_k=query.top_k * 3,
                context_type=query.context_type.value if query.context_type else None,
                scope=query.scope.value if query.scope else None,
                owner_id=query.owner_id,
                include_statuses=self._get_include_statuses(query),
                query_type=query.query_type,
                date_from=date_from,
                date_to=date_to,
                speaker=speaker,
            )

        step.candidates_in = 0
        step.candidates_out = len(all_results)
        step.latency_ms = (time.monotonic() - step_start) * 1000
        trace.add_step(step)

        # --- NER-aware query expansion + result boosting ---
        if self._ner_extractor is not None and all_results:
            all_results = await self._ner_boost_search(
                query, all_results, date_from, date_to, speaker,
            )

        return all_results

    async def _ner_boost_search(
        self,
        query: SearchQuery,
        existing_results: list[SearchResult],
        date_from: str | None,
        date_to: str | None,
        speaker: str | None,
    ) -> list[SearchResult]:
        """NER-aware query expansion: extract entities from query and boost
        existing results whose tags match those entities.

        With NER entity tagging (not separate entries), this only boosts
        results that already appear in the FTS+Vector results but have
        NER-related tags matching query entities. No new results are added,
        preventing dilution of the original result set.

        Degradation: NerExtractor returns empty → skip, keep FTS+Vector.
        """
        # Step 1: Extract entities from the query
        try:
            query_entities = self._ner_extractor.extract(query.text)
        except Exception as e:
            logger.debug(f"NER extraction from query failed: {e}")
            return existing_results

        if not query_entities:
            return existing_results

        # Build lookup sets for entity matching
        entity_texts = [ent["content"].strip() for ent in query_entities if ent.get("content", "").strip()]
        if not entity_texts:
            return existing_results

        entity_lower = {et.lower() for et in entity_texts}
        # Also build tag-form matches (ner_<entity> with spaces→underscores)
        entity_tag_lower: set[str] = set()
        for et in entity_texts:
            tag_name = "ner_" + "_".join(
                ch if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" else "_"
                for ch in et
            ).strip("_").lower()
            entity_tag_lower.add(tag_name)

        # Step 2: Boost existing results whose tags match query entities
        for r in existing_results:
            tags = r.entry.tags or []
            tags_lower = {t.lower() for t in tags}
            content_lower = (r.entry.l0_abstract or r.entry.l2_full or "").lower()

            has_ner_tag = "ner" in tags_lower
            matches_entity_tag = bool(tags_lower & entity_tag_lower)
            matches_content = any(el in content_lower for el in entity_lower if len(el) > 2)

            if matches_entity_tag:
                # Strongest signal: NER tag matches a query entity
                r.score *= (1.0 + self._ner_weight)
                r.matched_by = "ner+hybrid"
            elif has_ner_tag and matches_content:
                # NER-tagged entry with entity text in content
                r.score *= (1.0 + self._ner_weight * 0.8)
                r.matched_by = "ner+hybrid"
            elif matches_content:
                # Entity text appears in content (no NER tag needed)
                r.score *= (1.0 + self._ner_weight * 0.5)

        # Re-sort by score after boosting
        existing_results.sort(key=lambda r: r.score, reverse=True)

        return existing_results

    async def _hierarchical_search(self, query: SearchQuery, trace: RetrievalTrace) -> list[SearchResult]:
        """Hierarchical progressive search: L0 → L1 → L2."""
        # Step 1: L0 coarse filter
        l0_step = TraceStep(step="l0_search", input_summary=query.text)
        l0_start = time.monotonic()

        date_from, date_to, speaker = self._get_temporal_filter(query)
        l0_results = await self._index.search(
            query=query.text,
            top_k=query.top_k * 3,  # Over-fetch for L0
            context_type=query.context_type.value if query.context_type else None,
            scope=query.scope.value if query.scope else None,
            owner_id=query.owner_id,
            fts_column="l0_abstract",  # Only search L0 column
            include_statuses=self._get_include_statuses(query),
            query_type=query.query_type,
            date_from=date_from,
            date_to=date_to,
            speaker=speaker,
        )

        l0_step.candidates_in = 0
        l0_step.candidates_out = len(l0_results)
        l0_step.latency_ms = (time.monotonic() - l0_start) * 1000
        trace.add_step(l0_step)

        if not l0_results:
            return []

        # Step 2: L1 refinement (optional, load L1 overview)
        l1_step = TraceStep(step="l1_rerank")
        l1_start = time.monotonic()

        # Re-search on L1 column for better precision
        l1_results = await self._index.search(
            query=query.text,
            top_k=query.top_k * 2,
            context_type=query.context_type.value if query.context_type else None,
            scope=query.scope.value if query.scope else None,
            owner_id=query.owner_id,
            fts_column="l1_overview",  # Only search L1 column
            include_statuses=self._get_include_statuses(query),
            query_type=query.query_type,
            date_from=date_from,
            date_to=date_to,
            speaker=speaker,
        )

        # Merge L0 and L1 results (L1 results take priority)
        l0_ids = {r.entry.id for r in l0_results}
        merged = list(l1_results)
        for r in l0_results:
            if r.entry.id not in {mr.entry.id for mr in merged}:
                merged.append(r)

        l1_step.candidates_in = len(l0_results)
        l1_step.candidates_out = len(merged)
        l1_step.latency_ms = (time.monotonic() - l1_start) * 1000
        trace.add_step(l1_step)

        # Step 3: L2 on-demand load (done in _apply_load_level)
        l2_step = TraceStep(step="l2_load")
        l2_step.candidates_in = len(merged)
        l2_step.candidates_out = min(len(merged), query.top_k)
        trace.add_step(l2_step)

        return merged

    async def _fts_only_search(self, query: SearchQuery, trace: RetrievalTrace) -> list[SearchResult]:
        """FTS-only search (no vector)."""
        date_from, date_to, speaker = self._get_temporal_filter(query)
        return await self._index.search(
            query=query.text,
            top_k=query.top_k * 3,
            context_type=query.context_type.value if query.context_type else None,
            scope=query.scope.value if query.scope else None,
            owner_id=query.owner_id,
            include_statuses=self._get_include_statuses(query),
            query_type=query.query_type,
            date_from=date_from,
            date_to=date_to,
            speaker=speaker,
        )

    async def _vector_only_search(self, query: SearchQuery, trace: RetrievalTrace) -> list[SearchResult]:
        """Vector-only search."""
        if not self._index.vec_available:
            logger.warning("Vector search not available, falling back to FTS")
            results = await self._fts_only_search(query, trace)
            for r in results:
                r.degrade_reason = "vec_unavailable"
            return results

        date_from, date_to, speaker = self._get_temporal_filter(query)
        return await self._index.search(
            query=query.text,
            top_k=query.top_k * 3,
            context_type=query.context_type.value if query.context_type else None,
            scope=query.scope.value if query.scope else None,
            owner_id=query.owner_id,
            query_type=query.query_type,
            date_from=date_from,
            date_to=date_to,
            speaker=speaker,
        )

    @staticmethod
    def _apply_load_level(results: list[SearchResult], level: str) -> list[SearchResult]:
        """Apply load level — trim content based on specified level."""
        for r in results:
            r.loaded_level = level
            # Content is already loaded; the load_level just indicates
            # which fields the consumer should read
        return results

    def _apply_budget(self, results: list[SearchResult], budget: int) -> list[SearchResult]:
        """Trim results to fit within token budget, with content deduplication."""
        tb = TokenBudget(budget=budget)
        trimmed = []
        seen_content_keys: set[str] = set()
        for r in results:
            # Content deduplication: skip near-duplicate entries
            # Use 150-char fingerprint (not 80) to avoid false dedup of
            # entries that start similarly but contain different facts
            content = r.entry.l2_full or r.entry.l1_overview or r.entry.l0_abstract or ""
            content_key = content.strip().lower()[:150]
            if content_key and content_key in seen_content_keys:
                continue
            if content_key:
                seen_content_keys.add(content_key)
            if tb.try_allocate(content):
                trimmed.append(r)
            else:
                break
        return trimmed
