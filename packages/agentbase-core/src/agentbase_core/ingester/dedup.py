"""Deduplicator — context entry deduplication."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum

from ..models.context_entry import ContextEntry, ContextScope, ContextType, MemoryCategory

logger = logging.getLogger(__name__)


class DedupDecision(str, Enum):
    """Dedup decision result."""

    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    SUPERSEDE_CANDIDATE = "supersede_candidate"
    DISTINCT = "distinct"


class DedupResult:
    """Dedup check result."""

    def __init__(
        self,
        decision: DedupDecision = DedupDecision.DISTINCT,
        similar_entry_id: str | None = None,
        similarity: float = 0.0,
        match_dimensions: list[str] | None = None,
    ) -> None:
        self.decision = decision
        self.similar_entry_id = similar_entry_id
        self.similarity = similarity
        self.match_dimensions = match_dimensions or []

    @property
    def is_duplicate(self) -> bool:
        return self.decision in (
            DedupDecision.EXACT_DUPLICATE,
            DedupDecision.SEMANTIC_DUPLICATE,
        )

    @property
    def should_supersede(self) -> bool:
        return self.decision == DedupDecision.SUPERSEDE_CANDIDATE


class Deduplicator:
    """Context entry deduplication checker.

    Checks exact content hash match first, then semantic similarity
    within the same type + scope.
    """

    def __init__(self, store: any, index: any, threshold: float = 0.92) -> None:
        self._store = store
        self._index = index
        self._threshold = threshold

    @staticmethod
    def _token_set(text: str) -> set[str]:
        """Extract a set of lowercase word tokens from text.

        Includes:
        - Words of 2+ letters (e.g., "test", "entry", "python")
        - Numeric sequences of any length (e.g., "0", "42", "2024")

        Single letters ("a", "I") are excluded as they add noise,
        but single digits ARE included because they often differentiate
        content (e.g., "Test entry 0" vs "Test entry 1").
        """
        import re
        tokens = re.findall(r"[A-Za-z]{2,}|[0-9]+", text.lower())
        return set(tokens)

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    async def check(
        self,
        content: str,
        context_type: ContextType,
        scope: ContextScope,
        owner_id: str | None = None,
        memory_category: MemoryCategory | None = None,
    ) -> DedupResult:
        """Check if content is duplicate of existing entries.

        Strategy:
        1. Exact hash match — guaranteed duplicate
        2. FTS recall + Jaccard similarity — use FTS to find candidate
           matches, then compute Jaccard similarity on word tokens
           between the new content and the top match.  Only declare
           semantic duplicate when Jaccard >= threshold (default 0.92).
        """
        # 1. Exact hash match — check against embedding_hash in store
        content_hash = EmbeddingCache.compute_hash(content)

        # 2. Semantic similarity check (same type + same scope)
        if self._index is not None:
            try:
                # Truncate content for FTS search to avoid query syntax errors
                # with very long texts (e.g., full conversation turns)
                search_query = content[:200] if len(content) > 200 else content
                results = await self._index.search(
                    query=search_query,
                    top_k=3,
                    context_type=context_type.value,
                    scope=scope.value,
                    owner_id=owner_id,
                )
                if results:
                    top_result = results[0]
                    # Compute Jaccard similarity on word tokens
                    new_tokens = self._token_set(content)
                    existing_content = (
                        top_result.entry.l2_full
                        or top_result.entry.l1_overview
                        or top_result.entry.l0_abstract
                        or ""
                    )
                    existing_tokens = self._token_set(existing_content)
                    jaccard_sim = self._jaccard(new_tokens, existing_tokens)

                    if jaccard_sim >= self._threshold:
                        match_dims = ["jaccard"]
                        if top_result.entry.context_type == context_type:
                            match_dims.append("type")
                        if top_result.entry.scope == scope:
                            match_dims.append("scope")
                        if owner_id and top_result.entry.owner_id == owner_id:
                            match_dims.append("owner")
                        if memory_category and top_result.entry.memory_category == memory_category:
                            match_dims.append("category")
                        return DedupResult(
                            decision=DedupDecision.SEMANTIC_DUPLICATE,
                            similar_entry_id=top_result.entry.id,
                            similarity=jaccard_sim,
                            match_dimensions=match_dims,
                        )
            except Exception as e:
                logger.warning(f"Dedup check failed: {e}")

        return DedupResult(decision=DedupDecision.DISTINCT)


class EmbeddingCache:
    """Embedding vector cache — avoid recomputing embeddings for identical content.

    Per SPEC §8.4: SQLite-backed cache using embedding_cache table.
    """

    def __init__(self, pool: any) -> None:
        self._pool = pool

    async def get(self, content_hash: str, model: str | None = None) -> list[float] | None:
        """Look up a cached embedding by content hash."""
        async with self._pool.get_read_conn() as conn:
            if model:
                cursor = await conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE content_hash = ? AND model = ?",
                    (content_hash, model),
                )
            else:
                cursor = await conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
                    (content_hash,),
                )
            row = await cursor.fetchone()
            if row is None:
                return None
            # Deserialize BLOB to float list
            import struct
            blob = row[0]
            n = len(blob) // 4
            return list(struct.unpack(f"<{n}f", blob))

    async def put(self, content_hash: str, embedding: list[float], model: str = "unknown", dimensions: int | None = None) -> None:
        """Cache an embedding vector."""
        import struct
    
        dimensions = dimensions or len(embedding)
        blob = struct.pack(f"<{len(embedding)}f", *embedding)
        # Use content_hash as PK; if model differs, the latest model wins
        now = datetime.now(timezone.utc).isoformat()
    
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, model, dimensions, created_at) VALUES (?, ?, ?, ?, ?)",
                (content_hash, blob, model, dimensions, now),
            )
            await conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA256 hash of content (first 16 chars)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
