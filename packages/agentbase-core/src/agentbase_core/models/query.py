"""SearchQuery / SearchResult / SearchStrategy — retrieval models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .context_entry import ContextEntry, ContextScope, ContextType, EntryStatus


class SearchStrategy(str):
    """Search strategy enumeration."""

    FTS = "fts"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class SearchQuery(BaseModel):
    """Search request model."""

    text: str
    top_k: int = Field(default=20, ge=1, le=300)
    strategy: str = SearchStrategy.HYBRID
    context_type: ContextType | None = None
    scope: ContextScope | None = None
    owner_id: str | None = None
    tags: list[str] | None = None
    min_confidence: float | None = None
    token_budget: int | None = None
    load_level: str = "auto"  # auto/l0/l1/l2
    include_trace: bool = False
    include_embedding: bool = False
    include_statuses: list[EntryStatus] | None = None  # None = active only
    query_type: str | None = None  # temporal-reasoning / multi-session / knowledge-update / single-session-* / preference

    # Time-aware retrieval — filter entries by their valid_from / created_at date range.
    # These filter on the entry-level temporal fields set during ingestion
    # (e.g., session_date → entry.created_at).
    date_from: datetime | None = None  # inclusive lower bound
    date_to: datetime | None = None    # inclusive upper bound

    # Speaker/role-aware retrieval — filter entries by the speaker/role tag.
    # Matches against tags (e.g., "speaker:Caroline") or extra.role field.
    speaker: str | None = None  # case-insensitive match against role/speaker tag


class SearchResult(BaseModel):
    """Single search result."""

    entry: ContextEntry
    score: float = Field(default=0.0, ge=0.0)
    score_breakdown: dict[str, float] | None = None
    ranking_stage: str = "final"  # fts/vector/rrf/heuristic/llm
    matched_by: str = "hybrid"  # hybrid/fts/vector/hierarchical
    loaded_level: str = "l2"
    degrade_reason: str | None = None  # vec_unavailable/embedding_failed/query_embed_failed
    trace: Any | None = None  # RetrievalTrace — avoids circular import
