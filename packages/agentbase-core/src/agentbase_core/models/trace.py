"""RetrievalTrace / TraceStep — observability trace models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .context_entry import _new_ulid, _utcnow


class TraceStep(BaseModel):
    """A single step in the retrieval pipeline."""

    step: str  # intent_analysis/l0_search/l1_rerank/l2_load/final
    input_summary: str = ""
    candidates_in: int = 0
    candidates_out: int = 0
    latency_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class RetrievalTrace(BaseModel):
    """Complete retrieval trace record."""

    id: str = Field(default_factory=_new_ulid)
    query: str
    strategy: str = "hybrid"
    steps: list[TraceStep] = Field(default_factory=list)
    result_ids: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    token_budget_used: int = 0
    token_budget_limit: int = 0
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"from_attributes": True}

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def finish(self, result_ids: list[str], total_latency_ms: float) -> None:
        self.result_ids = result_ids
        self.total_latency_ms = total_latency_ms
