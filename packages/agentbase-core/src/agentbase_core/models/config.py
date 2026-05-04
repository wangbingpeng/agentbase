"""AgentBaseConfig — root configuration model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_base: str | None = None
    api_key: str | None = None
    max_concurrent: int = 10


class LLMConfig(BaseModel):
    """LLM configuration."""

    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024


class IndexConfig(BaseModel):
    """Index configuration."""

    vector_enabled: bool = True  # 向量检索开关，默认开启；若 embedder 不可用则自动降级为纯 FTS
    tokenizer: str = "auto"  # 分词器: "auto"（自动检测语言）| "jieba" | "char"
    fts_weight: float = 0.5
    vec_weight: float = 0.5
    rrf_k: int = 60
    dedup_threshold: float = 0.92


class GraphConfig(BaseModel):
    """Graph configuration."""

    enabled: bool = True  # 默认开启；若 LLM 不可用则 EntityService 仍可用（CRUD），EntityExtractor 自动降级跳过
    max_traversal_depth: int = 4
    max_entities: int = 10000
    max_relations: int = 50000
    extract_on_ingest: bool = False  # 默认关闭；开启后 add_conversation/ingest_direct 时自动提取实体和关系


class SessionConfig(BaseModel):
    """Session configuration."""

    enabled: bool = True  # 默认开启；若 LLM 不可用则 MemoryExtractor/SessionCompressor 自动降级跳过
    keep_recent_turns: int = 6
    auto_commit: bool = False
    extract_memories: bool = True
    extract_on_ingest: bool = False  # 默认关闭；开启后 add_conversation 时自动创建 Session 并提交提取记忆


class IngestConfig(BaseModel):
    """Ingest pipeline configuration."""

    session_summary: bool = True       # LLM session summary, degrades to truncation
    fact_extraction: bool = True       # LLM fact extraction, degrades to local regex
    ner_extraction: bool = True        # NER entity extraction (bilingual)
    extract_on_direct_ingest: bool = False  # Also extract on ingest_direct


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    default_top_k: int = 20
    default_token_budget: int = 24000
    freshness_half_life_days: float = 7.0
    knowledge_update_half_life_days: float = 14.0  # shortened from 30: stronger recency for knowledge-update
    query_decomposition: bool = True   # Local rule-based query decomposition
    ner_boost: bool = True            # NER-aware query expansion + result boosting
    ner_weight: float = 0.3           # NER signal weight in three-way fusion

    # D1: Session Co-Retrieval
    session_co_retrieval: bool = True  # Enable session co-retrieval for all query types
    co_retrieve_min_turns: int = 2     # Min turns per session before co-retrieval activates (lowered from 3)

    # D2: Aggregation-Aware
    agg_top_k: int = 120              # top_k for aggregation queries (raised from 80)
    agg_detection: bool = True         # Enable aggregation query detection + top_k boost


class TierConfig(BaseModel):
    """Layered context configuration (L0/L1/L2)."""

    enabled: bool = True
    async_generation: bool = True
    max_concurrent: int = 5
    fallback_to_truncation: bool = True


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    enabled: bool = False
    persist_traces: bool = True
    trace_sample_rate: float = 1.0
    max_trace_age_days: int = 30


class AgentBaseConfig(BaseSettings):
    """Root configuration for AgentBase."""

    data_dir: Path = Field(default=Path.home() / ".agentbase")
    db_filename: str = "agentbase.db"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tier: TierConfig = Field(default_factory=TierConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    model_config = {
        "env_prefix": "AGENTBASE_",
        "env_nested_delimiter": "__",
    }

    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database file."""
        return self.data_dir / self.db_filename

    def ensure_data_dir(self) -> None:
        """Create the data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: Path | str) -> AgentBaseConfig:
        """Load configuration from a YAML file.

        Environment variables take precedence over YAML values.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Save current configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        # Remove None values for cleaner output
        data = {k: v for k, v in data.items() if v is not None}

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
