"""AgentBase SDK — Context Database for AI Agents."""

from .client import AgentBase
from .adapters import (
    BaseAdapter,
    Mem0Adapter,
    LangChainMemoryAdapter,
    OpenAIAssistantAdapter,
    MinimalAdapter,
    AgentBaseChatStore,
)

# Re-export commonly used models for convenience
from agentbase_core.models import (
    AgentBaseConfig,
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from agentbase_core.models.entity import Entity, FactTimeline, Relation
from agentbase_core.models.session import Session, SessionMessage

__all__ = [
    "AgentBase",
    "AgentBaseConfig",
    "ContextEntry",
    "ContextScope",
    "ContextType",
    "EntryStatus",
    "Entity",
    "FactTimeline",
    "MemoryCategory",
    "Relation",
    "SearchQuery",
    "SearchResult",
    "SearchStrategy",
    "Session",
    "SessionMessage",
    # Adapters
    "BaseAdapter",
    "Mem0Adapter",
    "LangChainMemoryAdapter",
    "OpenAIAssistantAdapter",
    "MinimalAdapter",
    "AgentBaseChatStore",
]