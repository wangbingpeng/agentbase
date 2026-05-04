"""AgentBase exception hierarchy."""


class AgentBaseError(Exception):
    """Base exception for AgentBase."""


class StorageError(AgentBaseError):
    """SQLite operation failed."""


class IndexOpError(AgentBaseError):
    """Index operation failed (FTS5/sqlite-vec)."""


class EmbeddingError(AgentBaseError):
    """Embedding generation failed."""


class LLMError(AgentBaseError):
    """LLM call failed."""


class GraphError(AgentBaseError):
    """Graph operation failed."""


class SessionError(AgentBaseError):
    """Session operation failed."""


class ConfigError(AgentBaseError):
    """Configuration error."""


class ConflictError(AgentBaseError):
    """Fact conflict requires human intervention."""


class BackgroundJobError(AgentBaseError):
    """Background job execution failed."""


class ValidationError(AgentBaseError):
    """Input validation error (scope/owner_id constraint, etc.)."""
