"""AgentBase SDK — public API.

Usage:
    from agentbase import AgentBase

    db = AgentBase(path="./my_context.db")
    await db.initialize()

    # Add memory
    entry = await db.add_memory("User prefers Python 3.12", category="preference")

    # Search
    results = await db.find("Python preferences", top_k=5)

    # Entity
    entity = await db.add_entity(name="Python", entity_type="technology")

    # Session
    session = await db.create_session(agent_id="my-agent")

    await db.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentbase_core.engine import AgentBaseEngine
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


class AgentBase:
    """AgentBase Python SDK — the primary public interface.

    This is a thin wrapper around AgentBaseEngine that provides
    the user-friendly API described in the SPEC.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        config: AgentBaseConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self._config = config
        elif path is not None:
            path = Path(path)
            self._config = AgentBaseConfig(
                data_dir=path.parent if path.suffix else path,
                db_filename=path.name if path.suffix else "agentbase.db",
            )
        else:
            self._config = AgentBaseConfig()

        self._engine = AgentBaseEngine(config=self._config, **kwargs)

    async def initialize(self) -> None:
        """Initialize the database and all subsystems."""
        await self._engine.initialize()

    async def close(self) -> None:
        """Close all connections and shut down."""
        await self._engine.close()

    async def __aenter__(self) -> AgentBase:
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # --- Write operations ---

    async def add_memory(
        self,
        content: str,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
    ) -> ContextEntry:
        """Add a memory context entry."""
        return await self._engine.add_memory(
            content=content,
            category=category,
            tags=tags,
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
        )

    async def add_conversation(
        self,
        turns: list[dict[str, str]],
        session_date: str | None = None,
        session_index: int = 0,
        tags: list[str] | None = None,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
    ) -> list[ContextEntry]:
        """Add a multi-turn conversation as per-turn memory entries.

        Each turn dict should have keys: "role" (user/assistant) and "content".
        Optional: "date" for per-turn timestamp override.

        Returns list of created ContextEntry objects.
        """
        return await self._engine.add_conversation(
            turns=turns,
            session_date=session_date,
            session_index=session_index,
            tags=tags,
            scope=scope,
            owner_id=owner_id,
        )

    async def add_resource(
        self,
        url: str | None = None,
        content: str = "",
        format: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        reason: str = "",
    ) -> ContextEntry:
        """Add a resource context entry."""
        return await self._engine.add_resource(
            url=url,
            content=content,
            format=format,
            tags=tags,
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
            reason=reason,
        )

    async def add_skill(
        self,
        tool_name: str,
        description: str = "",
        api_spec: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
    ) -> ContextEntry:
        """Add a skill context entry."""
        return await self._engine.add_skill(
            tool_name=tool_name,
            description=description,
            api_spec=api_spec,
            tags=tags,
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
        )

    # --- Read operations ---

    async def get(self, entry_id: str, load_level: str = "l2") -> ContextEntry | None:
        """Get a context entry by ID."""
        return await self._engine.get(entry_id, load_level=load_level)

    async def find(
        self,
        query: str,
        top_k: int | None = None,
        context_type: ContextType | None = None,
        scope: ContextScope | None = None,
        owner_id: str | None = None,
        token_budget: int | None = None,
        include_trace: bool = False,
        query_type: str | None = None,
    ) -> list[SearchResult]:
        """Search for context entries (convenience alias)."""
        return await self._engine.find(
            query=query,
            top_k=top_k,
            context_type=context_type,
            scope=scope,
            owner_id=owner_id,
            token_budget=token_budget,
            include_trace=include_trace,
            query_type=query_type,
        )

    async def search(
        self,
        query: str | SearchQuery,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search for context entries."""
        if isinstance(query, str):
            query = SearchQuery(text=query, **kwargs)
        return await self._engine.search(query)

    # --- Update operations ---

    async def update(
        self,
        entry_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
    ) -> ContextEntry | None:
        """Update a context entry by ID.

        Only the provided fields will be updated; others remain unchanged.
        Returns the updated entry, or None if the entry was not found.
        """
        entry = await self._engine.get(entry_id)
        if entry is None:
            return None
        if content is not None:
            entry.l2_full = content
        if tags is not None:
            entry.tags = tags
        if confidence is not None:
            entry.confidence = confidence
        await self._engine.update(entry)
        return entry

    # --- Delete operations ---

    async def delete(self, entry_id: str) -> bool:
        """Soft delete a context entry."""
        return await self._engine.delete(entry_id)

    async def purge(self, entry_id: str) -> bool:
        """Hard delete a context entry."""
        return await self._engine.purge(entry_id)

    async def delete_all(
        self,
        owner_id: str | None = None,
        scope: ContextScope | None = None,
    ) -> int:
        """Delete all entries matching the given filters.

        Returns the number of entries deleted.
        """
        entries = await self._engine.list_entries(
            scope=scope,
            limit=10000,
        )
        # Filter by owner_id if specified
        if owner_id is not None:
            entries = [e for e in entries if e.owner_id == owner_id]
        count = 0
        for entry in entries:
            if await self._engine.delete(entry.id):
                count += 1
        return count

    # --- List operations ---

    async def list_entries(
        self,
        scope: ContextScope | None = None,
        context_type: ContextType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextEntry]:
        """List context entries."""
        return await self._engine.list_entries(
            scope=scope,
            context_type=context_type,
            limit=limit,
            offset=offset,
        )

    async def count(
        self,
        scope: ContextScope | None = None,
        context_type: ContextType | None = None,
    ) -> int:
        """Count context entries."""
        return await self._engine.count(scope=scope, context_type=context_type)

    # --- Ingest ---

    async def ingest_text(
        self,
        text: str,
        context_type: ContextType = ContextType.MEMORY,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ContextEntry]:
        """Ingest raw text via LLM extraction."""
        return await self._engine.ingest_text(
            text=text,
            context_type=context_type,
            scope=scope,
            owner_id=owner_id,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Entity / Graph operations
    # ------------------------------------------------------------------

    async def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: str = "",
        properties: dict | None = None,
    ) -> Entity:
        """Add or merge an entity."""
        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
        )
        return await self._engine.add_entity(entity)

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return await self._engine.get_entity(entity_id)

    async def find_entities(self, name: str, entity_type: str | None = None) -> list[Entity]:
        """Find entities by name."""
        return await self._engine.find_entities(name, entity_type=entity_type)

    async def add_alias(self, entity_id: str, alias: str) -> None:
        """Add an alias for entity disambiguation."""
        await self._engine.add_alias(entity_id, alias)

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        confidence: float = 1.0,
    ) -> Relation:
        """Add a relation between entities."""
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            predicate=predicate,
            confidence=confidence,
        )
        return await self._engine.add_relation(relation)

    async def get_current_relations(self, entity_id: str) -> list[Relation]:
        """Get current (valid) relations for an entity."""
        return await self._engine.get_current_relations(entity_id)

    async def graph_traversal(self, entity_name: str, depth: int = 2) -> list[dict]:
        """Traverse the knowledge graph from an entity."""
        return await self._engine.graph_traversal(entity_name, depth=depth)

    async def add_fact(self, entity_id: str, fact: str) -> None:
        """Add a fact to an entity's timeline."""
        ft = FactTimeline(entity_id=entity_id, fact=fact)
        await self._engine.add_fact(ft)

    async def get_current_facts(self, entity_id: str) -> list[FactTimeline]:
        """Get current (non-superseded) facts for an entity."""
        return await self._engine.get_current_facts(entity_id)

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self,
        agent_id: str = "default",
        project: str | None = None,
    ) -> Session:
        """Create a new conversation session."""
        return await self._engine.create_session(agent_id=agent_id, project=project)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> SessionMessage:
        """Add a message to a session."""
        return await self._engine.add_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    async def get_session(self, session_id: str, load_messages: bool = False) -> Session | None:
        """Get a session by ID."""
        return await self._engine.get_session(session_id, load_messages=load_messages)

    async def commit_session(
        self,
        session_id: str,
        mode: str = "full",
    ) -> list[ContextEntry]:
        """Commit a session: compress, archive, and extract memories."""
        return await self._engine.commit_session(session_id, mode=mode)

    # ------------------------------------------------------------------
    # Observability operations
    # ------------------------------------------------------------------

    async def get_metrics(self) -> dict[str, Any]:
        """Get context quality metrics."""
        return await self._engine.get_metrics()

    async def explain_query(self, query: str) -> dict:
        """Explain what a retrieval query would do (dry run)."""
        return await self._engine.explain_query(query)
