"""AgentBaseEngine — unified entry point for all operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .embedding.base import AbstractEmbedder
from .exceptions import ConfigError, StorageError, ValidationError
from .graph.conflict import ConflictResolver
from .graph.entity_service import EntityService
from .graph.extractor import EntityExtractor
from .index.hybrid import HybridIndex
from .index.sqlite_fts import SQLiteFTSIndex
from .index.sqlite_vec import SQLiteVecIndex
from .ingester.background_jobs import BackgroundJobRunner
from .ingester.pipeline import Ingester
from .llm.base import AbstractLLM
from .models.config import AgentBaseConfig
from .models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    OriginType,
)
from .models.entity import Entity, FactTimeline, Relation
from .models.query import SearchQuery, SearchResult
from .models.session import Session, SessionMessage
from .models.trace import RetrievalTrace
from .observability.observability_service import (
    ContextMetrics,
    DebugService,
    TraceCollector,
)
from .retrieval.engine import RetrievalEngine
from .retrieval.intent import IntentAnalyzer
from .session.session_service import (
    MemoryExtractor,
    SessionCompressor,
    SessionService,
)
from .store.connection import ConnectionPool
from .store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class AgentBaseEngine:
    """Unified engine — orchestrates all AgentBase operations.

    This is the core internal API. The public SDK wraps this class.
    """

    def __init__(
        self,
        config: AgentBaseConfig | None = None,
        db_path: Path | str | None = None,
        llm: AbstractLLM | None = None,
        embedder: AbstractEmbedder | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif db_path is not None:
            self._config = AgentBaseConfig(data_dir=Path(db_path).parent, db_filename=Path(db_path).name)
        else:
            self._config = AgentBaseConfig()

        self._llm = llm
        self._embedder = embedder
        self._pool: ConnectionPool | None = None
        self._store: SQLiteStore | None = None
        self._fts_index: SQLiteFTSIndex | None = None
        self._vec_index: SQLiteVecIndex | None = None
        self._index: HybridIndex | None = None
        self._ingester: Ingester | None = None
        self._retrieval_engine: RetrievalEngine | None = None
        self._entity_service: EntityService | None = None
        self._session_service: SessionService | None = None
        self._trace_collector: TraceCollector | None = None
        self._metrics: ContextMetrics | None = None
        self._debug_service: DebugService | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._conflict_resolver: ConflictResolver | None = None
        self._job_runner: BackgroundJobRunner | None = None
        self._initialized = False

    @property
    def config(self) -> AgentBaseConfig:
        return self._config

    @property
    def store(self) -> SQLiteStore:
        if self._store is None:
            raise StorageError("Engine not initialized. Call initialize() first.")
        return self._store

    @property
    def index(self) -> HybridIndex:
        if self._index is None:
            raise StorageError("Engine not initialized. Call initialize() first.")
        return self._index

    @property
    def ingester(self) -> Ingester:
        if self._ingester is None:
            raise StorageError("Engine not initialized. Call initialize() first.")
        return self._ingester

    @property
    def job_runner(self) -> BackgroundJobRunner:
        """Get the background job runner (per SPEC §9.4)."""
        if self._job_runner is None:
            raise StorageError("Engine not initialized. Call initialize() first.")
        return self._job_runner

    async def initialize(self) -> None:
        """Initialize the engine: create pool, store, indexes, ingester."""
        if self._initialized:
            return

        self._config.ensure_data_dir()

        # Connection pool
        self._pool = ConnectionPool(self._config.db_path)
        await self._pool.initialize()

        # Store
        self._store = SQLiteStore(self._pool)
        await self._store.initialize()

        # FTS Index
        self._fts_index = SQLiteFTSIndex(self._pool, tokenizer=self._config.index.tokenizer)

        # Vector Index (controlled by config.index.vector_enabled, default True)
        if self._config.index.vector_enabled:
            if self._embedder is None:
                logger.warning(
                    "vector_enabled=True but no embedder provided — "
                    "vector search will be disabled; falling back to FTS-only"
                )
                self._vec_index = None
            else:
                try:
                    self._vec_index = SQLiteVecIndex(
                        pool=self._pool,
                        dimensions=self._config.embedding.dimensions,
                    )
                    await self._vec_index.initialize()
                except Exception as e:
                    logger.warning(
                        f"Vector index initialization failed: {e} — "
                        "falling back to FTS-only search"
                    )
                    self._vec_index = None
        else:
            self._vec_index = None

        # Hybrid Index (vector optional based on vector_enabled)
        self._index = HybridIndex(
            fts_index=self._fts_index,
            vec_index=self._vec_index,
            fts_weight=self._config.index.fts_weight,
            vec_weight=self._config.index.vec_weight,
            rrf_k=self._config.index.rrf_k,
            embedder=self._embedder,
        )

        # Retrieval Engine
        self._retrieval_engine = RetrievalEngine(
            index=self._index,
            llm=self._llm,
            fts_weight=self._config.index.fts_weight,
            vec_weight=self._config.index.vec_weight,
            rrf_k=self._config.index.rrf_k,
            retrieval_config=self._config.retrieval,
            store=self._store,
        )

        # Ingester
        self._ingester = Ingester(
            store=self._store,
            index=self._index,
            llm=self._llm,
            embedder=self._embedder,
            tier_config=self._config.tier,
            ingest_config=self._config.ingest,
            dedup_threshold=self._config.index.dedup_threshold,
            tokenizer=self._config.index.tokenizer,
        )

        # Entity Service (feature-flagged, default enabled)
        if self._config.graph.enabled:
            self._entity_service = EntityService(self._pool)
            if self._llm is not None:
                self._entity_extractor = EntityExtractor(llm=self._llm)
                self._conflict_resolver = ConflictResolver(llm=self._llm)
            else:
                logger.info(
                    "graph.enabled=True but no LLM configured — "
                    "EntityService CRUD available, EntityExtractor skipped"
                )
                self._entity_extractor = None
                self._conflict_resolver = None
        else:
            self._entity_service = None
            self._entity_extractor = None
            self._conflict_resolver = None

        # Session Service (feature-flagged, default enabled)
        if self._config.session.enabled:
            self._session_service = SessionService(
                pool=self._pool,
                llm=self._llm,
                keep_recent_turns=self._config.session.keep_recent_turns,
            )
            if self._llm is None:
                logger.info(
                    "session.enabled=True but no LLM configured — "
                    "SessionService available (CRUD), MemoryExtractor/Compressor skipped"
                )
        else:
            self._session_service = None

        # Observability (feature-flagged)
        if self._config.observability.enabled:
            self._trace_collector = TraceCollector(
                pool=self._pool,
                sample_rate=self._config.observability.trace_sample_rate,
            )
            self._metrics = ContextMetrics(self._pool)
            self._debug_service = DebugService(self._pool)
        else:
            self._trace_collector = None
            self._metrics = None
            self._debug_service = None

        # Background Job Runner (per SPEC §9.4)
        self._job_runner = BackgroundJobRunner(pool=self._pool)
        await self._job_runner.start()

        self._initialized = True
        logger.info(f"AgentBase engine initialized: {self._config.db_path}")

    async def close(self) -> None:
        """Shut down the engine and close all connections."""
        if self._job_runner is not None:
            await self._job_runner.stop()
        if self._pool is not None:
            await self._pool.close_all()
        self._initialized = False
        logger.info("AgentBase engine closed")

    async def __aenter__(self) -> AgentBaseEngine:
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Context CRUD — delegates to store/ingester
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        source: str = "manual",
    ) -> ContextEntry:
        """Add a memory context entry."""
        entry = ContextEntry(
            l2_full=content,
            context_type=ContextType.MEMORY,
            memory_category=category,
            tags=tags or [],
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
            source=source,
            origin_type=OriginType.MANUAL,
        )
        return await self.ingester.ingest_direct(entry)

    async def add_conversation(
        self,
        turns: list[dict[str, str]],
        session_date: str | None = None,
        session_index: int = 0,
        tags: list[str] | None = None,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
    ) -> list[ContextEntry]:
        """Add a multi-turn conversation, storing each turn as a separate entry.

        Each turn dict should have keys: "role" (user/assistant) and "content".
        Optional: "date" for per-turn timestamp override.

        Per-turn entries get structured metadata:
        - tags: [original_tags..., "session_{i}", "turn_{j}", role]
        - extra: {"session_index": i, "turn_index": j, "session_date": ..., "role": ...}
        - created_at / valid_from set from session_date for correct temporal ranking
        - category: user→event/preference, assistant→entity

        Returns list of created ContextEntry objects.
        """
        from datetime import datetime, timezone

        entries: list[ContextEntry] = []
        base_tags = tags or []

        # Parse session-level date
        session_dt: datetime | None = None
        if session_date:
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    session_dt = datetime.strptime(session_date.strip(), fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue

        for j, turn in enumerate(turns):
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if not content:
                continue

            # Per-turn date override
            turn_dt = session_dt
            turn_date_str = turn.get("date")
            if turn_date_str:
                for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        turn_dt = datetime.strptime(turn_date_str.strip(), fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue

            # Auto-classify: user turns are events/preferences, assistant turns are entities
            if role == "user":
                # P4: Enhanced preference detection in user turns
                content_lower = content.lower()
                pref_indicators = [
                    # Explicit preference statements
                    "love", "hate", "enjoy", "favorite", "favourite",
                    "prefer", "really into", "big fan", "not a fan",
                    "obsessed", "can't stand", "i like", "i don't like",
                    "always", "never", "usually", "typically",
                    "my go-to", "i tend to", "best", "worst",
                    # Implicit preference indicators — activities/choices/purchases
                    "bought", "purchased", "i own", "i use", "using",
                    "switched to", "tried", "looking for", "need a",
                    "i want", "looking to", "considering", "decided to",
                    "i chose", "i picked", "i went with",
                    # Opinion/review indicators
                    "good", "great", "bad", "terrible", "amazing",
                    "better", "worse", "recommend", "suggest",
                    # Dietary/lifestyle preferences
                    "vegetarian", "vegan", "gluten-free", "kosher", "halal",
                    "allergic", "intolerant", "diet", "workout", "exercise",
                ]
                if any(ind in content_lower for ind in pref_indicators):
                    category = MemoryCategory.PREFERENCE
                else:
                    category = MemoryCategory.EVENT
            else:
                category = MemoryCategory.ENTITY

            # Build structured tags
            turn_tags = list(base_tags) + [
                f"session_{session_index}",
                f"turn_{j}",
                role,
            ]

            # Build content with turn prefix for context
            turn_content = f"[{role}]: {content}"

            entry = ContextEntry(
                l2_full=turn_content,
                context_type=ContextType.MEMORY,
                memory_category=category,
                tags=turn_tags,
                confidence=0.9,
                scope=scope,
                owner_id=owner_id,
                source="conversation",
                origin_type=OriginType.MANUAL,
                extra={
                    "session_index": session_index,
                    "turn_index": j,
                    "session_date": session_date,
                    "role": role,
                },
            )

            # Set temporal fields from session date for correct freshness ranking
            if turn_dt is not None:
                entry.created_at = turn_dt
                entry.valid_from = turn_dt

            result = await self.ingester.ingest_direct(entry)
            entries.append(result)

        # --- Post-ingestion: Entity Graph Extraction ---
        if self._config.graph.extract_on_ingest and self._entity_extractor is not None:
            try:
                await self._extract_entities_from_entries(entries)
            except Exception as e:
                logger.warning(f"Entity extraction during add_conversation failed: {e} — skipping")
        elif self._config.graph.extract_on_ingest:
            logger.info(
                "graph.extract_on_ingest=True but no LLM configured — "
                "entity extraction skipped (graceful degradation)"
            )

        # --- Post-ingestion: Session Management ---
        if self._config.session.extract_on_ingest and self._session_service is not None:
            try:
                session_id, session_memories = await self._process_conversation_session(
                    entries=entries,
                    session_date=session_date,
                    session_index=session_index,
                    tags=tags,
                    scope=scope,
                    owner_id=owner_id,
                )
                # Ingest extracted session memories into the store,
                # then write session_memory_links (FK requires context_id to exist first)
                ingested_ids: list[str] = []
                for mem in session_memories:
                    try:
                        persisted = await self.ingester.ingest_direct(mem)
                        ingested_ids.append(persisted.id)
                    except Exception as e:
                        logger.warning(f"Failed to ingest session memory: {e}")
                # Write session_memory_links now that entries are persisted
                if ingested_ids and session_id is not None:
                    try:
                        await self._session_service.link_memories(
                            session_id=session_id,
                            context_ids=ingested_ids,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write session_memory_links: {e}")
            except Exception as e:
                logger.warning(f"Session processing during add_conversation failed: {e} — skipping")
        elif self._config.session.extract_on_ingest:
            logger.info(
                "session.extract_on_ingest=True but no LLM configured — "
                "session processing skipped (graceful degradation)"
            )

        return entries

    async def _extract_entities_from_entries(self, entries):
        """Extract entities and relations from ingested entries and persist to graph.

        Graceful degradation: if extraction fails for an entry, skip and continue.
        """
        if self._entity_extractor is None or self._entity_service is None:
            return

        for entry in entries:
            if not entry.l2_full:
                continue
            try:
                entities, raw_relations = await self._entity_extractor.extract(entry.l2_full)
            except Exception as e:
                logger.warning(f"Entity extraction failed for entry {entry.id}: {e}")
                continue

            entity_name_to_id = {}
            for entity in entities:
                try:
                    persisted = await self._entity_service.add_entity(entity)
                    entity_name_to_id[entity.name] = persisted.id
                except Exception as e:
                    logger.warning(f"Failed to persist entity '{entity.name}': {e}")

            for rel in raw_relations:
                source_id = entity_name_to_id.get(rel["source"])
                target_id = entity_name_to_id.get(rel["target"])
                if not source_id or not target_id:
                    continue
                try:
                    from .models.entity import Relation
                    relation = Relation(
                        source_id=source_id,
                        target_id=target_id,
                        predicate=rel.get("predicate", "related_to"),
                        confidence=rel.get("confidence", 0.8),
                    )
                    await self._entity_service.add_relation(relation)
                except Exception as e:
                    logger.warning(
                        f"Failed to persist relation {rel['source']}->{rel['target']}: {e}"
                    )

            # Conflict resolution is per-fact; skip in batch extraction

    async def _process_conversation_session(
        self,
        entries,
        session_date,
        session_index,
        tags,
        scope,
        owner_id,
    ):
        """Create a session from a conversation, add messages, and commit.

        Returns (session_id, extracted_memories) tuple.
        Graceful degradation: returns (None, []) on failure.
        """
        if self._session_service is None:
            return None, []

        agent_id = owner_id or "default"

        try:
            session = await self._session_service.create_session(
                agent_id=agent_id,
                project=(tags[0] if tags else None),
            )
        except Exception as e:
            logger.warning(f"Failed to create session: {e}")
            return None, []

        for entry in entries:
            role = entry.extra.get("role", "unknown") if entry.extra else "unknown"
            content = entry.l2_full
            try:
                await self._session_service.add_message(
                    session_id=session.id,
                    role=role,
                    content=content,
                )
            except Exception as e:
                logger.warning(f"Failed to add message to session {session.id}: {e}")

        try:
            extracted = await self._session_service.commit_session(
                session_id=session.id,
                mode="full",
            )
            return session.id, extracted
        except Exception as e:
            logger.warning(f"Failed to commit session {session.id}: {e}")
            return None, []

    async def add_resource(
        self,
        url: str | None = None,
        content: str = "",
        format: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        source: str = "manual",
        reason: str = "",
    ) -> ContextEntry:
        """Add a resource context entry."""
        entry = ContextEntry(
            l2_full=content or reason or url or "",
            context_type=ContextType.RESOURCE,
            resource_url=url,
            resource_format=format,
            tags=tags or [],
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
            source=source,
            origin_type=OriginType.MANUAL,
        )
        return await self.ingester.ingest_direct(entry)

    async def add_skill(
        self,
        tool_name: str,
        description: str = "",
        api_spec: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        source: str = "manual",
    ) -> ContextEntry:
        """Add a skill context entry."""
        entry = ContextEntry(
            l2_full=description,
            context_type=ContextType.SKILL,
            skill_tool_name=tool_name,
            skill_api_spec=api_spec,
            tags=tags or [],
            confidence=confidence,
            scope=scope,
            owner_id=owner_id,
            source=source,
            origin_type=OriginType.MANUAL,
        )
        return await self.ingester.ingest_direct(entry)

    async def add(self, entry: ContextEntry) -> ContextEntry:
        """Add a pre-constructed ContextEntry."""
        return await self.ingester.ingest_direct(entry)

    async def get(self, entry_id: str, load_level: str = "l2") -> ContextEntry | None:
        """Get a context entry by ID."""
        entry = await self.store.get(entry_id)
        if entry is None:
            return None
        return entry

    async def find(
        self,
        query: str,
        top_k: int | None = None,
        context_type: ContextType | None = None,
        scope: ContextScope | None = None,
        owner_id: str | None = None,
        token_budget: int | None = None,
        include_trace: bool = False,
        strategy: str = "hybrid",
        query_type: str | None = None,
    ) -> list[SearchResult]:
        """Search for context entries (convenience method).

        When *query_type* is None (default), auto-detects the query type
        via :meth:`IntentAnalyzer.detect_query_type` using keyword rules.
        Explicitly pass a query_type to override auto-detection.
        """
        if top_k is None:
            top_k = self._config.retrieval.default_top_k
        if token_budget is None:
            token_budget = self._config.retrieval.default_token_budget

        # Auto-detect query type when not explicitly provided
        if query_type is None:
            query_type = IntentAnalyzer.detect_query_type(query)

        search_query = SearchQuery(
            text=query,
            top_k=top_k,
            strategy=strategy,
            context_type=context_type,
            scope=scope,
            owner_id=owner_id,
            token_budget=token_budget,
            include_trace=include_trace,
            query_type=query_type,
        )
        return await self.search(search_query)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for context entries using a SearchQuery with full retrieval pipeline."""
        if self._retrieval_engine is None:
            raise StorageError("Engine not initialized. Call initialize() first.")

        results = await self._retrieval_engine.search(query)

        # Apply confidence filter
        if query.min_confidence is not None:
            results = [r for r in results if r.entry.confidence >= query.min_confidence]

        # Apply tag filter
        if query.tags:
            tag_set = set(query.tags)
            results = [r for r in results if tag_set.issubset(set(r.entry.tags))]

        return results

    async def update(self, entry: ContextEntry) -> None:
        """Update a context entry."""
        await self.store.update(entry)

    async def delete(self, entry_id: str) -> bool:
        """Soft delete a context entry."""
        return await self.store.delete(entry_id)

    async def purge(self, entry_id: str) -> bool:
        """Hard delete a context entry."""
        return await self.store.purge(entry_id)

    async def list_entries(
        self,
        scope: ContextScope | None = None,
        context_type: ContextType | None = None,
        status: EntryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContextEntry]:
        """List context entries with filters."""
        return await self.store.list_entries(
            scope=scope,
            context_type=context_type,
            status=status,
            limit=limit,
            offset=offset,
        )

    async def count(
        self,
        scope: ContextScope | None = None,
        context_type: ContextType | None = None,
    ) -> int:
        """Count context entries."""
        return await self.store.count(scope=scope, context_type=context_type)

    async def ingest_text(
        self,
        text: str,
        context_type: ContextType = ContextType.MEMORY,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ContextEntry]:
        """Ingest raw text via LLM extraction."""
        return await self.ingester.ingest_text(
            text=text,
            context_type=context_type,
            scope=scope,
            owner_id=owner_id,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Entity / Graph operations
    # ------------------------------------------------------------------

    @property
    def entity_service(self) -> EntityService:
        if not self._initialized:
            raise StorageError("Engine not initialized. Call initialize() first.")
        if self._entity_service is None:
            raise ConfigError("Graph feature is disabled. Set config.graph.enabled=True to enable.")
        return self._entity_service

    async def add_entity(self, entity: Entity) -> Entity:
        """Add or merge an entity."""
        return await self.entity_service.add_entity(entity)

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return await self.entity_service.get_entity(entity_id)

    async def find_entities(self, name: str, entity_type: str | None = None) -> list[Entity]:
        """Find entities by name."""
        return await self.entity_service.find_entities(name, entity_type=entity_type)

    async def add_alias(self, entity_id: str, alias: str) -> None:
        """Add an alias for entity disambiguation."""
        await self.entity_service.add_alias(entity_id, alias)

    async def add_relation(self, relation: Relation) -> Relation:
        """Add a relation between entities."""
        return await self.entity_service.add_relation(relation)

    async def get_current_relations(self, entity_id: str) -> list[Relation]:
        """Get current (valid) relations for an entity."""
        return await self.entity_service.get_current_relations(entity_id)

    async def graph_traversal(self, entity_name: str, depth: int = 2) -> list[dict]:
        """Traverse the knowledge graph from an entity."""
        return await self.entity_service.graph_traversal(entity_name, depth=depth)

    async def add_fact(self, fact: FactTimeline) -> None:
        """Add a fact to the timeline."""
        await self.entity_service.add_fact(fact)

    async def get_current_facts(self, entity_id: str) -> list[FactTimeline]:
        """Get current (non-superseded) facts for an entity."""
        return await self.entity_service.get_current_facts(entity_id)

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    @property
    def session_service(self) -> SessionService:
        if not self._initialized:
            raise StorageError("Engine not initialized. Call initialize() first.")
        if self._session_service is None:
            raise ConfigError("Session feature is disabled. Set config.session.enabled=True to enable.")
        return self._session_service

    async def create_session(
        self,
        agent_id: str = "default",
        project: str | None = None,
    ) -> Session:
        """Create a new conversation session."""
        return await self.session_service.create_session(agent_id=agent_id, project=project)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> SessionMessage:
        """Add a message to a session."""
        return await self.session_service.add_message(
            session_id=session_id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    async def get_session(self, session_id: str, load_messages: bool = False) -> Session | None:
        """Get a session by ID."""
        return await self.session_service.get_session(session_id, load_messages=load_messages)

    async def commit_session(
        self,
        session_id: str,
        mode: str = "full",
    ) -> list[ContextEntry]:
        """Commit a session: compress, archive, and extract memories."""
        return await self.session_service.commit_session(session_id, mode=mode)

    # ------------------------------------------------------------------
    # Observability operations
    # ------------------------------------------------------------------

    @property
    def trace_collector(self) -> TraceCollector:
        if not self._initialized:
            raise StorageError("Engine not initialized. Call initialize() first.")
        if self._trace_collector is None:
            raise ConfigError("Observability feature is disabled. Set config.observability.enabled=True to enable.")
        return self._trace_collector

    @property
    def metrics(self) -> ContextMetrics:
        if not self._initialized:
            raise StorageError("Engine not initialized. Call initialize() first.")
        if self._metrics is None:
            raise ConfigError("Observability feature is disabled. Set config.observability.enabled=True to enable.")
        return self._metrics

    @property
    def debug(self) -> DebugService:
        if not self._initialized:
            raise StorageError("Engine not initialized. Call initialize() first.")
        if self._debug_service is None:
            raise ConfigError("Observability feature is disabled. Set config.observability.enabled=True to enable.")
        return self._debug_service

    async def get_metrics(self) -> dict[str, Any]:
        """Get context quality metrics.

        Basic metrics (entry counts, job stats) are always available.
        Extended metrics (query latency, p50) require observability to be enabled.
        """
        if self._metrics is not None:
            return await self._metrics.get_metrics()

        # Fallback: compute basic stats directly without observability
        from .observability.observability_service import ContextMetrics
        basic_metrics = ContextMetrics(self._pool)
        return await basic_metrics.get_metrics()

    async def explain_query(self, query: str) -> dict:
        """Explain what a retrieval query would do (dry run)."""
        return await self.debug.explain_query(query)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_budget(results: list[SearchResult], budget: int) -> list[SearchResult]:
        """Trim results to fit within a token budget (rough estimation: 1 char ≈ 0.5 token for Chinese)."""
        used = 0
        trimmed = []
        for r in results:
            content = r.entry.l2_full or r.entry.l1_overview or r.entry.l0_abstract
            estimated_tokens = len(content) // 2  # rough estimate
            if used + estimated_tokens <= budget:
                used += estimated_tokens
                trimmed.append(r)
            else:
                break
        return trimmed

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------

    async def reindex(self) -> dict[str, int]:
        """Rebuild all indexes from scratch.

        Returns counts of reindexed entries.
        """
        result: dict[str, int] = {}

        if self._fts_index is not None:
            count = await self._fts_index.rebuild()
            result["fts"] = count

        # TODO: vector index rebuild when sqlite-vec is available

        logger.info(f"Index rebuild complete: {result}")
        return result

    async def cleanup(
        self,
        traces_older_than_days: int | None = None,
        deleted_older_than_days: int | None = None,
        failed_jobs_older_than_days: int | None = None,
    ) -> dict[str, int]:
        """Clean up old data based on retention policies.

        Returns counts of deleted records per category.
        """
        result: dict[str, int] = {}

        if self._pool is None:
            raise StorageError("Engine not initialized.")

        async with self._pool.get_write_conn() as conn:
            # Clean old traces
            if traces_older_than_days is not None:
                cursor = await conn.execute(
                    "DELETE FROM retrieval_traces WHERE created_at < datetime('now', ? || ' days')",
                    (f"-{traces_older_than_days}",),
                )
                result["traces"] = cursor.rowcount

                # Also clean trace_steps
                await conn.execute(
                    "DELETE FROM trace_steps WHERE trace_id NOT IN (SELECT id FROM retrieval_traces)",
                )

            # Purge soft-deleted entries
            if deleted_older_than_days is not None:
                cursor = await conn.execute(
                    "DELETE FROM context_entries WHERE status = 'deleted' AND deleted_at < datetime('now', ? || ' days')",
                    (f"-{deleted_older_than_days}",),
                )
                result["deleted_entries"] = cursor.rowcount

            # Clean failed jobs
            if failed_jobs_older_than_days is not None:
                cursor = await conn.execute(
                    "DELETE FROM background_jobs WHERE status = 'failed' AND completed_at < datetime('now', ? || ' days')",
                    (f"-{failed_jobs_older_than_days}",),
                )
                result["failed_jobs"] = cursor.rowcount

            await conn.commit()

        if result:
            logger.info(f"Cleanup complete: {result}")
        return result

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space."""
        if self._pool is None:
            raise StorageError("Engine not initialized.")
        async with self._pool.get_write_conn() as conn:
            await conn.execute("VACUUM")
            await conn.commit()
        logger.info("VACUUM complete")
