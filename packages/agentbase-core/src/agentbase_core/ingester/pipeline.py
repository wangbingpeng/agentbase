"""Ingester — context entry ingestion pipeline."""

from __future__ import annotations

import logging

from ..embedding.base import AbstractEmbedder
from ..exceptions import StorageError, ValidationError
from ..index.hybrid import HybridIndex
from ..ingester.dedup import DedupDecision, Deduplicator, EmbeddingCache
from ..ingester.layer_gen import LayerGenerator
from ..ingester.local_fact import LocalFactExtractor
from ..ingester.ner_extractor import NerExtractor
from ..llm.base import AbstractLLM
from ..models.config import IngestConfig, TierConfig
from ..models.context_entry import (
    ContextEntry,
    ContextScope,
    ContextType,
    EntryStatus,
    MemoryCategory,
    OriginType,
)
from ..store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class Ingester:
    """Context entry ingestion pipeline.

    Flow:
    1. Validate + type route
    2. (Optional) LLM extract facts from text
    3. Dedup check
    4. Write to SQLite (transaction) + FTS auto-sync
    5. (Optional) Generate L0/L1 layers
    6. (Optional) Generate embedding
    7. Submit background jobs for async completion
    """

    def __init__(
        self,
        store: SQLiteStore,
        index: HybridIndex,
        llm: AbstractLLM | None = None,
        embedder: AbstractEmbedder | None = None,
        tier_config: TierConfig | None = None,
        ingest_config: IngestConfig | None = None,
        dedup_threshold: float = 0.92,
        tokenizer: str = "auto",
    ) -> None:
        self._store = store
        self._index = index
        self._llm = llm
        self._embedder = embedder
        self._tier_config = tier_config or TierConfig()
        self._ingest_config = ingest_config or IngestConfig()
        self._dedup = Deduplicator(store, index, threshold=dedup_threshold)
        self._layer_gen = LayerGenerator(llm=llm)
        self._local_fact_extractor = LocalFactExtractor()
        self._ner_extractor = NerExtractor()
        self._tokenizer = tokenizer

    async def ingest_direct(self, entry: ContextEntry) -> ContextEntry:
        """Ingest a pre-constructed ContextEntry directly.

        Steps:
        1. Validate scope/owner_id constraints
        2. Dedup check
        3. Generate L0/L1 (sync or async)
        4. Write to store
        5. Generate embedding (if embedder available)
        6. Update entry in store with generated fields
        """
        # Validate type constraints per SPEC §4.6
        entry.validate_type_constraints()

        # Dedup check
        dedup_result = await self._dedup.check(
            content=entry.l2_full,
            context_type=entry.context_type,
            scope=entry.scope,
            owner_id=entry.owner_id,
            memory_category=entry.memory_category,
        )

        if dedup_result.is_duplicate:
            logger.info(f"Skipping duplicate entry: {dedup_result.similar_entry_id}")
            # Return the existing entry
            existing = await self._store.get(dedup_result.similar_entry_id)
            if existing:
                return existing

        if dedup_result.should_supersede:
            # Generate layers for new entry, then supersede old one
            entry = await self._layer_gen.generate_layers(entry)
            entry.mark_active()
            await self._store.supersede(dedup_result.similar_entry_id, entry)
            return entry

        # Generate L0/L1 layers
        if self._tier_config.enabled and entry.l2_full:
            if self._tier_config.async_generation:
                # Apply truncation fallback immediately, async LLM gen later
                if self._tier_config.fallback_to_truncation:
                    entry.apply_truncation_fallback()
            else:
                entry = await self._layer_gen.generate_layers(entry)

        # Mark as active
        entry.mark_active()

        # Populate fts_text with tokenized content for CJK-aware FTS5 search
        if not entry.fts_text and entry.l2_full:
            from ..index.tokenizer import tokenize_text
            entry.fts_text = tokenize_text(entry.l2_full, tokenizer=self._tokenizer)

        # Write to store
        entry = await self._store.add(entry)

        # --- Ingest-time extraction: session summary + facts + NER ---
        await self._extract_and_ingest(entry)

        # Generate embedding asynchronously
        if self._embedder is not None and entry.l2_full:
            try:
                embedding_input = entry.get_embedding_input()
                embedding = await self._embedder.embed(embedding_input)
                entry.embedding_hash = EmbeddingCache.compute_hash(embedding_input)
                entry.embedding_model = self._embedder.model_name
                entry.embedding_dimensions = self._embedder.dimensions

                # Cache the embedding vector for later vector index insertion
                emb_cache = EmbeddingCache(self._store._pool)
                await emb_cache.put(
                    content_hash=entry.embedding_hash,
                    embedding=embedding,
                    model=self._embedder.model_name,
                    dimensions=self._embedder.dimensions,
                )

                await self._store.update(entry)
                # Add to vector index
                await self._index.add(entry)
            except Exception as e:
                logger.warning(f"Embedding generation failed for {entry.id}: {e}")

        return entry

    async def ingest_text(
        self,
        text: str,
        context_type: ContextType = ContextType.MEMORY,
        scope: ContextScope = ContextScope.GLOBAL,
        owner_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "text_input",
        confidence: float = 1.0,
        memory_category: MemoryCategory | None = None,
    ) -> list[ContextEntry]:
        """Ingest raw text content via LLM extraction.

        The LLM extracts structured facts from the text, each becoming
        a separate ContextEntry.
        """
        if self._llm is None:
            # Without LLM, just ingest the text as a single entry
            entry = ContextEntry(
                l2_full=text,
                context_type=context_type,
                scope=scope,
                owner_id=owner_id,
                tags=tags or [],
                source=source,
                confidence=confidence,
                memory_category=memory_category,
                origin_type=OriginType.MANUAL,
            )
            result = await self.ingest_direct(entry)
            return [result]

        # Use LLM to extract facts
        from ..llm.base import AbstractLLM

        extract_prompt = f"""从以下文本中提取结构化记忆。

Categories:
- profile: 人物画像（身份、角色、背景）
- preference: 偏好（喜欢/不喜欢的工具、风格、方式）
- entity: 实体（项目、产品、技术、组织）
- event: 事件（发生了什么、何时发生）
- case: 情景案例（问题-解决对、成功/失败经验）
- pattern: 行为模式（重复出现的模式、习惯）

Rules:
- 每条记忆必须简洁、事实化
- 每条记忆必须有 category 和 tags
- 最多提取 10 条
- 返回 JSON 数组

Text:
{text}

Return JSON array:
[
  {{"category": "preference", "content": "...", "tags": ["..."], "confidence": 0.9}}
]"""

        try:
            extracted = await self._llm.complete_json(
                prompt=extract_prompt,
                system="你是一个结构化信息提取器。从文本中提取关键事实和信息。",
            )
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to direct ingest: {e}")
            entry = ContextEntry(
                l2_full=text,
                context_type=context_type,
                scope=scope,
                owner_id=owner_id,
                tags=tags or [],
                source=source,
                confidence=confidence,
                origin_type=OriginType.MANUAL,
            )
            result = await self.ingest_direct(entry)
            return [result]

        if not isinstance(extracted, list):
            extracted = [extracted]

        entries = []
        for item in extracted[:10]:  # Max 10 items
            if not isinstance(item, dict):
                continue

            cat_str = item.get("category", "entity")
            try:
                category = MemoryCategory(cat_str)
            except ValueError:
                category = MemoryCategory.ENTITY

            entry = ContextEntry(
                l2_full=item.get("content", ""),
                context_type=context_type,
                memory_category=category,
                scope=scope,
                owner_id=owner_id,
                tags=item.get("tags", []),
                confidence=item.get("confidence", 0.8),
                source=source,
                origin_type=OriginType.EXTRACTED,
            )
            result = await self.ingest_direct(entry)
            entries.append(result)

        return entries

    # ------------------------------------------------------------------
    # Ingest-time extraction helpers
    # ------------------------------------------------------------------

    async def _extract_and_ingest(self, parent_entry: ContextEntry) -> None:
        """Run session-summary, fact, and NER extraction after ingest.

        Generated entries are written as children of *parent_entry* with
        appropriate tags and origin_type.

        IMPORTANT: Extracted entries (summaries, facts) are NOT re-extracted
        to prevent infinite recursion.
        """
        cfg = self._ingest_config
        if not cfg.session_summary and not cfg.fact_extraction and not cfg.ner_extraction:
            return

        # Prevent infinite recursion: skip entries that are already extracted/summary
        parent_extra = parent_entry.extra or {}
        if "summary_of" in parent_extra or "fact_of" in parent_extra:
            return
        if parent_entry.origin_type == OriginType.EXTRACTED:
            return

        content = parent_entry.l2_full or ""
        if not content.strip():
            return

        parent_tags = parent_entry.tags or []
        is_session_entry = any(t.startswith("session_") for t in parent_tags)

        # --- Session Summary ---
        if cfg.session_summary and is_session_entry:
            summary_result = await self._generate_session_summary(content, parent_tags, parent_extra)
            if summary_result:
                summary, structured = summary_result
                summary_tags = list(parent_tags) + ["session_summary"]
                # Merge structured fields (time_range / topics / entities)
                # into extra so the retrieval layer can filter / rank by them.
                summary_extra = {
                    **parent_extra,
                    "summary_of": parent_entry.id,
                    **(structured or {}),
                }
                summary_entry = ContextEntry(
                    l2_full=summary,
                    context_type=ContextType.MEMORY,
                    scope=parent_entry.scope,
                    owner_id=parent_entry.owner_id,
                    tags=summary_tags,
                    extra=summary_extra,
                    origin_type=OriginType.EXTRACTED,
                    source="session_summary",
                )
                # Propagate session date to created_at for correct temporal ranking
                session_date = parent_extra.get("session_date") if parent_extra else None
                if session_date:
                    from datetime import datetime, timezone
                    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                        try:
                            dt = datetime.strptime(session_date.strip(), fmt).replace(tzinfo=timezone.utc)
                            summary_entry.created_at = dt
                            summary_entry.valid_from = dt
                            break
                        except ValueError:
                            continue
                try:
                    await self.ingest_direct(summary_entry)
                except Exception as e:
                    logger.warning(f"Failed to ingest session summary: {e}")

        # --- Fact Extraction ---
        if cfg.fact_extraction:
            facts = await self._extract_facts(content, parent_extra)
            for fact in facts:
                cat_str = fact.get("category", "entity")
                try:
                    category = MemoryCategory(cat_str)
                except ValueError:
                    category = MemoryCategory.ENTITY
                fact_tags = list(parent_tags) + ["fact"] + fact.get("tags", [])
                fact_entry = ContextEntry(
                    l2_full=fact.get("content", ""),
                    context_type=ContextType.MEMORY,
                    memory_category=category,
                    scope=parent_entry.scope,
                    owner_id=parent_entry.owner_id,
                    tags=fact_tags,
                    extra={**parent_extra, "fact_of": parent_entry.id},
                    origin_type=OriginType.EXTRACTED,
                    source="fact_extraction",
                    confidence=fact.get("confidence", 0.8),
                )
                try:
                    await self.ingest_direct(fact_entry)
                except Exception as e:
                    logger.debug(f"Failed to ingest fact entry: {e}")

        # --- NER Extraction → Entity Tagging ---
        # Instead of creating separate NER entries (which dilute search results),
        # we add entity names as tags on the parent entry.
        # FTS5 already indexes tags, so entities become searchable via tag match.
        # The RetrievalEngine's NER boost also matches these tags.
        if cfg.ner_extraction:
            entities = self._ner_extractor.extract(content)
            if entities:
                entity_tags = ["ner"]
                for ent in entities:
                    ent_text = ent.get("content", "").strip()
                    if ent_text:
                        # Sanitize entity text for use as a tag:
                        # replace spaces with underscores, keep alphanumeric + CJK
                        tag_name = "ner_" + "_".join(
                            ch if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" else "_"
                            for ch in ent_text
                        ).strip("_")
                        # Avoid overly long tags (> 60 chars)
                        if len(tag_name) <= 60:
                            entity_tags.append(tag_name)
                if len(entity_tags) > 1:  # more than just "ner"
                    # Update parent entry tags in-place
                    existing_tags = set(parent_entry.tags or [])
                    new_tags = [t for t in entity_tags if t not in existing_tags]
                    if new_tags:
                        parent_entry.tags = list(parent_entry.tags or []) + new_tags
                        # Update fts_text to include new tags for FTS5 searchability
                        from ..index.tokenizer import tokenize_text
                        tag_text = " ".join(new_tags)
                        parent_entry.fts_text = (
                            (parent_entry.fts_text or "") + " " +
                            tokenize_text(tag_text, tokenizer=self._tokenizer)
                        ).strip()
                        try:
                            await self._store.update(parent_entry)
                        except Exception as e:
                            logger.debug(f"Failed to update NER tags on entry: {e}")

    async def _generate_session_summary(
        self, content: str, tags: list[str], extra: dict
    ) -> tuple[str, dict] | None:
        """Generate a session summary plus structured metadata.

        Returns ``(summary_text, structured_data)`` where ``structured_data``
        may contain:

        - ``time_range``: ``{"start": iso, "end": iso}`` derived from session_date
        - ``topics``: list of topic keywords (from NER)
        - ``entities``: list of entity names (from NER)
        - ``summary_source``: "llm" or "truncation"

        - LLM available → high-quality summary text
        - LLM unavailable → truncation fallback (first 300 chars)
        """
        if not content.strip():
            return None

        # --- 1. Generate summary text ---
        summary: str
        summary_source: str
        if self._llm is not None:
            try:
                prompt = (
                    "Summarize the following conversation in 2-3 concise sentences, "
                    "focusing on key facts, preferences, and events mentioned by the user:\n\n"
                    f"{content[:3000]}"
                )
                result = await self._llm.complete(prompt=prompt)
                summary = result.strip()[:500]
                summary_source = "llm"
                if not summary:
                    summary = content[:300] + "..." if len(content) > 300 else content
                    summary_source = "truncation"
            except Exception as e:
                logger.warning(f"LLM session summary failed, using truncation: {e}")
                summary = content[:300] + "..." if len(content) > 300 else content
                summary_source = "truncation"
        else:
            summary = content[:300] + "..." if len(content) > 300 else content
            summary_source = "truncation"

        # --- 2. Structured metadata ---
        structured: dict = {"summary_source": summary_source}

        # 2a. Time range from session_date (if any)
        session_date = extra.get("session_date") if extra else None
        if session_date:
            from datetime import datetime, timezone
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(session_date.strip(), fmt).replace(tzinfo=timezone.utc)
                    structured["time_range"] = {
                        "start": dt.isoformat(),
                        "end": dt.isoformat(),
                    }
                    break
                except ValueError:
                    continue

        # 2b. Topics / entities from NER extractor
        try:
            entities = self._ner_extractor.extract(content)
        except Exception:
            entities = []
        if entities:
            entity_names: list[str] = []
            topic_names: list[str] = []
            for ent in entities:
                name = (ent.get("content") or "").strip()
                if not name:
                    continue
                cat = (ent.get("category") or "").lower()
                if cat in {"person", "organization", "location", "product", "place"}:
                    if name not in entity_names:
                        entity_names.append(name)
                else:
                    if name not in topic_names:
                        topic_names.append(name)
            if entity_names:
                structured["entities"] = entity_names[:20]
            if topic_names:
                structured["topics"] = topic_names[:20]

        return summary, structured

    async def _extract_facts(self, content: str, extra: dict) -> list[dict]:
        """Extract facts from content.

        - LLM available → MemoryExtractor (high-quality)
        - LLM unavailable → LocalFactExtractor (regex fallback)
        """
        if self._llm is not None:
            try:
                from ..session.session_service import MemoryExtractor
                extractor = MemoryExtractor(self._llm)
                # MemoryExtractor expects a Session; create a minimal one
                from ..models.session import Session, SessionMessage
                session = Session(id="_extract", messages=[
                    SessionMessage(role="user", content=content[:3000])
                ])
                return await extractor.extract(session)
            except Exception as e:
                logger.warning(f"LLM fact extraction failed, using local regex: {e}")

        # Fallback: local regex
        return self._local_fact_extractor.extract(content)
