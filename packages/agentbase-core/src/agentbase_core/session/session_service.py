"""SessionService — conversation session lifecycle management."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ..llm.base import AbstractLLM
from ..models.context_entry import ContextEntry, ContextScope, ContextType, MemoryCategory, OriginType
from ..models.entity import _new_ulid, _utcnow
from ..models.session import Session, SessionMessage
from ..store.connection import ConnectionPool

logger = logging.getLogger(__name__)


class SessionCompressor:
    """Compress session messages — keep recent turns, archive older ones."""

    def __init__(self, llm: AbstractLLM | None = None, keep_recent_turns: int = 6) -> None:
        self._llm = llm
        self._keep_recent_turns = keep_recent_turns

    async def compress(self, session: Session) -> Session:
        """Compress session: keep recent turns, archive older ones."""
        turns = self._split_into_turns(session.messages)
        if len(turns) <= self._keep_recent_turns:
            return session

        to_archive = turns[: -self._keep_recent_turns]
        to_keep = turns[-self._keep_recent_turns :]

        archive_text = self._format_turns(to_archive)

        # Generate L0/L1 summaries for archived portion
        if self._llm:
            try:
                session.archived_summary_l0 = await self._generate_summary(archive_text, max_len=50)
                session.archived_summary_l1 = await self._generate_summary(archive_text, max_len=300)
            except Exception as e:
                logger.warning(f"Session summary generation failed: {e}")
                session.archived_summary_l0 = archive_text[:100] + "..."
                session.archived_summary_l1 = archive_text[:500] + "..."
        else:
            session.archived_summary_l0 = archive_text[:100] + "..."
            session.archived_summary_l1 = archive_text[:500] + "..."

        session.archived_message_count = sum(len(t) for t in to_archive)
        session.messages = [m for t in to_keep for m in t]
        return session

    async def _generate_summary(self, text: str, max_len: int = 300) -> str:
        """Generate a summary of archived conversation."""
        if not self._llm:
            return text[:max_len] + "..."
        prompt = f"请用不超过{max_len}字概括以下对话内容：\n\n{text}"
        result = await self._llm.complete(prompt=prompt)
        return result.strip()[:max_len]

    @staticmethod
    def _split_into_turns(messages: list[SessionMessage]) -> list[list[SessionMessage]]:
        """Split messages into turn units (each turn starts with a user message)."""
        turns: list[list[SessionMessage]] = []
        current_turn: list[SessionMessage] = []
        for msg in messages:
            if msg.role == "user" and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)
        return turns

    @staticmethod
    def _format_turns(turns: list[list[SessionMessage]]) -> str:
        """Format turns into readable text."""
        parts = []
        for turn in turns:
            for msg in turn:
                parts.append(f"[{msg.role}]: {msg.content}")
        return "\n".join(parts)


class MemoryExtractor:
    """Extract structured memories from session conversations."""

    _EXTRACT_PROMPT = """从以下对话中提取结构化记忆。

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

Conversation:
{conversation}

Return JSON array:
[
  {{"category": "preference", "content": "...", "tags": ["..."], "confidence": 0.9}}
]"""

    def __init__(self, llm: AbstractLLM) -> None:
        self._llm = llm

    async def extract(self, session: Session) -> list[dict]:
        """Extract memories from a session."""
        conversation = "\n".join(f"[{m.role}]: {m.content}" for m in session.messages)
        if not conversation.strip():
            return []

        try:
            result = await self._llm.complete_json(
                prompt=self._EXTRACT_PROMPT.format(conversation=conversation),
                system="你是一个结构化记忆提取器。",
            )
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")
            return []

        if not isinstance(result, list):
            result = [result]

        memories = []
        for item in result[:10]:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            cat = item.get("category", "entity")
            if cat not in ("profile", "preference", "entity", "event", "case", "pattern"):
                cat = "entity"
            memories.append({
                "category": cat,
                "content": item["content"],
                "tags": item.get("tags", []),
                "confidence": item.get("confidence", 0.8),
            })

        return memories


class SessionService:
    """Session lifecycle management — create, add messages, commit, archive."""

    def __init__(
        self,
        pool: ConnectionPool,
        llm: AbstractLLM | None = None,
        keep_recent_turns: int = 6,
    ) -> None:
        self._pool = pool
        self._compressor = SessionCompressor(llm=llm, keep_recent_turns=keep_recent_turns)
        self._memory_extractor = MemoryExtractor(llm) if llm else None

    async def create_session(
        self,
        agent_id: str = "default",
        project: str | None = None,
    ) -> Session:
        """Create a new session."""
        session = Session(agent_id=agent_id, project=project)
        now = _utcnow().isoformat()

        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, agent_id, project, status, created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?)",
                (session.id, agent_id, project, now, now),
            )
            await conn.commit()

        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> SessionMessage:
        """Add a message to a session (immediately persisted)."""
        msg = SessionMessage(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        now = _utcnow().isoformat()

        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "INSERT INTO session_messages (id, session_id, role, content, tool_call_id, tool_name, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (msg.id, session_id, role, content, tool_call_id, tool_name, now),
            )
            await conn.commit()

        return msg

    async def get_session(self, session_id: str, load_messages: bool = False) -> Session | None:
        """Get a session by ID."""
        async with self._pool.get_read_conn() as conn:
            cursor = await conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            if row is None:
                return None

        session = self._row_to_session(row)

        if load_messages:
            async with self._pool.get_read_conn() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM session_messages WHERE session_id = ? ORDER BY created_at",
                    (session_id,),
                )
                rows = await cursor.fetchall()
                session.messages = [self._row_to_message(r) for r in rows]

        return session

    async def commit_session(
        self,
        session_id: str,
        mode: str = "full",
    ) -> list[ContextEntry]:
        """Commit a session: compress, archive, and extract memories.

        mode: 'full' (compress + extract), 'archive_only', 'extract_only'
        """
        session = await self.get_session(session_id, load_messages=True)
        if session is None:
            return []

        extracted_memories: list[ContextEntry] = []

        if mode in ("full", "archive_only"):
            session = await self._compressor.compress(session)
            await self._persist_session_update(session)

        if mode in ("full", "extract_only") and self._memory_extractor:
            memories = await self._memory_extractor.extract(session)
            for mem in memories:
                confidence = mem.get("confidence", 0.8)
                if confidence < 0.5:
                    continue  # Skip low-confidence
                try:
                    category = MemoryCategory(mem["category"])
                except ValueError:
                    category = MemoryCategory.ENTITY

                entry = ContextEntry(
                    l2_full=mem["content"],
                    context_type=ContextType.MEMORY,
                    memory_category=category,
                    tags=mem.get("tags", []),
                    confidence=confidence,
                    scope=ContextScope.AGENT,
                    owner_id=session.agent_id,
                    origin_type=OriginType.SESSION,
                    origin_id=session_id,
                    source="session_extract",
                )
                extracted_memories.append(entry)

        # Mark session as committed
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE sessions SET status = 'archived', committed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, session_id),
            )
            await conn.commit()

        # NOTE: session_memory_links are written after the caller persists
        # extracted_memories into context_entries (via link_memories()).
        # Writing them here would violate the FK constraint because the
        # context_id doesn't exist in context_entries yet.

        return extracted_memories

    async def link_memories(self, session_id: str, context_ids: list[str]) -> None:
        """Write session_memory_links after extracted memories are persisted.

        Must be called AFTER the ContextEntry objects have been ingested into
        context_entries via ingest_direct(), otherwise the FK constraint on
        context_id will fail.
        """
        if not context_ids:
            return
        async with self._pool.get_write_conn() as conn:
            for cid in context_ids:
                await conn.execute(
                    "INSERT OR IGNORE INTO session_memory_links (session_id, context_id) VALUES (?, ?)",
                    (session_id, cid),
                )
            await conn.commit()

    async def _persist_session_update(self, session: Session) -> None:
        """Persist session metadata updates."""
        now = _utcnow().isoformat()
        async with self._pool.get_write_conn() as conn:
            await conn.execute(
                "UPDATE sessions SET archived_summary_l0 = ?, archived_summary_l1 = ?, archived_message_count = ?, total_tokens_used = ?, updated_at = ? WHERE id = ?",
                (
                    session.archived_summary_l0,
                    session.archived_summary_l1,
                    session.archived_message_count,
                    session.total_tokens_used,
                    now,
                    session.id,
                ),
            )
            await conn.commit()

    @staticmethod
    def _row_to_session(row: tuple) -> Session:
        return Session(
            id=row[0], agent_id=row[1], project=row[2], status=row[3],
            archived_summary_l0=row[4] or "",
            archived_summary_l1=row[5] or "",
            archived_message_count=row[6] or 0,
            total_tokens_used=row[7] or 0,
            extracted_memory_ids=json.loads(row[8]) if row[8] else [],
            created_at=datetime.fromisoformat(row[9]),
            updated_at=datetime.fromisoformat(row[10]),
            committed_at=datetime.fromisoformat(row[11]) if row[11] else None,
        )

    @staticmethod
    def _row_to_message(row: tuple) -> SessionMessage:
        return SessionMessage(
            id=row[0], session_id=row[1], role=row[2], content=row[3],
            tool_call_id=row[4], tool_name=row[5], token_count=row[6] or 0,
            created_at=datetime.fromisoformat(row[7]),
        )
