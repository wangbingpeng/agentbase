#!/usr/bin/env python3
"""AgentBase adapter for LongMemEval benchmark.

Usage:
    # Step 1: Run benchmark to generate predictions
    export DASHSCOPE_API_KEY=your_key
    uv run python benchmarks/run_longmemeval.py --data-dir ../LongMemEval/data --output predictions.jsonl

    # Step 2: Evaluate with official script
    cd ../LongMemEval/src/evaluation
    python evaluate_qa.py gpt-4o ../../agentbase/predictions.jsonl ../../data/longmemeval_s_cleaned.json
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "agentbase-core" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "agentbase-sdk" / "src"))

from agentbase import AgentBase
from agentbase_core.models import AgentBaseConfig, ContextScope, EmbeddingConfig, IndexConfig, MemoryCategory
from agentbase_core.embedding import LiteLLMEmbedder

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

_async_client = None

def _get_client():
    global _async_client
    if _async_client is None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        _async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _async_client


# Global reader model — set by run_benchmark from --reader-model
_READER_MODEL = "qwen-plus"


def set_reader_model(model: str) -> None:
    """Set the global reader model used by call_llm."""
    global _READER_MODEL
    _READER_MODEL = model


def get_reader_model() -> str:
    """Get the current reader model."""
    return _READER_MODEL


async def call_llm(prompt: str, model: str | None = None, max_tokens: int = 1024) -> str:
    """Call LLM to generate an answer.

    If *model* is None, uses the global _READER_MODEL (set via --reader-model).
    """
    import backoff
    from openai import RateLimitError, APIError

    effective_model = model or _READER_MODEL
    client = _get_client()

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
    async def _call():
        kwargs = dict(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        # qwen3 series requires enable_thinking=false for non-streaming calls
        if "qwen3" in effective_model.lower():
            kwargs["extra_body"] = {"enable_thinking": False}
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    return await _call()


# ---------------------------------------------------------------------------
# Memory ingestion — turn-level splitting with structured metadata
# Zero-LLM: local rules for session digest + fact extraction
# ---------------------------------------------------------------------------

import re as _re

# ===== Bilingual regex patterns for local fact extraction =====
# Each tuple: (compiled_pattern, category)
# English patterns
_FACT_PATTERNS_PREF = [
    # English
    _re.compile(r"(?:i |i'm )(?:love|enjoy|prefer|like|really into|big fan of|obsessed with)\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i |i'm )(?:hate|can't stand|not a fan of|don't like|dislike)\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"my (?:favorite|favourite)\s+(.+?)(?:is|are)\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    # Chinese
    _re.compile(r"我(?:喜欢|爱|热爱|偏好|最爱|超爱|特别爱|很爱)\s*(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"我(?:讨厌|不喜欢|恨|不喜|不爱|不太喜欢|不怎么喜欢)\s*(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"(?:最|特别)(?:喜欢|爱|偏好)(?:的)?(.+?)(?:是|为)\s*(.+?)(?:[。，！,.]|$)"),
]

_FACT_PATTERNS_ENTITY = [
    # English
    _re.compile(r"(?:i work (?:at|for|in))\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i live (?:in|at))\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:my (?:name|job|role|title|position))\s+(?:is|are)\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i (?:have|own|bought|purchased|use))\s+(?:a |an )?(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i (?:graduated|study|studied|major(?:ed)?))\s+(?:in |from |with )?(.+?)(?:[.!]|$)", _re.IGNORECASE),
    # Chinese
    _re.compile(r"我(?:在|于)(.+?)(?:工作|上班|任职|就职)"),
    _re.compile(r"我(?:住在?|生活在|搬到了?)(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"我(?:的)?(?:名字|职位|职务|岗位|职业)(?:是|叫)\s*(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"我(?:买了?|购入|入手了?|有|拥有|用的是?)(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"我(?:毕业于?|就读于?|从)(.+?)(?:毕业|的|[。，！,.]|$)"),
    _re.compile(r"我(?:的)?(?:专业|学历|学位)(?:是|为)\s*(.+?)(?:[。，！,.]|$)"),
]

_FACT_PATTERNS_EVENT = [
    # English
    _re.compile(r"(?:i (?:went|go|traveled|travelled|visited|visit|moved))\s+(?:to )?(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i (?:started|began|joined|enrolled))\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    _re.compile(r"(?:i (?:switched|changed|updated))\s+(.+?)(?:[.!]|$)", _re.IGNORECASE),
    # Chinese
    _re.compile(r"我(?:去了?|去过|前往|到)(.+?)(?:[。，！,.]|旅游|出差|玩|$)"),
    _re.compile(r"我(?:开始|加入|参加了?|报名了?)(.+?)(?:[。，！,.]|$)"),
    _re.compile(r"我(?:换|改成|更新了?|升级了?)(.+?)(?:[。，！,.]|了|$)"),
    _re.compile(r"我(?:搬|搬到|搬去|移居)(.+?)(?:[。，！,.]|了|$)"),
]


def _local_session_digest(session: list[dict], date: str | None, session_index: int) -> str | None:
    """Zero-LLM session digest: extract key user utterances.

    Concatenates the first 3 user turns as a session-level digest.
    This provides FTS/vector-searchable keywords for session-level
    recall without any LLM cost.
    """
    user_utterances = [
        t["content"].strip()
        for t in session
        if t.get("role") == "user" and t.get("content", "").strip()
    ]
    if not user_utterances:
        return None

    header = f"[Session {session_index} | Date: {date or 'unknown'} | Turns: {len(session)}]"
    top_utts = user_utterances[:3]
    lines = [header]
    for utt in top_utts:
        lines.append(f"- {utt[:200]}")
    return "\n".join(lines)


def _local_extract_facts(session: list[dict], date: str | None, session_index: int) -> list[dict]:
    """Zero-LLM fact extraction using bilingual regex patterns.

    Extracts structured factual statements from user turns via
    pattern matching (English + Chinese). Returns up to 8 facts.
    """
    from agentbase_core.models import MemoryCategory
    facts: list[dict] = []

    for turn in session:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue
        content_lower = content.lower()

        # --- Preference patterns ---
        for pat in _FACT_PATTERNS_PREF:
            for m in pat.finditer(content if pat.flags & _re.IGNORECASE else content):
                span_text = content[m.start():min(m.end() + 20, len(content))].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.PREFERENCE,
                    "tags": ["preference"],
                    "confidence": 0.8,
                    "fact_type": "preference",
                })

        # --- Entity patterns ---
        for pat in _FACT_PATTERNS_ENTITY:
            for m in pat.finditer(content if pat.flags & _re.IGNORECASE else content):
                span_text = content[m.start():min(m.end() + 20, len(content))].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.ENTITY,
                    "tags": ["entity"],
                    "confidence": 0.8,
                    "fact_type": "entity",
                })

        # --- Event patterns ---
        for pat in _FACT_PATTERNS_EVENT:
            for m in pat.finditer(content if pat.flags & _re.IGNORECASE else content):
                span_text = content[m.start():min(m.end() + 20, len(content))].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.EVENT,
                    "tags": ["event"],
                    "confidence": 0.8,
                    "fact_type": "event",
                })

    # Deduplicate by content prefix
    seen: set[str] = set()
    unique: list[dict] = []
    for f in facts:
        key = f["content"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:8]


def _extract_ner_entries(session: list[dict], date: str | None, session_index: int) -> list[dict]:
    """Extract NER entities with surrounding sentence context.

    Unlike bare entity extraction (just "Hawaii"), this stores the
    full sentence containing the entity (e.g., "I went on a week-long
    trip to Hawaii with my family."). This makes NER entries
    FTS-searchable by both entity name AND context keywords.

    Uses regex-based NER (no spaCy dependency) for reliability.
    """
    from agentbase_core.models import MemoryCategory
    entries: list[dict] = []

    # Regex patterns for entity detection in user turns
    # Pattern 1: Multi-word capitalized entities (not sentence-initial)
    _RE_CAPS = _re.compile(
        r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    )
    # Pattern 2: Quoted names
    _RE_QUOTED = _re.compile(r'["\']([A-Z][a-z]+(?:\s+\w+)?)["\']')
    # Pattern 3: Number + unit
    _RE_NUM_UNIT = _re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:hours?|days?|weeks?|months?|years?|minutes?|"
        r"dollars?|cents?|miles?|km|kg|lbs?|GB|MB|TB|Mbps|Hz)\b",
        _re.IGNORECASE,
    )
    # Pattern 4: "X is/was Y" definitions (captures entity + value)
    _RE_DEFINITION = _re.compile(
        r"\b(my\s+(?:\w+\s+){0,2}(?:is|was|are|were)\s+(.+?))(?:[.,;!]|$)",
        _re.IGNORECASE,
    )

    seen_entities: set[str] = set()

    for turn in session:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue

        # Split content into sentences for context extraction
        sentences = _re.split(r'(?<=[.!?])\s+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Check for entities in this sentence
            entities_in_sentence = []

            # Multi-word capitalized entities
            for m in _RE_CAPS.finditer(sentence):
                entity = m.group(1).strip()
                if len(entity) > 3 and entity.lower() not in seen_entities:
                    entities_in_sentence.append(entity)
                    seen_entities.add(entity.lower())

            # Quoted names
            for m in _RE_QUOTED.finditer(sentence):
                entity = m.group(1).strip()
                if len(entity) > 2 and entity.lower() not in seen_entities:
                    entities_in_sentence.append(entity)
                    seen_entities.add(entity.lower())

            # Number + unit
            for m in _RE_NUM_UNIT.finditer(sentence):
                entity = m.group(0).strip()
                if entity.lower() not in seen_entities:
                    entities_in_sentence.append(entity)
                    seen_entities.add(entity.lower())

            # If this sentence contains entities, create NER entries
            # with the FULL sentence as context (not just the entity name)
            if entities_in_sentence:
                for entity in entities_in_sentence:
                    # Store: "ENTITY_NAME || FULL_SENTENCE"
                    # This way FTS can match both the entity name and context keywords
                    ner_content = f"{entity} || {sentence}"
                    entries.append({
                        "content": ner_content,
                        "entity_name": entity,
                        "category": MemoryCategory.ENTITY,
                        "tags": ["ner", "entity"],
                        "confidence": 0.75,
                        "fact_type": "ner_entity",
                    })

    # Also extract definition-style facts ("my X is Y")
    for turn in session:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue

        for m in _RE_DEFINITION.finditer(content):
            full_match = m.group(1).strip().rstrip(".,;:")
            value = m.group(2).strip().rstrip(".,;:")
            if len(value) > 2 and value.lower() not in seen_entities:
                seen_entities.add(value.lower())
                entries.append({
                    "content": full_match,
                    "entity_name": value,
                    "category": MemoryCategory.ENTITY,
                    "tags": ["ner", "definition"],
                    "confidence": 0.8,
                    "fact_type": "ner_definition",
                })

    # Deduplicate by entity name
    seen_names: set[str] = set()
    unique: list[dict] = []
    for e in entries:
        key = e["entity_name"].lower()
        if key not in seen_names:
            seen_names.add(key)
            unique.append(e)
    return unique[:15]  # Cap at 15 NER entries per session


async def ingest_history(db: AgentBase, item: dict, session_summary: bool = True) -> None:
    """Ingest all haystack sessions into AgentBase as per-turn memory entries.

    Zero-LLM ingestion: uses local rules (regex + string concatenation)
    instead of LLM calls for session digest and fact extraction.
    Ingest cost is near-instant regardless of session count.
    """
    from agentbase_core.models import ContextEntry, ContextType, OriginType
    from agentbase_core.index.tokenizer import tokenize_text

    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]

    # Parse all session datetimes upfront
    from datetime import datetime, timezone
    session_dts: list[datetime | None] = []
    for date in dates:
        session_dt = None
        if date:
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    session_dt = datetime.strptime(date.strip(), fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
        session_dts.append(session_dt)

    # --- Build all ContextEntry objects (zero LLM) ---
    all_entries: list[ContextEntry] = []

    for i, (session, date) in enumerate(zip(sessions, dates)):
        session_dt = session_dts[i]

        for j, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if not content:
                continue

            # Auto-classify category
            content_lower = content.lower()
            pref_indicators = [
                "love", "hate", "enjoy", "favorite", "favourite",
                "prefer", "really into", "big fan", "not a fan",
                "obsessed", "can't stand", "i like", "i don't like",
                "always", "never", "usually", "typically",
                "my go-to", "i tend to", "best", "worst",
                "bought", "purchased", "i own", "i use", "using",
                "switched to", "tried", "looking for", "need a",
                "i want", "looking to", "considering", "decided to",
                "i chose", "i picked", "i went with",
                "good", "great", "bad", "terrible", "amazing",
                "better", "worse", "recommend", "suggest",
                "vegetarian", "vegan", "gluten-free", "kosher", "halal",
                "allergic", "intolerant", "diet", "workout", "exercise",
            ]
            if role == "user":
                category = MemoryCategory.PREFERENCE if any(ind in content_lower for ind in pref_indicators) else MemoryCategory.EVENT
            else:
                category = MemoryCategory.ENTITY

            turn_content = f"[{role}]: {content}"
            entry = ContextEntry(
                l2_full=turn_content,
                context_type=ContextType.MEMORY,
                memory_category=category,
                tags=["longmemeval", f"session_{i}", f"turn_{j}", role],
                confidence=0.9,
                scope=ContextScope.GLOBAL,
                source="conversation",
                origin_type=OriginType.MANUAL,
                extra={
                    "session_index": i,
                    "turn_index": j,
                    "session_date": date,
                    "role": role,
                },
            )
            if session_dt is not None:
                entry.created_at = session_dt
                entry.valid_from = session_dt

            entry.fts_text = tokenize_text(turn_content, tokenizer="icu")
            entry.mark_active()
            all_entries.append(entry)

        # --- Session digest + local facts (zero LLM) ---
        if session_summary:
            # Session digest entry (replaces LLM-generated summary)
            digest_text = _local_session_digest(session, date, i)
            if digest_text:
                full_transcript = "\n".join(
                    f"[{t.get('role', 'unknown')}]: {t.get('content', '').strip()}"
                    for t in session
                    if t.get("content", "").strip()
                )
                summary_content = f"{digest_text}\n\n--- Full Transcript ---\n{full_transcript}"

                summary_entry = ContextEntry(
                    l0_abstract=digest_text,
                    l1_overview=digest_text,
                    l2_full=summary_content,
                    context_type=ContextType.MEMORY,
                    memory_category=MemoryCategory.EVENT,
                    tags=["longmemeval", f"session_{i}", "session_summary"],
                    confidence=0.95,
                    scope=ContextScope.GLOBAL,
                    source="conversation",
                    origin_type=OriginType.MANUAL,
                    extra={
                        "session_index": i,
                        "session_date": date,
                        "role": "summary",
                        "num_turns": len(session),
                    },
                )
                if session_dt is not None:
                    summary_entry.created_at = session_dt
                    summary_entry.valid_from = session_dt

                summary_entry.fts_text = tokenize_text(summary_content, tokenizer="icu")
                summary_entry.mark_active()
                all_entries.append(summary_entry)

            # Local fact entries (bilingual regex, zero LLM)
            facts = _local_extract_facts(session, date, i)
            for fact in facts:
                fact_entry = ContextEntry(
                    l0_abstract=fact["content"],
                    l1_overview=fact["content"],
                    l2_full=fact["content"],
                    context_type=ContextType.MEMORY,
                    memory_category=fact["category"],
                    tags=["longmemeval", f"session_{i}", "fact"] + fact.get("tags", []),
                    confidence=fact.get("confidence", 0.8),
                    scope=ContextScope.GLOBAL,
                    source="conversation",
                    origin_type=OriginType.MANUAL,
                    extra={
                        "session_index": i,
                        "session_date": date,
                        "role": "fact",
                        "fact_type": fact.get("fact_type", "statement"),
                    },
                )
                if session_dt is not None:
                    fact_entry.created_at = session_dt
                    fact_entry.valid_from = session_dt

                fact_entry.fts_text = tokenize_text(fact["content"], tokenizer="icu")
                fact_entry.mark_active()
                all_entries.append(fact_entry)

            # NER entity tagging: add entity names as tags on turn entries
            # (not separate NER entries — avoids diluting search results)
            ner_entries = _extract_ner_entries(session, date, i)
            session_ner_tags = ["ner"]
            for ner in ner_entries:
                ent_name = ner.get("entity_name", "").strip()
                if ent_name:
                    # Sanitize: spaces→underscores, prefix with ner_
                    tag_name = "ner_" + "_".join(
                        ch if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" else "_"
                        for ch in ent_name
                    ).strip("_")
                    if len(tag_name) <= 60 and tag_name not in session_ner_tags:
                        session_ner_tags.append(tag_name)

            # Apply NER tags to all turn entries from this session
            if len(session_ner_tags) > 1:
                for entry in all_entries:
                    entry_tags = entry.tags or []
                    extra = entry.extra or {}
                    # Only tag turn-level and fact entries from this session
                    if (extra.get("session_index") == i and
                        ("turn_" in str(entry_tags) or "fact" in entry_tags)):
                        for tag in session_ner_tags:
                            if tag not in entry_tags:
                                entry_tags.append(tag)
                        entry.tags = entry_tags
                        # Also enrich fts_text with NER tags for FTS5 searchability
                        tag_text = " ".join(session_ner_tags)
                        entry.fts_text = (
                            (entry.fts_text or "") + " " +
                            tokenize_text(tag_text, tokenizer="icu")
                        ).strip()

    # Batch insert: bypass dedup + LLM layer generation for speed
    if all_entries:
        store = db._engine._store
        index = db._engine._index
        # Batch add to store (SQLite INSERT)
        for entry in all_entries:
            await store.add(entry)
        # Batch add to FTS index
        try:
            await index.add_batch(all_entries)
        except Exception:
            # Fallback: add one by one
            for entry in all_entries:
                try:
                    await index.add(entry)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Retrieval + Answer generation — Two-Stage Retrieve-Then-Read
# ---------------------------------------------------------------------------

# Answer prompts — use {context} (retrieved conversation history)

ANSWER_PROMPT_GENERIC = """Based on the following conversation history, answer the question concisely and factually.
- If the information is in the history, provide a direct, specific answer.
- If the information requires some reasoning or cross-referencing, provide your best answer based on the available evidence.
- Only say "I don't have that information" if there is absolutely NO relevant information in the history.

Conversation History:
{context}

Question: {question}

Answer:"""

ANSWER_PROMPT_TEMPORAL = """Answer the temporal question based on the following conversation history.

IMPORTANT RULES:
- Each passage is labeled with its session date (e.g., [Session 0 | Date: 2024/01/15]). Use these dates for ALL time calculations.
- When asked "how many days/weeks/months between X and Y", calculate the exact difference from the session dates.
- When asked "what happened first/last", compare the session dates to determine the order.
- When asked "how many times", carefully count each distinct occurrence across all dated sessions.
- Pay close attention to relative time expressions like "today", "yesterday", "last week", "2 weeks ago" — these are relative to the session date, NOT today's date.
- If the question mentions a specific event, search ALL history for mentions and compare their dates.
- For "when was the first/last time" questions, identify the earliest/latest session date where the event appears.
- If dates seem inconsistent, use the session date label as ground truth.
- Off-by-one tolerance: If the question asks for the number of days/weeks/months, a difference of ±1 is acceptable (e.g., answering 18 when the exact answer is 19 is still correct).
- Provide a direct, specific answer (just the number when asked "how many").
- Show your work: mention the relevant session dates you used for the calculation.
- Only say "I don't have that information" if there is absolutely NO relevant information in the history.

Conversation History:
{context}

Question: {question}

Answer:"""

ANSWER_PROMPT_KNOWLEDGE_UPDATE = """Answer the question about CURRENT/LATEST information based on the following conversation history.
- Focus on the MOST RECENT mentions of the topic.
- If the information was updated or changed, provide the LATEST value.
- Ignore older, superseded information.
- Only say "I don't have that information" if there is absolutely NO relevant information in the history.

Conversation History:
{context}

Question: {question}

Answer:"""

ANSWER_PROMPT_PREFERENCE = """Infer the user's preferences and answer the question based on the following conversation history.

IMPORTANT RULES:
- The user may NOT explicitly state their preferences. You must INFER them from:
  - Products they own or have mentioned (brands, models, types)
  - Activities they enjoy or have done
  - Past choices and decisions they've described
  - Brands, categories, or specific items they've discussed
  - Their hobbies, interests, and habits revealed in conversation
  - Their lifestyle, dietary choices, living situation, and routines
  - Things they have bought, used, or expressed interest in
  - Opinions they have shared (positive or negative) about products or experiences
- Use these inferred preferences to make personalized suggestions.
- Focus on what the USER said/did, not the assistant's responses.
- When recommending, be specific — mention actual brands, products, or types that match the inferred preference.
- Do NOT give generic advice. Tailor your answer specifically to what you know about this user.
- CRITICAL: Even if the user did not directly state a preference, you can still infer it. For example:
  - If the user mentions owning a Dell laptop, you can infer they prefer Dell/Windows.
  - If the user mentions being vegetarian, you can infer they prefer vegetarian food options.
  - If the user mentions going to the gym 3 times a week, you can infer they are fitness-oriented.
- Provide a direct, specific answer based on the inferred preferences.
- Only say "I don't have that information" if there is absolutely NO relevant information in the history.

Conversation History:
{context}

Question: {question}

Answer:"""

ANSWER_PROMPT_MULTI_SESSION = """Answer the question based on the following conversation history from MULTIPLE sessions.

IMPORTANT RULES:
- Each passage is labeled with its session number and date (e.g., [Session 2 | Date: 2024/03/15 | Role: user]).
- You MUST search through ALL history exhaustively. Do not stop after finding one mention.
- When asked "how many" or "how much", carefully enumerate EACH instance from EACH session. Count methodically:
  1. Go through each session in date order
  2. In each session, find every relevant mention
  3. Count them all
  4. Report the total count
- Double-check your count against all the history provided.
- When asked about specific items/people/events across sessions, list each one with its session.
- If dates are relevant, use the session dates for calculations.
- Pay attention to the Role label — focus on what the user said, not the assistant.
- CRITICAL: Even if the information requires reasoning or cross-referencing multiple sessions, provide your BEST answer based on the available evidence. Do NOT refuse to answer just because the evidence is incomplete or requires inference — use the dates, quantities, and details you found to construct the best possible answer.
- Provide a CONCISE answer. Start with the direct answer, then briefly list the supporting evidence.
  Format: "X. Evidence: (1) Session N: ... (2) Session M: ..."
- Do NOT write long paragraphs. Keep the answer structured and to the point.
- Only say "I don't have that information" if there is absolutely NO relevant information anywhere in the history.

Conversation History:
{context}

Question: {question}

Answer:"""


# Map question_type to answer prompt and query_type
QTYPE_CONFIG = {
    "temporal-reasoning": {
        "prompt": ANSWER_PROMPT_TEMPORAL,
        "query_type": "temporal-reasoning",
    },
    "knowledge-update": {
        "prompt": ANSWER_PROMPT_KNOWLEDGE_UPDATE,
        "query_type": "knowledge-update",
    },
    "single-session-preference": {
        "prompt": ANSWER_PROMPT_PREFERENCE,
        "query_type": "single-session-preference",
    },
    "multi-session": {
        "prompt": ANSWER_PROMPT_MULTI_SESSION,
        "query_type": "multi-session",
    },
    # Default for other types
    "single-session-assistant": {
        "prompt": ANSWER_PROMPT_GENERIC,
        "query_type": "single-session-assistant",
    },
    "single-session-user": {
        "prompt": ANSWER_PROMPT_GENERIC,
        "query_type": "single-session-user",
    },
}


def _decompose_query(question: str) -> list[str]:
    """Decompose a temporal/multi-session query into sub-queries for broader recall.

    Strategy:
    1. Always include the original question
    2. Extract key entities/phrases and search for them individually
    3. For "how many X" questions, search for X as a standalone query
    """
    import re

    queries = [question]  # always include original

    q_lower = question.lower()

    # Pattern: "between X and Y" → extract X and Y separately
    between_match = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\.)", question, re.IGNORECASE)
    if between_match:
        queries.append(between_match.group(1).strip())
        queries.append(between_match.group(2).strip())

    # Pattern: "first/last, X or Y" → extract X and Y
    order_match = re.search(r"(?:first|last|order).+?(\w+(?:\s+\w+){0,3})\s+or\s+(\w+(?:\s+\w+){0,3})", question, re.IGNORECASE)
    if order_match:
        queries.append(order_match.group(1).strip())
        queries.append(order_match.group(2).strip())

    # Pattern: "how many X" / "how much X" → extract X as a search term
    how_match = re.search(r"how (?:many|much|often|long|far).+?(?:did|have|was|were|do|does|has|is|are)\s+(.+?)(?:\?|$|\.)", question, re.IGNORECASE)
    if how_match:
        entity = how_match.group(1).strip()
        if len(entity) > 2:
            queries.append(entity)

    # Pattern: "the day I X" / "when I X" → extract X
    when_match = re.search(r"(?:the day|the time|when) (?:I|you|we) (.+?)(?:\?|$|,|\.)", question, re.IGNORECASE)
    if when_match:
        queries.append(when_match.group(1).strip())

    # For multi-session "how many" queries, also try extracting noun phrases
    # e.g. "How many different doctors did I visit?" → "doctors"
    if "how many" in q_lower or "how much" in q_lower:
        # Extract key nouns (simple heuristic: last few content words)
        words = re.findall(r"\b([a-z]{3,}(?:s|es|ies)?)\b", q_lower)
        # Filter stop words
        stop_words = {"the", "and", "was", "were", "did", "have", "has", "had",
                      "been", "how", "many", "much", "does", "that", "this",
                      "from", "with", "for", "not", "but", "are", "its",
                      "all", "total", "overall", "different", "various"}
        keywords = [w for w in words if w not in stop_words]
        if keywords:
            # Take the most significant keyword as an extra search query
            queries.append(keywords[-1])  # usually the target noun
            # Also try a 2-word phrase if available
            if len(keywords) >= 2:
                queries.append(f"{keywords[-2]} {keywords[-1]}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_stripped = q.strip().rstrip("?.!")
        if q_stripped and q_stripped.lower() not in seen:
            seen.add(q_stripped.lower())
            unique.append(q_stripped)

    return unique[:5]  # cap at 5 sub-queries


def _decompose_preference_query(question: str) -> list[str]:
    """Decompose a preference query into sub-queries for broader recall.

    Strategy:
    1. Always include the original question
    2. Extract key entities/topics and search for them directly
    3. Add preference-indicator search terms
    """
    import re

    queries = [question]  # always include original
    q_lower = question.lower()

    # Extract the core topic from common preference question patterns
    # Pattern: "What kind of X should I..." / "What type of X..." / "Can you suggest X"
    topic_patterns = [
        r"what kind of (.+?)(?:\s+should|\s+can|\s+would|\?|$)",
        r"what type of (.+?)(?:\s+should|\s+can|\s+would|\?|$)",
        r"what sort of (.+?)(?:\s+should|\s+can|\s+would|\?|$)",
        r"can you (?:suggest|recommend) (?:a|an|some)?\s*(.+?)(?:\?|$|\.)",
        r"(?:suggest|recommend) (?:a|an|some)?\s*(.+?)(?:\?|$|\.)",
        r"what(?:'s| is) (?:a |the )?(?:good|best) (.+?)(?:\?|$|\.)",
        r"(?:any|some) (.+?) (?:tips|advice|ideas|suggestions)(?:\?|$|\.)",
    ]
    for pattern in topic_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            topic = match.group(1).strip().rstrip("?.! ")
            if len(topic) > 2:
                queries.append(topic)
                # Also search with preference-related modifiers
                queries.append(f"{topic} preference")
                queries.append(f"{topic} like")
            break

    # Extract key nouns as standalone queries
    words = re.findall(r"\b([a-z]{3,}(?:s|es|ies)?)\b", q_lower)
    stop_words = {"the", "and", "was", "were", "did", "have", "has", "had",
                  "been", "how", "many", "much", "does", "that", "this",
                  "from", "with", "for", "not", "but", "are", "its",
                  "all", "what", "kind", "type", "sort", "should",
                  "could", "would", "can", "you", "suggest", "recommend",
                  "any", "some", "good", "best", "tips", "advice"}
    keywords = [w for w in words if w not in stop_words]
    if keywords:
        # Add the most relevant keyword
        queries.append(keywords[-1])
        if len(keywords) >= 2:
            queries.append(f"{keywords[-2]} {keywords[-1]}")

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        q_stripped = q.strip().rstrip("?.!")
        if q_stripped and q_stripped.lower() not in seen:
            seen.add(q_stripped.lower())
            unique.append(q_stripped)

    return unique[:5]  # cap at 5 sub-queries


async def answer_question(
    db: AgentBase,
    question: str,
    question_type: str | None = None,
) -> str:
    """Retrieve relevant memories and generate an answer.

    Uses query-type-aware retrieval with type-specific prompts,
    session-aware context assembly, and query decomposition for
    temporal/multi-session questions.
    """
    # Determine query type and prompt from question_type
    cfg = QTYPE_CONFIG.get(question_type, {"prompt": ANSWER_PROMPT_GENERIC, "query_type": None})
    answer_prompt = cfg["prompt"]
    query_type = cfg["query_type"]

    # --- Type-aware retrieval parameters ---
    top_k = 30  # default (increased from 20 for better recall)
    if question_type == "multi-session":
        top_k = 200  # M1: need exhaustive recall across sessions — expanded from 100
    elif question_type == "temporal-reasoning":
        top_k = 60  # need more context for date reasoning — expanded from 50
    elif question_type == "single-session-preference":
        top_k = 50  # need broader context for preference inference — expanded from 40
    elif question_type == "single-session-user":
        top_k = 50  # increased from 40: details buried in large haystacks
    elif question_type == "knowledge-update":
        top_k = 40  # increased from 30: need recent entries with broader context

    # --- Aggregation-aware top_k boost ---
    # Questions like "how many", "total", "how much" need exhaustive recall
    # D2: Use IntentAnalyzer from core for consistent aggregation detection
    from agentbase_core.retrieval.intent import IntentAnalyzer as _CoreIntentAnalyzer
    is_aggregation = _CoreIntentAnalyzer.is_aggregation_query(question)
    if is_aggregation and top_k < 120:
        top_k = 120  # aggregation queries need broad recall to count correctly

    # --- Type-aware max_tokens for LLM answer generation ---
    max_tokens = 1024  # default (G2: increased from 512)
    if question_type == "temporal-reasoning":
        max_tokens = 1536  # temporal answers need more reasoning space
    elif question_type == "multi-session":
        max_tokens = 1536  # enumeration needs more space

    # --- Query decomposition for temporal/multi-session/preference (T1/M4/P2) ---
    # Also apply to single-session-user for better recall (S1)
    all_results: list = []
    seen_ids: set[str] = set()

    if question_type == "multi-session":
        # Multi-session: search with decomposed sub-queries + parallel Stage 2.
        # Stage 2 uses batch SQLite queries (not serial db.find) to fetch
        # uncovered sessions, avoiding the 30+ serial embedding API calls
        # that caused the original 33x slowdown.
        sub_queries = _decompose_query(question)
        for sq in sub_queries:
            results = await db.find(query=sq, query_type=query_type, top_k=top_k)
            for r in results:
                if r.entry.id not in seen_ids:
                    seen_ids.add(r.entry.id)
                    all_results.append(r)

        # Stage 2 (uncovered session completion) is now handled by the
        # core engine's _multi_session_strategy, so no need to do it here.

    elif question_type == "temporal-reasoning":
        sub_queries = _decompose_query(question)
        for sq in sub_queries:
            results = await db.find(query=sq, query_type=query_type, top_k=top_k)
            for r in results:
                if r.entry.id not in seen_ids:
                    seen_ids.add(r.entry.id)
                    all_results.append(r)
    elif question_type == "single-session-preference":
        # P2: For preference queries, search with both the original query and
        # extracted entity keywords to broaden preference-signal recall
        sub_queries = _decompose_preference_query(question)
        for sq in sub_queries:
            results = await db.find(query=sq, query_type="preference", top_k=top_k)
            for r in results:
                if r.entry.id not in seen_ids:
                    seen_ids.add(r.entry.id)
                    all_results.append(r)
    elif question_type == "single-session-user":
        # S1: For single-session-user, also decompose query for better recall
        # This helps find details buried in large haystacks (40-50 sessions)
        sub_queries = _decompose_query(question)
        for sq in sub_queries:
            results = await db.find(query=sq, query_type=query_type, top_k=top_k)
            for r in results:
                if r.entry.id not in seen_ids:
                    seen_ids.add(r.entry.id)
                    all_results.append(r)
    else:
        all_results = await db.find(query=question, query_type=query_type, top_k=top_k)

    if not all_results:
        return "I don't have that information."

    # --- G1: Context assembly with session metadata ---
    # For multi-session: prioritize session summaries, then group turns by session
    is_multi_session = question_type == "multi-session"

    if is_multi_session:
        # Separate summary, fact, and turn entries
        summary_results = []
        fact_results = []
        turn_results = []
        for r in all_results:
            tags = r.entry.tags or []
            if "session_summary" in tags:
                summary_results.append(r)
            elif "fact" in tags:
                fact_results.append(r)
            else:
                turn_results.append(r)

        # Build context: summaries first (session overview), then facts, then turn details
        context_parts = []

        # Session summaries — provide session-level overview
        if summary_results:
            context_parts.append("=== SESSION SUMMARIES ===")
            # Sort summaries by session index
            summary_results.sort(
                key=lambda r: (r.entry.extra or {}).get("session_index", 0)
            )
            for r in summary_results:
                entry = r.entry
                extra = entry.extra or {}
                session_idx = extra.get("session_index", "")
                session_date = extra.get("session_date", "")
                # For summaries, use the l0_abstract (concise summary text)
                summary_text = entry.l0_abstract or entry.l1_overview or ""
                header_parts = []
                if session_idx != "":
                    header_parts.append(f"Session {session_idx}")
                if session_date:
                    header_parts.append(f"Date: {session_date}")
                header = " | ".join(header_parts)
                context_parts.append(f"[{header} — Summary]\n{summary_text}")

        # Extracted facts — provide precise factual statements
        if fact_results:
            context_parts.append("=== KEY FACTS ===")
            # Group facts by session
            facts_by_session: dict[int, list] = {}
            for r in fact_results:
                extra = r.entry.extra or {}
                si = extra.get("session_index", 0)
                if si not in facts_by_session:
                    facts_by_session[si] = []
                facts_by_session[si].append(r)
            for si in sorted(facts_by_session.keys()):
                session_date = (facts_by_session[si][0].entry.extra or {}).get("session_date", "")
                header_parts = [f"Session {si}"]
                if session_date:
                    header_parts.append(f"Date: {session_date}")
                header = " | ".join(header_parts)
                fact_texts = []
                for r in facts_by_session[si]:
                    fact_content = r.entry.l0_abstract or r.entry.l2_full or ""
                    if fact_content.strip():
                        fact_texts.append(f"  - {fact_content.strip()}")
                if fact_texts:
                    context_parts.append(f"[{header} — Facts]\n" + "\n".join(fact_texts))

        if summary_results or fact_results:
            context_parts.append("=== DETAILED TURNS ===")

        # Turn-level details
        for r in turn_results:
            entry = r.entry
            text = entry.l2_full or entry.l1_overview or entry.l0_abstract or ""
            if not text.strip():
                continue

            extra = entry.extra or {}
            session_date = extra.get("session_date", "")
            session_idx = extra.get("session_index", "")
            role = extra.get("role", "")

            if session_date or session_idx != "":
                header_parts = []
                if session_idx != "":
                    header_parts.append(f"Session {session_idx}")
                if session_date:
                    header_parts.append(f"Date: {session_date}")
                if role:
                    header_parts.append(f"Role: {role}")
                header = " | ".join(header_parts)
                text = f"[{header}]\n{text}"

            context_parts.append(text)
    else:
        # Default context assembly (non-multi-session)
        # Separate fact and turn entries for better context structure
        fact_results = []
        turn_results = []
        for r in all_results:
            tags = r.entry.tags or []
            if "fact" in tags:
                fact_results.append(r)
            else:
                turn_results.append(r)

        context_parts = []

        # Show extracted facts first for quick reference
        if fact_results:
            context_parts.append("=== KEY FACTS ===")
            for r in fact_results:
                entry = r.entry
                fact_content = entry.l0_abstract or entry.l2_full or ""
                if not fact_content.strip():
                    continue
                extra = entry.extra or {}
                session_date = extra.get("session_date", "")
                session_idx = extra.get("session_index", "")
                header_parts = []
                if session_idx != "":
                    header_parts.append(f"Session {session_idx}")
                if session_date:
                    header_parts.append(f"Date: {session_date}")
                header = " | ".join(header_parts)
                if header:
                    context_parts.append(f"[{header} — Fact] {fact_content.strip()}")
                else:
                    context_parts.append(f"[Fact] {fact_content.strip()}")
            context_parts.append("=== CONVERSATION ===")

            context_parts.append("=== CONVERSATION ===")

        # Turn-level details
        for r in turn_results:
            entry = r.entry
            text = entry.l2_full or entry.l1_overview or entry.l0_abstract or ""
            if not text.strip():
                continue

            extra = entry.extra or {}
            session_date = extra.get("session_date", "")
            session_idx = extra.get("session_index", "")
            role = extra.get("role", "")

            if session_date or session_idx != "":
                header_parts = []
                if session_idx != "":
                    header_parts.append(f"Session {session_idx}")
                if session_date:
                    header_parts.append(f"Date: {session_date}")
                if role:
                    header_parts.append(f"Role: {role}")
                header = " | ".join(header_parts)
                text = f"[{header}]\n{text}"

            context_parts.append(text)

    context = "\n\n---\n\n".join(context_parts)

    # --- Context deduplication and session-ordered sorting ---
    # Deduplicate entries that appear in both summary and turn sections
    seen_content_hashes: set[str] = set()
    deduped_parts = []
    for part in context_parts:
        # Use first 80 chars as dedup key (handles near-duplicate turns)
        content_key = part.strip().lower()[:150]
        if content_key not in seen_content_hashes:
            seen_content_hashes.add(content_key)
            deduped_parts.append(part)
    context = "\n\n---\n\n".join(deduped_parts)

    if not context.strip():
        return "I don't have that information."

    # --- Generate answer from context (single LLM call) ---
    prompt = answer_prompt.format(context=context, question=question)

    # D2: Append aggregation-aware prompt suffix for counting/totaling queries
    if is_aggregation:
        from agentbase_core.retrieval.engine import RetrievalEngine as _CoreRE
        prompt += _CoreRE.get_aggregation_prompt_suffix()

    answer = await call_llm(prompt, max_tokens=max_tokens)
    return answer


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    data_file: str,
    output_file: str,
    limit: int | None = None,
    reader_model: str = "qwen-plus",
    vector_enabled: bool = False,
    resume: bool = False,
    session_summary: bool = True,
) -> None:
    """Run the full LongMemEval benchmark with AgentBase."""
    set_reader_model(reader_model)
    print(f"Loading data from {data_file}...")
    with open(data_file) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"Limited to first {limit} questions")

    # Resume: skip already completed questions
    completed_qids: set[str] = set()
    if resume and Path(output_file).exists():
        with open(output_file) as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        completed_qids.add(rec["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Resume mode: skipping {len(completed_qids)} already completed questions")

    print(f"Total questions: {len(data)}")
    if completed_qids:
        remaining = [d for d in data if d["question_id"] not in completed_qids]
        print(f"Remaining questions: {len(remaining)}")
    print(f"Vector enabled: {vector_enabled}")
    print(f"Session summary: {session_summary}")
    print(f"Reader model: {reader_model}")

    # Prepare embedding if vector search is enabled
    embedder = None
    if vector_enabled:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        emb_config = EmbeddingConfig(
            model="text-embedding-v4",
            dimensions=1024,
            api_base=base_url,
            api_key=api_key,
        )
        embedder = LiteLLMEmbedder(emb_config)
        print(f"Embedding model: {emb_config.model} (dim={emb_config.dimensions})")

    # Prepare output directory for per-question databases
    db_dir = Path(tempfile.gettempdir()) / f"longmemeval_agentbase_{int(time.time())}"
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
        except OSError:
            print(f"  Warning: could not clean up {db_dir}, creating new dir anyway")
    db_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0
    completed = 0

    # Filter out completed questions if resuming
    if completed_qids:
        data = [d for d in data if d["question_id"] not in completed_qids]

    # Open output file in append mode for streaming writes
    out_f = open(output_file, "a")

    for idx, item in enumerate(data):
        qid = item["question_id"]
        question = item["question"]
        qtype = item["question_type"]

        print(f"\n[{idx+1}/{len(data)}] QID={qid} type={qtype}")
        print(f"  Q: {question[:100]}...")

        start_time = time.time()

        # Each question gets an isolated database
        db_path = db_dir / f"{qid}.db"
        if vector_enabled:
            cfg = AgentBaseConfig(
                data_dir=db_path.parent,
                db_filename=db_path.name,
                embedding=emb_config,
                index=IndexConfig(vector_enabled=True),
            )
            db = AgentBase(config=cfg, embedder=embedder)
        else:
            db = AgentBase(path=str(db_path))
        await db.initialize()

        try:
            # Step 1: Ingest history (optimized, no dedup)
            t_ingest_start = time.time()
            await ingest_history(db, item, session_summary=session_summary)
            t_ingest = time.time() - t_ingest_start

            # Step 2: Retrieve and answer (with query_type)
            t_answer_start = time.time()
            hypothesis = await answer_question(db, question, question_type=qtype)
            t_answer = time.time() - t_answer_start
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            hypothesis = "Error: unable to answer"
            t_ingest = 0.0
            t_answer = 0.0
        finally:
            await db.close()

        elapsed = time.time() - start_time
        total_time += elapsed
        completed += 1

        # Streaming write: append result immediately
        result = {"question_id": qid, "hypothesis": hypothesis}
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()

        print(f"  A: {hypothesis[:200]}...")
        print(f"  Time: {elapsed:.1f}s (ingest={t_ingest:.1f}s answer={t_answer:.1f}s)")

        # Print ETA every 50 questions
        if completed % 50 == 0 and completed > 0:
            avg = total_time / completed
            eta = avg * (len(data) - completed)
            print(f"\n  === Progress: {completed}/{len(data)} ({completed/len(data)*100:.0f}%) | Avg: {avg:.1f}s/q | ETA: {eta/60:.1f}min ===")

    out_f.close()

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average per question: {total_time/len(data):.1f}s")
    print(f"\nNext step: Run evaluation with:")
    print(f"  cd /path/to/LongMemEval/src/evaluation")
    print(f"  python evaluate_qa.py gpt-4o {Path(output_file).resolve()} {Path(data_file).resolve()}")

    # Cleanup
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
            print(f"\nCleaned up temp databases in {db_dir}")
        except OSError:
            print(f"\nWarning: could not fully clean up {db_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LongMemEval with AgentBase")
    parser.add_argument("--data-dir", default="../LongMemEval/data", help="LongMemEval data directory")
    parser.add_argument("--dataset", default="longmemeval_s_cleaned.json", help="Dataset file name")
    parser.add_argument("--output", default="predictions.jsonl", help="Output predictions file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--reader-model", default="qwen-plus", help="LLM model for answering (default: qwen-plus via DashScope)")
    parser.add_argument("--vector", action="store_true", help="Enable vector search (RRF hybrid retrieval)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output, skipping completed questions")
    parser.add_argument("--session-summary", action="store_true", default=True, help="Generate LLM session summaries and ADD-only facts for each session (default: True)")
    parser.add_argument("--no-session-summary", action="store_false", dest="session_summary", help="Disable session summaries and fact extraction")
    args = parser.parse_args()

    data_file = Path(args.data_dir) / args.dataset
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please download the dataset first. See README for instructions.")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable is required")
        sys.exit(1)

    asyncio.run(run_benchmark(
        data_file=str(data_file),
        output_file=args.output,
        limit=args.limit,
        reader_model=args.reader_model,
        vector_enabled=args.vector,
        resume=args.resume,
        session_summary=args.session_summary,
    ))


if __name__ == "__main__":
    main()
