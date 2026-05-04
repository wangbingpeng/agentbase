#!/usr/bin/env python3
"""AgentBase adapter for LoCoMo benchmark.

LoCoMo (Long-term Conversational Memory) is an ACL 2024 benchmark for
evaluating very long-term conversational memory of LLM agents.

Usage:
    # Step 1: Run benchmark to generate predictions
    export DASHSCOPE_API_KEY=your_key
    uv run python benchmarks/run_locomo.py --data-dir ../locomo/data --output predictions_locomo.jsonl

    # Step 2 (optional): Evaluation is built-in (F1 for QA, ROUGE for event summarization)
    # Results are printed at the end of the run.

Data format (locomo10.json):
    Each sample represents one conversation with:
    - conversation: sessions (session_1, session_2, ...) with timestamps
    - qa: question-answer pairs with category labels
    - event_summary: annotated event summaries per speaker per session
    - observation / session_summary: generated metadata (used in RAG baselines)

QA categories:
    1 = single-hop, 2 = multi-hop, 3 = temporal,
    4 = open-domain (commonsense/world knowledge), 5 = adversarial
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "agentbase-core" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "agentbase-sdk" / "src"))

# Optimization A: maximum concurrent QA requests (avoids DashScope rate limits)
QA_CONCURRENCY = 5

# Default reader model (overridden by --reader-model CLI flag at startup)
DEFAULT_READER_MODEL = "qwen3.6-plus"
_READER_MODEL: str = DEFAULT_READER_MODEL

from agentbase import AgentBase
from agentbase_core.models import AgentBaseConfig, ContextScope, EmbeddingConfig, GraphConfig, IndexConfig, MemoryCategory, RetrievalConfig, SearchQuery, SessionConfig
from agentbase_core.embedding.litellm import LiteLLMEmbedder
from agentbase_core.ingester.ner_extractor import NerExtractor
from agentbase_core.retrieval.query_decompose import LocalQueryDecomposer

# Shared instances (zero LLM; regex/spaCy)
_NER_EXTRACTOR = NerExtractor(max_entities=15)
_QUERY_DECOMPOSER = LocalQueryDecomposer()


# ---------------------------------------------------------------------------
# BatchEmbedder — precomputes all embeddings in one API call (Optimization B)
# ---------------------------------------------------------------------------


class BatchEmbedder:
    """Wrapper that precomputes embeddings in batch before ingestion.

    Collects texts via precompute() which calls embed_batch() ONCE,
    then subsequent embed() calls return cached results instantly.
    Reduces N embedding API calls to 1 per session (or 1 per entire sample).
    """

    def __init__(self, real_embedder: LiteLLMEmbedder) -> None:
        self._real = real_embedder
        self._cache: dict[str, list[float]] = {}

    async def precompute(self, texts: list[str]) -> list[list[float]]:
        """Precompute embeddings for all texts in batched calls (≤10 per batch).

        DashScope text-embedding-v4 limits batch size to 10.
        """
        if not texts:
            return []
        # Deduplicate: same text → same embedding
        unique_texts = list(dict.fromkeys(texts))
        # Split into batches of 10 (DashScope limit)
        batch_size = 10
        all_embeddings: list[list[float]] = []
        for i in range(0, len(unique_texts), batch_size):
            chunk = unique_texts[i : i + batch_size]
            chunk_embs = await self._real.embed_batch(chunk)
            all_embeddings.extend(chunk_embs)
        for text, emb in zip(unique_texts, all_embeddings):
            self._cache[text] = emb
        # Map back to original order (with duplicates)
        result = [self._cache[t] for t in texts]
        return result

    async def embed(self, text: str) -> list[float]:
        """Return cached embedding if precomputed, else call real embedder."""
        if text in self._cache:
            return self._cache[text]
        # Fallback: single embedding call (should not happen if precompute covers all)
        return await self._real.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — first check cache, then fall back to real embedder."""
        uncached = []
        uncached_idxs = []
        results = [None] * len(texts)
        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached.append(t)
                uncached_idxs.append(i)
        if uncached:
            new_embs = await self._real.embed_batch(uncached)
            for idx, emb in zip(uncached_idxs, new_embs):
                self._cache[texts[idx]] = emb
                results[idx] = emb
        return results

    @property
    def dimensions(self) -> int:
        return self._real.dimensions

    @property
    def model_name(self) -> str:
        return self._real.model_name


# ---------------------------------------------------------------------------
# Zero-LLM Fact Extraction — bilingual regex patterns (from LongMemEval)
# ---------------------------------------------------------------------------

import re as _re

# Each tuple: (compiled_pattern, category)
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

# --- NEW: Temporal / quantity patterns (strengthen temporal & multi-hop recall) ---
_FACT_PATTERNS_TEMPORAL = [
    # Absolute month [+ year]: "May 2022", "in April", "on March 15"
    _re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{1,2})?(?:\s*,?\s*\d{4})?\b", _re.IGNORECASE),
    # Seasons: "summer of 2021", "last winter"
    _re.compile(r"\b(?:spring|summer|fall|autumn|winter)(?:\s+of)?(?:\s+\d{4})?\b", _re.IGNORECASE),
    # Years: 2019-2029
    _re.compile(r"\b(20[12]\d)\b"),
    # Relative: "3 weeks ago", "two months later"
    _re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\s+(?:ago|later|before|after)\b", _re.IGNORECASE),
    # Weekdays
    _re.compile(r"\b(?:last|next|this)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", _re.IGNORECASE),
    # Generic time adverbs
    _re.compile(r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year|weekend)|next\s+(?:week|month|year|weekend)|this\s+(?:week|month|year|weekend))\b", _re.IGNORECASE),
]

_FACT_PATTERNS_QUANTITY = [
    # Numeric quantity + unit (hours/days/dollars/miles/people/...)
    _re.compile(r"\b(\d+(?:\.\d+)?)\s+(?:hours?|days?|weeks?|months?|years?|times?|people|persons?|friends?|kids?|children|dollars?|usd|miles?|km|kg|lbs?|pounds?|gallons?|litres?|minutes?|seconds?)\b", _re.IGNORECASE),
    # Ordinal events: "the first time I ..."
    _re.compile(r"\b(?:the\s+)?(?:first|second|third|last|latest|earliest)\s+time\s+(?:i|we|he|she|they)\s+(.+?)(?:[.,!?]|$)", _re.IGNORECASE),
]


def _local_session_digest(
    turns: list[dict],
    date: str | None,
    session_index: int,
    *,
    speaker_a: str | None = None,
    speaker_b: str | None = None,
    facts: list[dict] | None = None,
) -> str | None:
    """Zero-LLM session digest (enriched).

    Includes:
    - Session header (index, date, turn count)
    - Participants (both speakers)
    - Top 3 user utterances + last assistant reply
    - Entity / fact-type distribution summary
    The header + participant + date line becomes searchable via FTS.
    """
    user_utterances = [
        t["content"].strip()
        for t in turns
        if t.get("role") == "user" and t.get("content", "").strip()
    ]
    assistant_utterances = [
        t["content"].strip()
        for t in turns
        if t.get("role") == "assistant" and t.get("content", "").strip()
    ]
    if not user_utterances and not assistant_utterances:
        return None

    date_str = date or "unknown"
    header = f"[Session {session_index} | Date: {date_str} | Turns: {len(turns)}]"
    lines: list[str] = [header]

    # Participants line (searchable by speaker name)
    if speaker_a or speaker_b:
        parts = [s for s in (speaker_a, speaker_b) if s]
        lines.append(f"Participants: {', '.join(parts)}")

    # Fact distribution (gives LLM a quick topic sketch)
    if facts:
        dist: dict[str, int] = {}
        for f in facts:
            ft = str(f.get("fact_type", "other"))
            dist[ft] = dist.get(ft, 0) + 1
        if dist:
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            lines.append(f"Fact distribution: {dist_str}")

    # Top 3 user utterances
    for utt in user_utterances[:3]:
        lines.append(f"- [user] {utt[:200]}")

    # Last assistant reply (closure of the session)
    if assistant_utterances:
        lines.append(f"- [assistant] {assistant_utterances[-1][:200]}")

    return "\n".join(lines)


def _local_extract_facts(turns: list[dict], date: str | None, session_index: int) -> list[dict]:
    """Zero-LLM fact extraction using bilingual regex patterns + NER.

    Extracts structured factual statements from turns via:
    1. Preference / Entity / Event regex (bilingual)
    2. Temporal regex (month/season/year/relative time)
    3. Quantity regex (numbers + units)
    4. NerExtractor (spaCy or regex fallback)
    Returns up to 25 facts per session (up from 8).
    """
    facts: list[dict] = []

    for turn in turns:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "").strip()
        if not content:
            continue

        # --- Preference patterns ---
        for pat in _FACT_PATTERNS_PREF:
            for m in pat.finditer(content if pat.flags & _re.IGNORECASE else content):
                span_text = content[m.start():min(m.end() + 20, len(content))].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.PREFERENCE,
                    "tags": ["preference", "fact"],
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
                    "tags": ["entity", "fact"],
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
                    "tags": ["event", "fact"],
                    "confidence": 0.8,
                    "fact_type": "event",
                })

        # --- NEW: Temporal patterns (date/time anchors) ---
        for pat in _FACT_PATTERNS_TEMPORAL:
            for m in pat.finditer(content):
                # Keep a wider context for temporal anchors (+40 chars either side)
                start = max(0, m.start() - 20)
                end = min(len(content), m.end() + 40)
                span_text = content[start:end].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.EVENT,
                    "tags": ["temporal", "fact"],
                    "confidence": 0.75,
                    "fact_type": "temporal",
                })

        # --- NEW: Quantity patterns (how-many anchors) ---
        for pat in _FACT_PATTERNS_QUANTITY:
            for m in pat.finditer(content):
                start = max(0, m.start() - 20)
                end = min(len(content), m.end() + 30)
                span_text = content[start:end].strip().rstrip(".,;，。；")
                if len(span_text) < 5:
                    continue
                facts.append({
                    "content": span_text,
                    "category": MemoryCategory.EVENT,
                    "tags": ["quantity", "fact"],
                    "confidence": 0.75,
                    "fact_type": "quantity",
                })

    # --- NEW: NER entities aggregated from all user turns ---
    # Generates entity-name-specific tags (ner_<name>) so that the
    # RetrievalEngine._ner_boost_search can match query entities to stored
    # entity entries and apply score boosts at retrieval time.
    try:
        user_text = " \n ".join(
            t.get("content", "") for t in turns if t.get("role") == "user"
        )
        if user_text.strip():
            ner_items = _NER_EXTRACTOR.extract(user_text)
            for ent in ner_items:
                name = (ent.get("content") or "").strip()
                if len(name) < 2:
                    continue
                cat = MemoryCategory.EVENT if ent.get("category") == "event" else MemoryCategory.ENTITY
                # Build entity-name tag matching RetrievalEngine._ner_boost_search format
                entity_tag = "ner_" + "_".join(
                    ch if (ch.isalnum() or "\u4e00" <= ch <= "\u9fff") else "_"
                    for ch in name
                ).strip("_").lower()
                tags = ["ner", "fact", entity_tag] + list(ent.get("tags", []))
                facts.append({
                    "content": f"Entity mentioned: {name}",
                    "category": cat,
                    "tags": tags,
                    "confidence": float(ent.get("confidence", 0.7)),
                    "fact_type": "ner",
                })
    except Exception:
        # NER failure must not break ingest
        pass

    # Deduplicate by content prefix (lower-cased, 80 chars)
    seen: set[str] = set()
    unique: list[dict] = []
    for f in facts:
        key = f["content"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:25]


# ---------------------------------------------------------------------------
# LLM helper — DashScope native SDK (AioMultiModalConversation)
# qwen3.6-plus / qwen3.6-flash MUST go through MultiModalConversation
# endpoint; AioGeneration.call returns 400 "url error" for these models.
# ---------------------------------------------------------------------------

_DASHSCOPE_API_KEY: str | None = None


def _get_dashscope_api_key() -> str:
    global _DASHSCOPE_API_KEY
    if _DASHSCOPE_API_KEY is None:
        _DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    return _DASHSCOPE_API_KEY


async def call_llm(
    prompt: str, model: str | None = None, max_tokens: int = 1024
) -> str:
    """Call DashScope LLM via native SDK (AioMultiModalConversation).

    qwen3.6-plus / qwen3.6-flash require the MultiModalConversation
    endpoint; enable_thinking=False is passed for qwen3 series.
    Includes request_timeout (120s) and asyncio.wait_for (150s) to
    prevent indefinite hangs on large prompts.
    """
    import dashscope
    import backoff

    effective_model = model or _READER_MODEL
    api_key = _get_dashscope_api_key()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=90)
    async def _call():
        kwargs: dict[str, Any] = dict(
            api_key=api_key,
            model=effective_model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            temperature=0,
            max_tokens=max_tokens,
            request_timeout=120,
        )
        # qwen3 / qwen3.6 series require enable_thinking=False
        m_lower = effective_model.lower()
        if "qwen3" in m_lower or "qwen-3" in m_lower:
            kwargs["enable_thinking"] = False

        response = await dashscope.AioMultiModalConversation.call(**kwargs)

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope error {response.status_code}: "
                f"{getattr(response, 'code', '?')} {getattr(response, 'message', '')}"
            )
        content = response.output.choices[0].message.content[0]["text"]
        return (content or "").strip()

    # Outer timeout guard — if backoff retries exhaust or the SDK
    # internal timeout fails to fire, asyncio.wait_for ensures we
    # never hang forever.
    try:
        return await asyncio.wait_for(_call(), timeout=150)
    except asyncio.TimeoutError:
        logger.warning(f"call_llm timed out (150s) for model={effective_model}")
        return ""


# ---------------------------------------------------------------------------
# Data loading — parse LoCoMo JSON format
# ---------------------------------------------------------------------------

QA_CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def load_locomo(data_file: str | Path) -> list[dict]:
    """Load LoCoMo dataset and return list of parsed samples.

    Each sample dict has:
      - sample_id: str
      - speaker_a, speaker_b: str
      - sessions: list[{session_id, date_time, turns: [{speaker, text}]}]
      - qa: list[{question, answer, category, evidence, adversarial_answer}]
      - event_summary: dict  (session -> speaker -> events list)
      - session_summary: dict (session -> summary text)
    """
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for idx, raw in enumerate(raw_data):
        # --- Parse conversation ---
        conv = raw["conversation"]
        speaker_a = conv.get("speaker_a", "SpeakerA")
        speaker_b = conv.get("speaker_b", "SpeakerB")

        sessions = []
        # Find all session keys (session_1, session_2, ...)
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),
        )

        for skey in session_keys:
            session_id = int(skey.split("_")[1])
            date_time = conv.get(f"{skey}_date_time", "")
            raw_turns = conv[skey]
            if not isinstance(raw_turns, list):
                continue

            turns = []
            for turn in raw_turns:
                text = turn.get("text", "")
                # Handle image content: use BLIP caption as text substitute
                if "img_url" in turn and "blip_caption" in turn:
                    caption = f"[Image: {turn['blip_caption']}]"
                    text = f"{caption} {text}".strip() if text else caption
                if text.strip():
                    turns.append({
                        "speaker": turn.get("speaker", ""),
                        "text": text,
                    })

            if turns:
                sessions.append({
                    "session_id": session_id,
                    "date_time": date_time,
                    "turns": turns,
                })

        # --- Parse QA ---
        qa_list = []
        for qa in raw.get("qa", []):
            qa_list.append({
                "question": qa["question"],
                "answer": qa.get("answer", ""),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
                "adversarial_answer": qa.get("adversarial_answer"),
            })

        # --- Parse event_summary ---
        event_summary = raw.get("event_summary", {})

        # --- Parse session_summary ---
        session_summary = raw.get("session_summary", {})

        samples.append({
            "sample_id": str(idx),
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "sessions": sessions,
            "qa": qa_list,
            "event_summary": event_summary,
            "session_summary": session_summary,
        })

    return samples


# ---------------------------------------------------------------------------
# Memory ingestion — Zero-LLM: direct ContextEntry construction + batch insert
# (Ported from LongMemEval for maximum ingest speed)
# ---------------------------------------------------------------------------


async def ingest_conversation(
    db: AgentBase, sample: dict, batch_embedder: "BatchEmbedder | None" = None
) -> None:
    """Zero-LLM ingest: build ContextEntry objects directly, batch-insert.

    Replaces add_conversation()/add_memory() with direct store+index calls.
    No LLM calls during ingest — uses local regex facts + truncation digests.
    """
    from agentbase_core.models import ContextEntry, ContextType, OriginType
    from agentbase_core.index.tokenizer import tokenize_text
    from datetime import datetime, timezone

    speaker_a = sample["speaker_a"]
    speaker_b = sample["speaker_b"]

    # --- Parse all session datetimes upfront ---
    session_dts: list[datetime | None] = []
    for session in sample["sessions"]:
        date = session["date_time"]
        session_dt = None
        if date:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
                try:
                    session_dt = datetime.strptime(date.strip(), fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
        session_dts.append(session_dt)

    # --- Phase 1: Build all ContextEntry objects + collect texts for embedding ---
    all_entries: list[ContextEntry] = []
    all_texts_for_embed: list[str] = []
    sample_id = sample["sample_id"]

    # Auto-classify preference indicators (from LongMemEval)
    _PREF_INDICATORS = [
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
    ]

    for i, (session, session_dt) in enumerate(zip(sample["sessions"], session_dts)):
        session_id = session["session_id"]
        date = session["date_time"]
        raw_turns = session["turns"]

        # Derive date anchors for FTS enrichment
        date_anchor = (date or "").strip()
        year_anchor = ""
        month_anchor = ""
        if session_dt is not None:
            year_anchor = str(session_dt.year)
            month_anchor = session_dt.strftime("%B")  # e.g. "May"

        # Build mapped turns list
        turns: list[dict] = []
        for turn in raw_turns:
            speaker = turn["speaker"]
            text = turn["text"]
            if speaker == speaker_a:
                role = "user"
            elif speaker == speaker_b:
                role = "assistant"
            else:
                role = "user"
            turns.append({"role": role, "content": text, "speaker": speaker})

        if not turns:
            continue

        # ---- Turn-level entries ----
        for j, turn in enumerate(turns):
            content = turn["content"].strip()
            if not content:
                continue

            # Auto-classify category
            content_lower = content.lower()
            if turn["role"] == "user" and any(ind in content_lower for ind in _PREF_INDICATORS):
                category = MemoryCategory.PREFERENCE
            elif turn["role"] == "user":
                category = MemoryCategory.EVENT
            else:
                category = MemoryCategory.ENTITY

            speaker_name = turn.get("speaker") or (speaker_a if turn["role"] == "user" else speaker_b)
            turn_content = f"[{turn['role']}]: {content}"

            # Enriched FTS text: speaker + date + year + month prepended so
            # FTS/BM25 can hit the turn by temporal or speaker queries.
            fts_parts = [
                f"speaker:{speaker_name}" if speaker_name else "",
                f"date:{date_anchor}" if date_anchor else "",
                f"year:{year_anchor}" if year_anchor else "",
                f"month:{month_anchor}" if month_anchor else "",
                f"session:{session_id}",
                turn_content,
            ]
            fts_enriched = " \n ".join(p for p in fts_parts if p)

            tag_list = [
                "locomo", f"sample_{sample_id}",
                f"session_{session_id}", f"turn_{j}", turn["role"],
            ]
            if speaker_name:
                tag_list.append(f"speaker:{speaker_name}")
            if date_anchor:
                tag_list.append(f"date:{date_anchor}")
            if year_anchor:
                tag_list.append(f"year:{year_anchor}")

            entry = ContextEntry(
                l2_full=turn_content,
                context_type=ContextType.MEMORY,
                memory_category=category,
                tags=tag_list,
                confidence=0.9,
                scope=ContextScope.GLOBAL,
                source="conversation",
                origin_type=OriginType.MANUAL,
                extra={
                    "session_index": session_id - 1,
                    "session_date": date,
                    "turn_index": j,
                    "role": turn["role"],
                    "speaker": speaker_name,
                },
            )
            if session_dt is not None:
                entry.created_at = session_dt
                entry.valid_from = session_dt

            entry.fts_text = tokenize_text(fts_enriched, tokenizer="icu")
            entry.mark_active()
            all_entries.append(entry)
            all_texts_for_embed.append(turn_content)

        # ---- Local fact entries (zero-LLM regex extraction) ----
        facts = _local_extract_facts(turns, date, session_id - 1)

        # ---- Session digest entry (zero-LLM, replaces add_memory) ----
        digest_text = _local_session_digest(
            turns, date, session_id - 1,
            speaker_a=speaker_a, speaker_b=speaker_b, facts=facts,
        )
        if digest_text:
            full_transcript = "\n".join(
                f"[{t['role']}] {t.get('speaker') or ''}: {t['content']}" for t in turns
            )
            summary_content = (
                f"[Full conversation of Session {session_id} | Date: {date} | "
                f"Participants: {speaker_a}, {speaker_b}]\n"
                f"{full_transcript}"
            )
            summary_fts_parts = [
                f"speaker:{speaker_a}" if speaker_a else "",
                f"speaker:{speaker_b}" if speaker_b else "",
                f"date:{date_anchor}" if date_anchor else "",
                f"year:{year_anchor}" if year_anchor else "",
                f"month:{month_anchor}" if month_anchor else "",
                f"session:{session_id}",
                digest_text,
                summary_content,
            ]
            summary_fts_enriched = " \n ".join(p for p in summary_fts_parts if p)

            summary_tags = [
                "locomo", f"sample_{sample_id}",
                f"session_{session_id}", "session_summary",
            ]
            if speaker_a:
                summary_tags.append(f"speaker:{speaker_a}")
            if speaker_b:
                summary_tags.append(f"speaker:{speaker_b}")
            if date_anchor:
                summary_tags.append(f"date:{date_anchor}")
            if year_anchor:
                summary_tags.append(f"year:{year_anchor}")

            summary_entry = ContextEntry(
                l0_abstract=digest_text,
                l1_overview=digest_text,
                l2_full=summary_content,
                context_type=ContextType.MEMORY,
                memory_category=MemoryCategory.EVENT,
                tags=summary_tags,
                confidence=0.95,
                scope=ContextScope.GLOBAL,
                source="conversation",
                origin_type=OriginType.MANUAL,
                extra={
                    "session_index": session_id - 1,
                    "session_date": date,
                    "role": "summary",
                    "num_turns": len(turns),
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            )
            if session_dt is not None:
                summary_entry.created_at = session_dt
                summary_entry.valid_from = session_dt

            summary_entry.fts_text = tokenize_text(summary_fts_enriched, tokenizer="icu")
            summary_entry.mark_active()
            all_entries.append(summary_entry)
            all_texts_for_embed.append(summary_content)

        # ---- Fact entries (emit after digest to keep facts list available) ----
        for fact in facts:
            fact_content = fact["content"]
            fact_fts_parts = [
                f"date:{date_anchor}" if date_anchor else "",
                f"year:{year_anchor}" if year_anchor else "",
                f"month:{month_anchor}" if month_anchor else "",
                f"session:{session_id}",
                fact_content,
            ]
            fact_fts_enriched = " \n ".join(p for p in fact_fts_parts if p)

            fact_tags = [
                "locomo", f"sample_{sample_id}",
                f"session_{session_id}", "fact",
            ] + list(fact.get("tags", []))
            if date_anchor:
                fact_tags.append(f"date:{date_anchor}")
            if year_anchor:
                fact_tags.append(f"year:{year_anchor}")

            fact_entry = ContextEntry(
                l0_abstract=fact_content,
                l1_overview=fact_content,
                l2_full=fact_content,
                context_type=ContextType.MEMORY,
                memory_category=fact["category"],
                tags=fact_tags,
                confidence=fact.get("confidence", 0.8),
                scope=ContextScope.GLOBAL,
                source="conversation",
                origin_type=OriginType.MANUAL,
                extra={
                    "session_index": session_id - 1,
                    "session_date": date,
                    "role": "fact",
                    "fact_type": fact.get("fact_type", "statement"),
                },
            )
            if session_dt is not None:
                fact_entry.created_at = session_dt
                fact_entry.valid_from = session_dt

            fact_entry.fts_text = tokenize_text(fact_fts_enriched, tokenizer="icu")
            fact_entry.mark_active()
            all_entries.append(fact_entry)
            all_texts_for_embed.append(fact_content)

    # --- Phase 2: Precompute embeddings in one batch (Optimization B) ---
    if batch_embedder and all_texts_for_embed:
        t0 = time.time()
        await batch_embedder.precompute(all_texts_for_embed)
        elapsed = time.time() - t0
        print(f"  Batch embedding: {len(all_texts_for_embed)} texts in {elapsed:.1f}s")

    # --- Phase 3: Store embeddings in cache + batch insert into store/index ---
    if all_entries:
        store = db._engine._store
        index = db._engine._index

        # Batch add to store (SQLite INSERT) — must happen first so vec_meta rows exist
        for entry in all_entries:
            await store.add(entry)

        # Store embeddings in embedding_cache so SQLiteVecIndex can find them
        if batch_embedder and all_texts_for_embed:
            from agentbase_core.ingester.dedup import EmbeddingCache
            emb_cache = EmbeddingCache(store._pool)
            embed_model = getattr(batch_embedder._real, "model_name", "text-embedding-v4")
            embed_dim = getattr(batch_embedder._real, "dimensions", 1024)
            # Deduplicate entries by embed text to avoid duplicate cache writes
            seen_hashes: set[str] = set()
            for entry, embed_text in zip(all_entries, all_texts_for_embed):
                if embed_text not in batch_embedder._cache:
                    continue
                content_hash = EmbeddingCache.compute_hash(embed_text)
                entry.embedding_hash = content_hash
                entry.embedding_model = embed_model
                entry.embedding_dimensions = embed_dim
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    await emb_cache.put(
                        content_hash,
                        batch_embedder._cache[embed_text],
                        model=embed_model,
                        dimensions=embed_dim,
                    )
            # Update entries with embedding metadata
            for entry in all_entries:
                if entry.embedding_hash:
                    await store.update(entry)

        # Batch add to FTS + vec index (vec reads embeddings from cache)
        try:
            await index.add_batch(all_entries)
        except Exception:
            for entry in all_entries:
                try:
                    await index.add(entry)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# QA Task — Single-Stage Retrieve-then-Read (no evidence extraction)
# ---------------------------------------------------------------------------

# Answer prompts — use {context} (retrieved conversation history)

ANSWER_PROMPT_SINGLE_HOP = """Answer the question using ONLY the conversation history below. Do NOT use any information outside the history.

Conversation History:
{context}

Question: {question}

CRITICAL: Answer with ONLY the minimal words needed. No explanations, no full sentences.
Examples: sushi | Central Park | Nike | Sarah
If the history doesn't contain the answer, say "I don't have that information."

Answer:"""

ANSWER_PROMPT_MULTI_HOP = """Answer the question by connecting pieces of information from the conversation history.

IMPORTANT RULES:
- Each passage is labeled with its session number and date.
- You MUST search through ALL history exhaustively. Do not stop after finding one mention.
- When asked "how many" or "how much", carefully enumerate EACH instance from EACH session.
- Combine facts from different pieces of evidence into one short answer.

Conversation History:
{context}

Question: {question}

CRITICAL: Keep it under 20 words. No explanations.
Examples: yoga and hiking | a Canon camera and a travel journal
If the history doesn't contain the answer, say "I don't have that information."

Answer:"""

ANSWER_PROMPT_TEMPORAL = """Answer the temporal question using the conversation history.

IMPORTANT RULES:
- Each passage is labeled with its session date. Use these dates for ALL time calculations.
- When asked "how many days/weeks/months between X and Y", calculate from the session dates.
- When asked "what happened first/last", compare session dates to determine order.
- When asked "how many times", carefully count each distinct occurrence.
- For "when was the first/last time", identify the earliest/latest session date.
- Show your work: mention the relevant session dates you used.

Conversation History:
{context}

Question: {question}

CRITICAL: Answer with ONLY the date, number, or minimal phrase. No explanations.
Examples: 14 days | the hiking trip | March 15, 2024 | 3
If the history doesn't contain the answer, say "I don't have that information."

Answer:"""

ANSWER_PROMPT_OPEN_DOMAIN = """Answer the question using the conversation history and your world knowledge.

Conversation History:
{context}

Question: {question}

CRITICAL: Answer with ONLY the minimal words needed. No explanations.
- Use evidence first, then apply commonsense/world knowledge.
Examples: Japan | French | vitamin C
If the history doesn't contain the answer, say "I don't have that information."

Answer:"""

ANSWER_PROMPT_ADVERSARIAL = """Answer the question carefully. The question may contain traps.

Conversation History:
{context}

Question: {question}

RULES:
- Verify WHO actually said/did what — the question may swap speakers. The history below is split by speaker; check BOTH speakers before answering.
- Verify WHAT actually happened — the question may assume something that didn't occur.
- If the question's premise is wrong (e.g. wrong speaker, wrong event, wrong date), respond with the correct fact and explicitly reject the false premise.
- If the information is missing for the specified person/event, answer "I don't have that information." rather than guessing.
- Do NOT fabricate or carry over a fact from the other speaker.

CRITICAL: Keep answer under 20 words. State only the corrected fact or the denial.
Examples: John did not quit his job | It was Tom, not Mary | They did not cancel | Not mentioned for Sarah
If the history doesn't contain the answer, say "I don't have that information."

Answer:"""

# Map QA category to prompt and retrieval config
QA_CATEGORY_CONFIG = {
    1: {  # single-hop
        "prompt": ANSWER_PROMPT_SINGLE_HOP,
        "query_type": None,
        "top_k": 60,
        "token_budget": 12000,
        "max_tokens": 1024,
    },
    2: {  # multi-hop
        "prompt": ANSWER_PROMPT_MULTI_HOP,
        "query_type": "multi-session",
        "top_k": 100,
        "token_budget": 16000,
        "max_tokens": 1024,
    },
    3: {  # temporal
        "prompt": ANSWER_PROMPT_TEMPORAL,
        "query_type": "temporal-reasoning",
        "top_k": 100,
        "token_budget": 16000,
        "max_tokens": 1024,
    },
    4: {  # open-domain
        "prompt": ANSWER_PROMPT_OPEN_DOMAIN,
        "query_type": None,
        "top_k": 60,
        "token_budget": 12000,
        "max_tokens": 1024,
    },
    5: {  # adversarial
        "prompt": ANSWER_PROMPT_ADVERSARIAL,
        "query_type": "multi-session",
        "top_k": 80,
        "token_budget": 14000,
        "max_tokens": 1024,
        "adversarial": True,
    },
}


# --- Extended temporal tokens (supplementing LocalQueryDecomposer) ---
_RQ_TEMPORAL_EXTRA = [
    # "May 2022", "April of 2023"
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+of)?\s+\d{4}\b", re.IGNORECASE),
    # "summer 2021", "winter of 2020"
    re.compile(r"\b(?:spring|summer|fall|autumn|winter)(?:\s+of)?\s+\d{4}\b", re.IGNORECASE),
    # Standalone year 2019-2029
    re.compile(r"\b(20[12]\d)\b"),
    # "3 weeks ago", "last month", "two years later"
    re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:day|week|month|year)s?\s+(?:ago|later|before|after)\b", re.IGNORECASE),
    re.compile(r"\b(?:last|next|this)\s+(?:week|month|year|weekend)\b", re.IGNORECASE),
]


def _extract_extra_temporal_tokens(question: str) -> list[str]:
    tokens: list[str] = []
    for pat in _RQ_TEMPORAL_EXTRA:
        for m in pat.finditer(question):
            tok = m.group(0).strip()
            if tok and tok.lower() not in {t.lower() for t in tokens}:
                tokens.append(tok)
    return tokens


def _decompose_query(question: str, query_type: str | None = None) -> list[str]:
    """Decompose a complex query into sub-queries for broader recall.

    Layers:
    1. Core LocalQueryDecomposer (rule-based, with temporal tokens when
       query_type is temporal-reasoning / multi-session).
    2. Extended temporal tokens (May 2022 / summer 2021 / standalone year).
    3. Bilingual pattern rules from the original local version (between /
       first-or-last / how-many / the day I X).
    Cap: 7 sub-queries (was 5).
    """
    queries: list[str] = [question]
    q_lower = question.lower()

    # ----- Layer 1: core LocalQueryDecomposer -----
    try:
        core_subs = _QUERY_DECOMPOSER.decompose(question, query_type=query_type)
        for sq in core_subs:
            if sq and sq.lower().strip() not in {q.lower().strip() for q in queries}:
                queries.append(sq)
    except Exception:
        pass

    # ----- Layer 2: extended temporal tokens -----
    extra_tokens = _extract_extra_temporal_tokens(question)
    for tok in extra_tokens:
        if tok.lower() not in {q.lower() for q in queries}:
            queries.append(tok)

    # ----- Layer 3: existing bilingual rules -----
    between_match = re.search(
        r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\.)", question, re.IGNORECASE
    )
    if between_match:
        queries.append(between_match.group(1).strip())
        queries.append(between_match.group(2).strip())

    order_match = re.search(
        r"(?:first|last|order).+?(\w+(?:\s+\w+){0,3})\s+or\s+(\w+(?:\s+\w+){0,3})",
        question,
        re.IGNORECASE,
    )
    if order_match:
        queries.append(order_match.group(1).strip())
        queries.append(order_match.group(2).strip())

    how_match = re.search(
        r"how (?:many|much|often|long|far).+?(?:did|have|was|were|do|does|has|is|are)\s+(.+?)(?:\?|$|\.)",
        question,
        re.IGNORECASE,
    )
    if how_match:
        entity = how_match.group(1).strip()
        if len(entity) > 2:
            queries.append(entity)

    when_match = re.search(
        r"(?:the day|the time|when) (?:I|you|we) (.+?)(?:\?|$|,|\.)",
        question,
        re.IGNORECASE,
    )
    if when_match:
        queries.append(when_match.group(1).strip())

    if "how many" in q_lower or "how much" in q_lower:
        words = re.findall(r"\b([a-z]{3,}(?:s|es|ies)?)\b", q_lower)
        stop_words = {
            "the", "and", "was", "were", "did", "have", "has", "had",
            "been", "how", "many", "much", "does", "that", "this",
            "from", "with", "for", "not", "but", "are", "its",
            "all", "total", "overall", "different", "various",
        }
        keywords = [w for w in words if w not in stop_words]
        if keywords:
            queries.append(keywords[-1])
            if len(keywords) >= 2:
                queries.append(f"{keywords[-2]} {keywords[-1]}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_stripped = (q or "").strip().rstrip("?.!")
        if q_stripped and q_stripped.lower() not in seen:
            seen.add(q_stripped.lower())
            unique.append(q_stripped)

    return unique[:7]


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_SEASON_MONTHS = {
    "spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2),
}


def _extract_date_range(question: str) -> tuple["datetime | None", "datetime | None"]:
    """Extract an absolute [date_from, date_to] range from a temporal question.

    Currently handles:
    - "<Month> <Year>" → that month
    - "<Season> <Year>" / "<Season> of <Year>" → that season's months
    - Standalone "<Year>" → whole year (Jan 1 – Dec 31)
    Returns (None, None) if nothing matches.
    """
    from datetime import datetime, timezone
    import calendar

    q = question.lower()

    # "May 2022" / "April of 2023"
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:of\s+)?(\d{4})\b",
        q,
    )
    if m:
        mo = _MONTH_NAMES[m.group(1)]
        yr = int(m.group(2))
        last_day = calendar.monthrange(yr, mo)[1]
        return (
            datetime(yr, mo, 1, tzinfo=timezone.utc),
            datetime(yr, mo, last_day, 23, 59, 59, tzinfo=timezone.utc),
        )

    # "summer 2021" / "winter of 2020"
    m = re.search(r"\b(spring|summer|fall|autumn|winter)\s+(?:of\s+)?(\d{4})\b", q)
    if m:
        season = m.group(1)
        yr = int(m.group(2))
        mo_from, mo_to = _SEASON_MONTHS[season]
        if mo_from <= mo_to:
            from_dt = datetime(yr, mo_from, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(yr, mo_to)[1]
            to_dt = datetime(yr, mo_to, last_day, 23, 59, 59, tzinfo=timezone.utc)
        else:
            # winter spans Dec (yr) → Feb (yr+1)
            from_dt = datetime(yr, mo_from, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(yr + 1, mo_to)[1]
            to_dt = datetime(yr + 1, mo_to, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return from_dt, to_dt

    # Standalone "<Year>"
    m = re.search(r"\b(20[12]\d)\b", q)
    if m:
        yr = int(m.group(1))
        return (
            datetime(yr, 1, 1, tzinfo=timezone.utc),
            datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

    return None, None


def _extract_mentioned_speakers(question: str, known_speakers: tuple[str, ...]) -> list[str]:
    """Return the subset of known_speakers that appear (case-insensitive) in the question."""
    q_lower = question.lower()
    hits: list[str] = []
    for name in known_speakers:
        if not name:
            continue
        if name.lower() in q_lower and name not in hits:
            hits.append(name)
    return hits


def _parse_locomo_date(date_str: str) -> "datetime | None":
    """Parse a LoCoMo date string (e.g. '2022/05/14 (Sat) 10:00') into datetime."""
    from datetime import datetime, timezone
    if not date_str or not date_str.strip():
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _assemble_es_context(
    results: list,
    speaker: str | None = None,
    max_session_index: int | None = None,
) -> str:
    """Assemble ES-specific context with speaker filtering and time-window truncation.

    E2 optimization: filters results to only include the target speaker's turns
    and entries from sessions up to max_session_index. This reduces noise and
    focuses the LLM on relevant events.
    """
    # Separate by tag type
    summary_results = []
    fact_results = []
    turn_results = []
    for r in results:
        tags = r.entry.tags or []
        extra = r.entry.extra or {}
        si = extra.get("session_index", 0)

        # Time-window truncation: only include sessions up to max_session_index
        if max_session_index is not None and si > max_session_index:
            continue

        if "session_summary" in tags:
            summary_results.append(r)
        elif "fact" in tags:
            # For facts, only keep those related to the speaker
            if speaker:
                fact_content = (r.entry.l0_abstract or r.entry.l2_full or "").lower()
                if speaker.lower() not in fact_content:
                    continue
            fact_results.append(r)
        else:
            # For turns, only keep those by/from the target speaker
            if speaker:
                entry_speaker = (extra.get("speaker") or "").strip().lower()
                entry_role = (extra.get("role") or "").strip().lower()
                # Include turns where the speaker is the speaker OR
                # turns that mention the speaker (for context)
                if entry_speaker == speaker.lower():
                    turn_results.append(r)
                elif speaker.lower() in (r.entry.l2_full or "").lower():
                    turn_results.append(r)
            else:
                turn_results.append(r)

    context_parts: list[str] = []

    # --- Layer 1: Session summaries ---
    if summary_results:
        context_parts.append("=== SESSION SUMMARIES ===")
        summary_results.sort(
            key=lambda r: (r.entry.extra or {}).get("session_index", 0)
        )
        for r in summary_results:
            entry = r.entry
            extra = entry.extra or {}
            session_idx = extra.get("session_index", "")
            session_date = extra.get("session_date", "")
            summary_text = entry.l0_abstract or entry.l1_overview or entry.l2_full or ""
            header_parts = []
            if session_idx != "":
                header_parts.append(f"Session {session_idx}")
            if session_date:
                header_parts.append(f"Date: {session_date}")
            header = " | ".join(header_parts)
            context_parts.append(f"[{header} — Summary]\n{summary_text}")

    # --- Layer 2: Facts related to speaker ---
    if fact_results:
        context_parts.append("=== KEY FACTS ===")
        for r in fact_results:
            extra = r.entry.extra or {}
            si = extra.get("session_index", 0)
            session_date = extra.get("session_date", "")
            fact_content = r.entry.l0_abstract or r.entry.l2_full or ""
            if fact_content.strip():
                header = f"[Session {si} | Date: {session_date}]"
                context_parts.append(f"{header}\n  - {fact_content.strip()}")

    # --- Layer 3: Speaker's turns + mentions ---
    if turn_results:
        context_parts.append(f"=== TURNS INVOLVING {speaker or 'USER'} ===")
        def _sort_key(r):
            extra = r.entry.extra or {}
            return (extra.get("session_index", 0), extra.get("turn_index", 0))
        sorted_turns = sorted(turn_results, key=_sort_key)

        current_session = None
        current_date = ""
        current_lines: list[str] = []

        for r in sorted_turns:
            entry = r.entry
            text = entry.l2_full or entry.l1_overview or entry.l0_abstract or ""
            if not text.strip():
                continue
            extra = entry.extra or {}
            session_idx = extra.get("session_index", "")
            session_date_val = extra.get("session_date", "")

            if session_idx != current_session:
                if current_lines:
                    header = f"[Session {current_session}"
                    if current_date:
                        header += f" | Date: {current_date}"
                    header += "]"
                    context_parts.append(header + "\n" + "\n".join(current_lines))
                current_session = session_idx
                current_date = session_date_val
                current_lines = []
            current_lines.append(text)

        if current_lines:
            header = f"[Session {current_session}"
            if current_date:
                header += f" | Date: {current_date}"
            header += "]"
            context_parts.append(header + "\n" + "\n".join(current_lines))

    return "\n\n---\n\n".join(context_parts)


def _assemble_context(results: list, adversarial_speakers: list[str] | None = None) -> str:
    """Assemble retrieved results into layered context (session→fact→turn).

    When ``adversarial_speakers`` is provided, the detailed-turn layer is
    additionally split by speaker so the LLM can compare them side by side
    (reduces speaker-confusion in adversarial QA).
    """
    # Separate by tag type
    summary_results = []
    fact_results = []
    turn_results = []
    for r in results:
        tags = r.entry.tags or []
        if "session_summary" in tags:
            summary_results.append(r)
        elif "fact" in tags:
            fact_results.append(r)
        else:
            turn_results.append(r)

    context_parts: list[str] = []

    # --- Layer 1: Session summaries (session-level overview) ---
    if summary_results:
        context_parts.append("=== SESSION SUMMARIES ===")
        summary_results.sort(
            key=lambda r: (r.entry.extra or {}).get("session_index", 0)
        )
        for r in summary_results:
            entry = r.entry
            extra = entry.extra or {}
            session_idx = extra.get("session_index", "")
            session_date = extra.get("session_date", "")
            summary_text = entry.l0_abstract or entry.l1_overview or entry.l2_full or ""
            header_parts = []
            if session_idx != "":
                header_parts.append(f"Session {session_idx}")
            if session_date:
                header_parts.append(f"Date: {session_date}")
            header = " | ".join(header_parts)
            context_parts.append(f"[{header} — Summary]\n{summary_text}")

    # --- Layer 2: Extracted facts (precise factual statements) ---
    if fact_results:
        context_parts.append("=== KEY FACTS ===")
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

    # --- Layer 3: Turn-level details ---
    if turn_results:
        if summary_results or fact_results:
            context_parts.append("=== DETAILED TURNS ===")

        def _sort_key(r):
            extra = r.entry.extra or {}
            return (extra.get("session_index", 0), extra.get("turn_index", 0))
        sorted_turns = sorted(turn_results, key=_sort_key)

        # Adversarial mode: group turns by speaker for side-by-side comparison.
        if adversarial_speakers:
            groups: dict[str, list] = {s: [] for s in adversarial_speakers}
            groups["__other__"] = []
            for r in sorted_turns:
                extra = r.entry.extra or {}
                sp = (extra.get("speaker") or "").strip()
                matched = None
                for s in adversarial_speakers:
                    if sp and sp.lower() == s.lower():
                        matched = s
                        break
                if matched:
                    groups[matched].append(r)
                else:
                    groups["__other__"].append(r)

            for spk in adversarial_speakers:
                grp = groups.get(spk) or []
                if not grp:
                    continue
                lines: list[str] = []
                for r in grp:
                    text = r.entry.l2_full or r.entry.l1_overview or r.entry.l0_abstract or ""
                    if not text.strip():
                        continue
                    ex = r.entry.extra or {}
                    s_idx = ex.get("session_index", "")
                    s_date = ex.get("session_date", "")
                    lines.append(f"[Session {s_idx} | Date: {s_date}] {text}")
                if lines:
                    context_parts.append(
                        f"--- Turns by {spk} ---\n" + "\n".join(lines)
                    )

            other = groups.get("__other__") or []
            if other:
                lines = []
                for r in other:
                    text = r.entry.l2_full or r.entry.l1_overview or r.entry.l0_abstract or ""
                    if not text.strip():
                        continue
                    ex = r.entry.extra or {}
                    s_idx = ex.get("session_index", "")
                    s_date = ex.get("session_date", "")
                    lines.append(f"[Session {s_idx} | Date: {s_date}] {text}")
                if lines:
                    context_parts.append(
                        "--- Other turns ---\n" + "\n".join(lines)
                    )
            return "\n\n---\n\n".join(context_parts)

        current_session = None
        current_date = ""
        current_lines: list[str] = []

        for r in sorted_turns:
            entry = r.entry
            text = entry.l2_full or entry.l1_overview or entry.l0_abstract or ""
            if not text.strip():
                continue
            extra = entry.extra or {}
            session_idx = extra.get("session_index", "")
            session_date = extra.get("session_date", "")

            if session_idx != current_session:
                if current_lines:
                    header = f"[Session {current_session}"
                    if current_date:
                        header += f" | Date: {current_date}"
                    header += "]"
                    context_parts.append(header + "\n" + "\n".join(current_lines))
                current_session = session_idx
                current_date = session_date
                current_lines = []
            current_lines.append(text)

        if current_lines:
            header = f"[Session {current_session}"
            if current_date:
                header += f" | Date: {current_date}"
            header += "]"
            context_parts.append(header + "\n" + "\n".join(current_lines))

    return "\n\n---\n\n".join(context_parts)


async def answer_qa(
    db: AgentBase,
    question: str,
    category: int | None = None,
    speakers: tuple[str, str] | None = None,
) -> str:
    """Single-Stage Retrieve-then-Read QA pipeline.

    Stage 1 (Retrieve): Search with query decomposition + type-specific
    filters (date range for temporal, per-speaker shards for adversarial),
    then layered assembly.
    Stage 2 (Answer): Single LLM call from full context (no evidence extraction).
    """
    # Determine config from QA category
    cfg = QA_CATEGORY_CONFIG.get(
        category,
        {
            "prompt": ANSWER_PROMPT_SINGLE_HOP,
            "query_type": None,
            "top_k": 40,
            "token_budget": 12000,
            "max_tokens": 1024,
        },
    )
    answer_prompt = cfg["prompt"]
    query_type = cfg.get("query_type")
    top_k = cfg.get("top_k", 40)
    token_budget = cfg.get("token_budget", 12000)
    max_tokens = cfg.get("max_tokens", 1024)
    is_adversarial = bool(cfg.get("adversarial"))

    sub_queries = _decompose_query(question, query_type=query_type)

    all_results: list = []
    seen_ids: set[str] = set()

    def _absorb(results: list) -> None:
        for r in results:
            if r.entry.id not in seen_ids:
                seen_ids.add(r.entry.id)
                all_results.append(r)

    # ================================================================
    # Stage 1a: Base retrieval (always run baseline db.find)
    # ================================================================
    for sq in sub_queries:
        try:
            results = await db.find(
                query=sq, query_type=query_type,
                top_k=top_k, token_budget=token_budget,
            )
            _absorb(results)
        except Exception as e:
            print(f"    [retrieve] baseline failed for sub-query '{sq[:60]}': {e}")

    # ================================================================
    # Stage 1b: Temporal — add a second pass with [date_from, date_to]
    # ================================================================
    if query_type == "temporal-reasoning":
        date_from, date_to = _extract_date_range(question)
        if date_from is not None or date_to is not None:
            for sq in sub_queries:
                try:
                    results = await db.search(
                        sq,
                        top_k=top_k,
                        query_type=query_type,
                        token_budget=token_budget,
                        date_from=date_from,
                        date_to=date_to,
                    )
                    _absorb(results)
                except Exception as e:
                    print(f"    [retrieve] temporal-filter failed: {e}")

    # ================================================================
    # Stage 1c: Adversarial — add a per-speaker pass for mentioned speakers
    # ================================================================
    mentioned_speakers: list[str] = []
    if is_adversarial and speakers:
        known = tuple(s for s in speakers if s)
        mentioned_speakers = _extract_mentioned_speakers(question, known)
        if mentioned_speakers:
            for spk in mentioned_speakers:
                for sq in sub_queries:
                    try:
                        results = await db.search(
                            sq,
                            top_k=top_k,
                            query_type=query_type,
                            token_budget=token_budget,
                            speaker=spk,
                        )
                        _absorb(results)
                    except Exception as e:
                        print(f"    [retrieve] speaker-filter failed ({spk}): {e}")
        # Always include both known speakers so the other side is visible too.
        extra_speakers = [s for s in known if s not in mentioned_speakers]
        for spk in extra_speakers:
            try:
                results = await db.search(
                    question,
                    top_k=max(20, top_k // 2),
                    query_type=query_type,
                    token_budget=token_budget,
                    speaker=spk,
                )
                _absorb(results)
            except Exception as e:
                print(f"    [retrieve] counter-speaker failed ({spk}): {e}")
        # Ensure mentioned_speakers always has both known speakers for display,
        # so the LLM sees both sides (prevents speaker confusion).
        if speakers:
            mentioned_speakers = list(dict.fromkeys((mentioned_speakers + list(known))))

    if not all_results:
        return "I don't have that information."

    # ================================================================
    # Stage 2: ASSEMBLE layered context
    # ================================================================
    context = _assemble_context(
        all_results,
        adversarial_speakers=mentioned_speakers if is_adversarial else None,
    )
    if not context.strip():
        return "I don't have that information."

    # ================================================================
    # Stage 3: ANSWER — single LLM call from full context
    # ================================================================
    final_prompt = answer_prompt.format(context=context, question=question)

    # D2: Append aggregation instruction for count/total queries
    # (the retrieval engine already boosted top_k for aggregation;
    # now ensure the LLM enumerates each instance before counting)
    from agentbase_core.retrieval.engine import RetrievalEngine
    from agentbase_core.retrieval.intent import IntentAnalyzer
    if IntentAnalyzer.is_aggregation_query(question):
        final_prompt += RetrievalEngine.get_aggregation_prompt_suffix()

    answer = await call_llm(final_prompt, max_tokens=max_tokens)
    return answer


# ---------------------------------------------------------------------------
# Event Summarization Task
# ---------------------------------------------------------------------------

EVENT_SUMMARY_PROMPT = """List the significant events for {speaker} from {start_time} to {end_time}.

Rules:
- Output ONLY event descriptions, one per line, NO preamble or introduction.
- Each line: a specific event with who/what/when details.
- Focus on {speaker}'s actions, decisions, experiences, and changes.
- Include causal connections between events across sessions when they exist.
- Use information ONLY from sessions dated on or before {end_time}.
- If no relevant events are found, output: No significant events found.

Context:
{context}

Events for {speaker}:"""


async def summarize_events(
    db: AgentBase,
    sample: dict,
) -> dict[str, dict[str, str]]:
    """Generate event summaries for each speaker in each session.

    E1: Multi-path retrieval — speaker-specific + session-scoped + time-windowed.
    E2: Speaker-filtered context assembly with time-window truncation.
    E3: Structured prompt with no preamble.

    Returns: {session_key: {speaker: predicted_summary}}
    """
    event_summary = sample["event_summary"]
    predictions: dict[str, dict[str, str]] = {}

    for session_key, speakers_events in event_summary.items():
        # session_key format: "events_session_N"
        if not session_key.startswith("events_session_"):
            continue

        session_num = session_key.replace("events_session_", "")
        predictions[session_key] = {}

        # Find the session date for context
        session_date = ""
        session_idx = int(session_num) - 1  # 0-based
        for s in sample["sessions"]:
            if s["session_id"] == int(session_num):
                session_date = s["date_time"]
                break

        # Determine time range (from session 1 to this session)
        all_dates = [s["date_time"] for s in sample["sessions"] if s["date_time"]]
        start_time = all_dates[0] if all_dates else "unknown"
        end_time = session_date or "unknown"

        # Parse date_to for time-window filtering
        date_to = _parse_locomo_date(session_date) if session_date else None

        for speaker in speakers_events:
            # ==============================================================
            # E1: Multi-path retrieval for ES
            # ==============================================================
            all_results: list = []
            seen_ids: set[str] = set()

            def _absorb(results: list) -> None:
                for r in results:
                    if r.entry.id not in seen_ids:
                        seen_ids.add(r.entry.id)
                        all_results.append(r)

            # Path 1: Speaker + session scoped search
            try:
                r1 = await db.find(
                    query=f"{speaker} events",
                    top_k=40, token_budget=8000,
                    query_type="multi-session",
                )
                _absorb(r1)
            except Exception:
                pass

            # Path 2: Speaker-scoped search with time filtering
            if date_to is not None:
                try:
                    sq = SearchQuery(
                        text=speaker,
                        top_k=30, token_budget=6000,
                        query_type="multi-session",
                        date_to=date_to,
                    )
                    r2 = await db.search(sq)
                    _absorb(r2)
                except Exception:
                    pass

            # Path 3: Fallback — just speaker name
            if len(all_results) < 5:
                try:
                    r3 = await db.find(
                        query=speaker, top_k=20, token_budget=4000,
                    )
                    _absorb(r3)
                except Exception:
                    pass

            if not all_results:
                predictions[session_key][speaker] = "No significant events found."
                continue

            # ==============================================================
            # E2: Speaker-filtered + time-windowed context assembly
            # ==============================================================
            context = _assemble_es_context(
                all_results,
                speaker=speaker,
                max_session_index=session_idx,
            )

            if not context.strip():
                predictions[session_key][speaker] = "No significant events found."
                continue

            # ==============================================================
            # E3: Structured prompt (no preamble)
            # ==============================================================
            prompt = EVENT_SUMMARY_PROMPT.format(
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                context=context,
            )

            summary = await call_llm(prompt, max_tokens=1024)
            predictions[session_key][speaker] = summary

    return predictions


# ---------------------------------------------------------------------------
# Evaluation metrics — F1 (QA) and ROUGE-L (event summarization)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for F1 computation: lowercase, remove articles/punctuation."""
    text = str(text).lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference.

    This is the standard metric used in LoCoMo for QA evaluation.
    """
    pred_normalized = _normalize_text(prediction)
    ref_normalized = _normalize_text(reference)

    pred_tokens = pred_normalized.split()
    ref_tokens = ref_normalized.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    # Count occurrences for precision/recall
    pred_counts: dict[str, int] = defaultdict(int)
    ref_counts: dict[str, int] = defaultdict(int)
    for t in pred_tokens:
        pred_counts[t] += 1
    for t in ref_tokens:
        ref_counts[t] += 1

    num_common = sum(min(pred_counts[t], ref_counts[t]) for t in common)

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference.

    Used for event summarization evaluation.
    Uses Longest Common Subsequence (LCS).
    """
    pred_normalized = _normalize_text(prediction)
    ref_normalized = _normalize_text(reference)

    pred_tokens = pred_normalized.split()
    ref_tokens = ref_normalized.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Compute LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    data_file: str,
    output_file: str,
    limit: int | None = None,
    reader_model: str = DEFAULT_READER_MODEL,
    run_qa: bool = True,
    run_event_summary: bool = True,
    resume: bool = False,
) -> None:
    """Run the full LoCoMo benchmark with AgentBase (zero-LLM ingest)."""
    # Register reader_model as module-level so call_llm picks it up by default.
    global _READER_MODEL
    _READER_MODEL = reader_model
    print(f"Reader model: {reader_model} (thinking disabled for qwen* models)")

    print(f"Loading data from {data_file}...")
    samples = load_locomo(data_file)

    if limit:
        samples = samples[:limit]
        print(f"Limited to first {limit} samples")

    print(f"Total conversations: {len(samples)}")

    # Resume: skip already completed samples
    completed_ids: set[str] = set()
    if resume and Path(output_file).exists():
        with open(output_file) as rf:
            for line in rf:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        sid = rec.get("sample_id", "")
                        if sid:
                            completed_ids.add(sid)
                    except (json.JSONDecodeError, KeyError):
                        pass
        if completed_ids:
            remaining = [s for s in samples if s["sample_id"] not in completed_ids]
            print(f"Resume mode: skipping {len(completed_ids)} completed samples, {len(remaining)} remaining")
            samples = remaining

    # Prepare output directory for per-conversation databases
    db_dir = Path(tempfile.gettempdir()) / f"locomo_agentbase_{int(time.time())}"
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
        except OSError:
            print(f"  Warning: could not clean up {db_dir}, creating new dir anyway")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    qa_results: list[dict] = []
    event_results: list[dict] = []
    total_time = 0

    for sample_idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        speaker_a = sample["speaker_a"]
        speaker_b = sample["speaker_b"]
        num_sessions = len(sample["sessions"])
        num_qa = len(sample["qa"])

        print(f"\n{'='*60}")
        print(f"[{sample_idx+1}/{len(samples)}] Sample {sample_id}")
        print(f"  Speakers: {speaker_a} & {speaker_b}")
        print(f"  Sessions: {num_sessions}, QA pairs: {num_qa}")

        # Each conversation gets an isolated database with RRF hybrid retrieval
        # Zero-LLM ingest: no LLM needed for entity extraction or session processing
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        embed_config = EmbeddingConfig(
            model="text-embedding-v4",
            dimensions=1024,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=dashscope_api_key,
        )
        ab_config = AgentBaseConfig(
            data_dir=db_dir,
            db_filename=f"sample_{sample_id}.db",
            embedding=embed_config,
            index=IndexConfig(
                vector_enabled=True,
                fts_weight=0.4,
                vec_weight=0.6,
                rrf_k=60,
            ),
            retrieval=RetrievalConfig(
                default_top_k=20,
                default_token_budget=12000,
                query_decomposition=True,      # Local rule-based query decomposition
                ner_boost=True,                # NER-aware query expansion + result boosting
                ner_weight=0.3,                # NER signal weight in three-way fusion
                session_co_retrieval=True,     # D1: Session Co-Retrieval
                co_retrieve_min_turns=3,       # D1: min turns per session before co-retrieval
                agg_detection=True,            # D2: aggregation query detection + top_k boost
                agg_top_k=80,                  # D2: top_k for aggregation queries
                freshness_half_life_days=7.0,   # Recency boost half-life
                knowledge_update_half_life_days=30.0,
            ),
        )
        # Disable graph and session services (zero-LLM ingest handles everything locally)
        ab_config.graph = GraphConfig(enabled=False)
        ab_config.session = SessionConfig(enabled=False)

        # Optimization B: wrap embedder with batch caching
        real_embedder = LiteLLMEmbedder(embed_config)
        batch_embedder = BatchEmbedder(real_embedder)
        db = AgentBase(config=ab_config, embedder=batch_embedder)
        await db.initialize()

        try:
            # Step 1: Ingest conversation (zero-LLM, batch-embedding)
            print(f"  Ingesting {num_sessions} sessions (zero-LLM)...")
            ingest_start = time.time()
            await ingest_conversation(db, sample, batch_embedder=batch_embedder)
            ingest_time = time.time() - ingest_start
            print(f"  Ingestion done: {ingest_time:.1f}s")

            # Step 2: QA Task (Optimization A: concurrent execution)
            if run_qa:
                print(f"\n  Running QA task ({num_qa} questions, {QA_CONCURRENCY} concurrent)...")
                qa_start = time.time()

                # Build list of QA items with ground truth pre-computed
                qa_items: list[dict] = []
                for qa_idx, qa in enumerate(sample["qa"]):
                    category = qa["category"]
                    ground_truth = qa["answer"] or ""
                    if category == 5 and qa.get("adversarial_answer"):
                        ground_truth = qa["adversarial_answer"]
                    ground_truth = str(ground_truth) if ground_truth is not None else ""
                    qa_items.append({
                        "idx": qa_idx,
                        "question": qa["question"],
                        "category": category,
                        "ground_truth": ground_truth,
                    })

                sem = asyncio.Semaphore(QA_CONCURRENCY)
                write_lock = asyncio.Lock()

                async def answer_one(item: dict) -> dict:
                    """Answer a single QA (runs concurrently via gather)."""
                    async with sem:
                        cat_name = QA_CATEGORY_NAMES.get(item["category"], "unknown")
                        print(f"    [{item['idx']+1}/{num_qa}] cat={cat_name}: {item['question'][:80]}...")
                        try:
                            hypothesis = await answer_qa(
                                db, item["question"], category=item["category"],
                                speakers=(sample["speaker_a"], sample["speaker_b"]),
                            )
                        except Exception as e:
                            print(f"    ERROR [{item['idx']+1}]: {e}")
                            hypothesis = "Error: unable to answer"

                        f1 = compute_token_f1(hypothesis, item["ground_truth"])
                        print(f"    F1={f1:.3f} | A: {hypothesis[:100]}...")

                        result = {
                            "sample_id": sample_id,
                            "question": item["question"],
                            "ground_truth": item["ground_truth"],
                            "hypothesis": hypothesis,
                            "category": item["category"],
                            "category_name": cat_name,
                            "f1": f1,
                        }
                        # Stream-write with lock to avoid file corruption
                        async with write_lock:
                            with open(output_file, "a") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        return result

                # Launch all QA questions concurrently
                tasks = [answer_one(item) for item in qa_items]
                results_raw = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and collect valid results
                sample_qa_results = []
                for r in results_raw:
                    if isinstance(r, Exception):
                        print(f"    QA task crashed: {r}")
                    elif isinstance(r, dict):
                        sample_qa_results.append(r)
                        qa_results.append(r)

                qa_time = time.time() - qa_start
                print(f"  QA task done: {qa_time:.1f}s ({len(sample_qa_results)} answers)")

            # Step 3: Event Summarization Task
            if run_event_summary and sample["event_summary"]:
                print(f"\n  Running Event Summarization task...")
                es_start = time.time()

                try:
                    predictions = await summarize_events(db, sample)

                    # Compute ROUGE-L for each prediction
                    for session_key, speakers_events in sample["event_summary"].items():
                        if session_key not in predictions:
                            continue
                        for speaker in speakers_events:
                            # Skip non-speaker keys like "date"
                            if speaker in ("date",) or not isinstance(speakers_events[speaker], list):
                                continue
                            # Skip speakers with empty event lists
                            gt_events = speakers_events[speaker]
                            if not gt_events:
                                continue
                        
                            pred_summary = predictions[session_key].get(speaker, "")
                            # Ground truth is a list of events — join them
                            if isinstance(gt_events, list):
                                gt_text = " ".join(gt_events)
                            else:
                                gt_text = str(gt_events)

                            rouge_l = compute_rouge_l(pred_summary, gt_text)

                            event_results.append({
                                "sample_id": sample_id,
                                "session_key": session_key,
                                "speaker": speaker,
                                "ground_truth": gt_text,
                                "hypothesis": pred_summary,
                                "rouge_l": rouge_l,
                            })

                            # Stream-write each event result immediately
                            with open(output_file, "a") as f:
                                f.write(json.dumps(event_results[-1], ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"    ERROR in event summarization: {e}")

                es_time = time.time() - es_start
                print(f"  Event Summarization done: {es_time:.1f}s")

        except Exception as e:
            print(f"  ERROR processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await db.close()

        elapsed = time.time() - total_time - 0  # already tracked per task
        total_time += elapsed

    # ---------------------------------------------------------------------------
    # Write results
    # ---------------------------------------------------------------------------

    # Write final consolidated results (overwrite with clean file)
    all_results = qa_results + event_results
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total results: {len(all_results)}")

    # ---------------------------------------------------------------------------
    # Print evaluation summary
    # ---------------------------------------------------------------------------

    if qa_results:
        print(f"\n{'='*60}")
        print("QA EVALUATION SUMMARY")
        print(f"{'='*60}")

        # Overall F1
        all_f1 = [r["f1"] for r in qa_results]
        overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
        print(f"  Overall F1: {overall_f1:.3f} ({len(all_f1)} questions)")

        # Per-category F1
        cat_f1s: dict[int, list[float]] = defaultdict(list)
        for r in qa_results:
            if r["category"] is not None:
                cat_f1s[r["category"]].append(r["f1"])

        for cat_id in sorted(cat_f1s.keys()):
            cat_name = QA_CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
            f1s = cat_f1s[cat_id]
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0
            print(f"  {cat_name:15s}: F1={avg_f1:.3f} ({len(f1s)} questions)")

    if event_results:
        print(f"\n{'='*60}")
        print("EVENT SUMMARIZATION EVALUATION SUMMARY")
        print(f"{'='*60}")

        all_rouge = [r["rouge_l"] for r in event_results]
        overall_rouge = sum(all_rouge) / len(all_rouge) if all_rouge else 0
        print(f"  Overall ROUGE-L: {overall_rouge:.3f} ({len(all_rouge)} summaries)")

    # Cleanup
    if db_dir.exists():
        try:
            shutil.rmtree(db_dir)
            print(f"\nCleaned up temp databases in {db_dir}")
        except OSError:
            print(f"\nWarning: could not fully clean up {db_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark with AgentBase")
    parser.add_argument(
        "--data-dir",
        default="../locomo/data",
        help="LoCoMo data directory containing locomo10.json",
    )
    parser.add_argument(
        "--dataset",
        default="locomo10.json",
        help="Dataset file name (default: locomo10.json)",
    )
    parser.add_argument(
        "--output",
        default="predictions_locomo.jsonl",
        help="Output predictions file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations",
    )
    parser.add_argument(
        "--reader-model",
        default=DEFAULT_READER_MODEL,
        help=f"LLM model for answering (default: {DEFAULT_READER_MODEL} via DashScope native SDK; thinking disabled for qwen* models)",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Run only QA task (skip event summarization)",
    )
    parser.add_argument(
        "--event-summary-only",
        action="store_true",
        help="Run only event summarization task (skip QA)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output, skipping completed samples",
    )
    args = parser.parse_args()

    data_file = Path(args.data_dir) / args.dataset
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please download the LoCoMo dataset first:")
        print("  git clone https://github.com/snap-research/locomo.git")
        print(f"  Expected data at: {data_file}")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable is required")
        sys.exit(1)

    run_qa = not args.event_summary_only
    run_event_summary = not args.qa_only

    asyncio.run(
        run_benchmark(
            data_file=str(data_file),
            output_file=args.output,
            limit=args.limit,
            reader_model=args.reader_model,
            run_qa=run_qa,
            run_event_summary=run_event_summary,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
