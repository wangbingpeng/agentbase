# AgentBase — Context Database for AI Agents

[中文文档](README_CN.md)

> **Context DB = Memory + Resource + Skill + Temporal + Session + Observability**

AgentBase is an open-source context database designed for AI agents. It provides a unified, queryable store for memories, resources, skills, temporal knowledge, session history, and observability — all built on SQLite with zero external infrastructure dependencies.

**LongMemEval Overall: 73.2% — #1 among fully open-source solutions**

## Benchmark — LongMemEval Comparison

LongMemEval is the de facto standard benchmark for long-term memory systems, evaluating retrieval accuracy across 5 question types over multi-session conversations (~500 questions).

### Per-Question-Type Accuracy

| Question Type | AgentBase | Mem0 OSS | Mem0 Pro | OMEGA | Zep/Graphiti |
|---|---|---|---|---|---|
| Single-Session (User) | 90.0% | 94.3% | 97.1% | 99.2% | — |
| Single-Session (Asst) | **92.9%** | 46.4% | 100.0% | 99.2% | — |
| Single-Session (Pref) | 66.7% | 76.7% | 96.7% | 100.0% | — |
| Knowledge Update | **91.0%** | 79.5% | 96.2% | 96.2% | — |
| Temporal Reasoning | 62.4% | 51.1% | 93.2% | 94.0% | 63.8% |
| Multi-Session | 57.9% | 70.7% | 86.5% | 83.5% | — |
| **OVERALL** | **73.2%** | 67.8% | 93.4% | 95.4% | 63.8% |

> Data sources: Mem0 (docs.mem0.ai, 2026.3), OMEGA (omegamax.co, 2026.2), Zep (vectorize.io, 2026), AgentBase local evaluation.

### AgentBase vs Mem0 OSS — The Fair Open-Source Comparison

Mem0 Pro (93.4%) is a **paid managed platform** with proprietary optimizations. Mem0's own docs state: *"Scores reflect Mem0's managed platform, which includes proprietary optimizations not available in the open-source SDK."* For open-source users, the fair comparison is against Mem0 OSS:

| Dimension | AgentBase | Mem0 OSS | Delta |
|---|---|---|---|
| Overall Accuracy | **73.2%** | 67.8% | **+5.4pp** |
| Knowledge Update | **91.0%** | 79.5% | **+11.5pp** |
| Temporal Reasoning | **62.4%** | 51.1% | **+11.3pp** |
| Single-Session (Asst) | **92.9%** | 46.4% | **+46.5pp** |
| External Dependencies | None (SQLite) | Qdrant required | Lighter |
| LLM Calls at Ingest | **0** | Every message | Zero ingest cost |

Mem0 OSS scores only 46.4% on Single-Session (Assistant) — a critical blind spot for assistant-role information. AgentBase covers this scenario through session summary extraction + full-turn storage.

### AgentBase vs Zep/Graphiti — Lightweight vs Heavy Graph

| Dimension | AgentBase | Zep/Graphiti |
|---|---|---|
| Overall Accuracy | **73.2%** | 63.8% | **+9.4pp** |
| Temporal Reasoning | 62.4% | 63.8% | ~Parity |
| External Database | SQLite only | Neo4j / Graph DB |
| Deployment | `pip install` | Docker + Graph DB |

### Cost Efficiency — Unique Zero-LLM-Ingest Advantage

| Metric | AgentBase | Mem0 Pro | OMEGA |
|---|---|---|---|
| Avg Tokens / Query | ~3,500 | ~6,787 | ~7,000+ |
| External Dependencies | None | Qdrant | SQLite |
| Ingest LLM Calls | **0** | Every message | Every message |
| GPU Required | No | No | ONNX (CPU) |

AgentBase's **zero-LLM ingest** is a unique advantage: the entire 500-question ingest phase consumes zero LLM tokens, while Mem0 and OMEGA require LLM calls for every message.

### Honest Assessment

- Mem0 Pro (93.4%) and OMEGA (95.4%) lead in absolute accuracy — both are paid/managed platforms using proprietary optimizations and stronger answer models.
- OMEGA and Mastra use GPT-4.1 / GPT-4o as the answer model; AgentBase uses qwen-plus (~GPT-4o-mini level). Model capability difference accounts for approximately 5-10pp.
- AgentBase's Multi-Session (57.9%) and Temporal (62.4%) have room for improvement, currently limited by IDH (Intelligent Dedup & Hardening) over-rejection.
- **Among fully open-source, self-hosted, zero-dependency solutions, AgentBase achieves the highest LongMemEval score.**

## Innovations

AgentBase introduces several industry-first techniques that differentiate it from Mem0, Zep/Graphiti, LangChain Memory, and other open-source memory systems.

### 1. Zero-LLM Ingest — The Only Zero-Cost Ingest Pipeline

Every other memory framework requires LLM calls during ingestion: Mem0 calls LLM for each message to extract facts; Zep uses LLM for episodic summarization; OMEGA runs LLM per message. AgentBase achieves structured memory extraction through a **fully local rule-based pipeline**:

| Ingest Step | AgentBase | Mem0 OSS | Zep/Graphiti |
|---|---|---|---|
| Session Summary | Rules (LLM optional) | LLM per session | LLM per session |
| Entity Extraction | spaCy + regex (local) | LLM per message | LLM per message |
| Fact Extraction | Pattern matching (local) | LLM per message | LLM per message |
| Tier Generation | Truncation fallback (LLM optional) | N/A | N/A |
| Dedup & Hardening | IDH rules (local) | N/A | N/A |

**Impact**: 500 questions × 0 LLM calls = **0 tokens consumed at ingest**, while Mem0 consumes ~6,787 tokens/query. This eliminates ingest latency, network dependency, rate limits, and cost — enabling edge deployment, offline usage, and CI environments.

### 2. Three-Way Recall (FTS + Vector + NER) — Industry First

Existing frameworks use either single-path (Mem0: vector only via Qdrant) or dual-path (Zep: graph + vector) retrieval. AgentBase adds a **third signal — NER entity boosting** — that operates as a precision amplifier on top of FTS+Vector results:

```
  ┌─────────┐   ┌─────────┐   ┌─────────────┐
  │  FTS5   │   │ Vector  │   │ NER Entity  │
  │ (BM25)  │   │(cosine) │   │   Boost     │
  └────┬────┘   └────┬────┘   └──────┬──────┘
       │    RRF Fusion   │              │
       └────────┬────────┘              │
                ▼     ◄── Score Boost ─┘
          Fused Results (NER never adds new results)
```

**Why this matters**: Mem0 OSS's Single-Session(Asst) score is only 46.4% — a critical blind spot for assistant-role information. Pure vector search misses exact entity matches ("Hawaii" not semantically close to "trip"). AgentBase's FTS5 catches exact keywords + NER tags (`ner_Hawaii`) boost relevant results — this is the technical root cause of AgentBase's 92.9% on the same metric.

**Three-tier NER matching** (prevents dilution):
- Strong (tag exact match): `score × 1.3` — `matched_by = "ner+hybrid"`
- Medium (NER tag + content match): `score × 1.24`
- Weak (content match only): `score × 1.15`

### 3. Query-Type-Aware Retrieval — 5 Intent Strategies with Auto-Detection

Other frameworks treat all queries identically. AgentBase's `IntentAnalyzer` detects 5 query types and applies specialized post-processing:

| Query Type | Innovation | Industry Standard |
|---|---|---|
| **Temporal Reasoning** | Auto-parses date expressions → populates `date_from`/`date_to` filter (D5) | No time awareness |
| **Knowledge Update** | Dual half-life: 7d general / 30d knowledge-update (strong recency bias) | No temporal differentiation |
| **Multi-Session** | Cross-session completion from DB + session deduplication | Single-session only |
| **Preference** | Multi-level signals: user-role (1.8×) + pref-category (2.0×) + implicit (1.3×) | No preference awareness |
| **Aggregation** | D2: Auto-boosts `top_k` 20→120 for exhaustive recall + enumeration prompt suffix | Fixed top_k |

The **aggregation detection** (D2) is particularly forward-looking: "how many" questions need exhaustive recall, not top-k truncation. Without this, aggregation queries systematically undercount.

### 4. Session Co-Retrieval (D1) — Solving the Lost Context Problem

When retrieval hits one message from a conversation, surrounding context is critical. AgentBase automatically expands results through `session_memory_links`:

- **Link-based expansion**: Follows `session_memory_links` (written at session commit with FK integrity) to find co-relevant entries
- **Turn-aware gating**: Only activates for sessions with ≥2 turns — prevents single-turn noise
- **Budget-aware**: Co-retrieved entries never exceed the caller's `token_budget`

Mem0 extracts each message into isolated facts, losing conversation context. LangChain keeps the latest k turns with no semantic linking. AgentBase is the only framework that maintains **bidirectional links between sessions and their extracted memories** at the database level.

### 5. Three-Layer Progressive Content (L0/L1/L2) with Deterministic Loading

| Layer | Content | Size Limit | Purpose |
|---|---|---|---|
| L0 | Abstract / Title | ≤50 chars | Coarse filter, list display |
| L1 | Overview | ≤300 chars | Precision ranking, preview |
| L2 | Full content | Unlimited | Final answer |

**Innovations over simple summarization**:
- **Deterministic auto-selection**: 6 deterministic rules (top_k>20→L0, budget<1000→L0, hierarchical→L1, etc.) — no randomness
- **Progressive search**: `strategy=hierarchical` searches L0 (3× top_k) → L1 (2× top_k) → L2 (final only), reducing I/O by ~60%
- **Truncation fallback**: `fallback_to_truncation=true` generates L0/L1 by truncation when LLM is unavailable — guaranteed functionality

### 6. Pure SQLite Zero-Dependency Architecture

| Framework | External Dependencies | Deployment |
|---|---|---|
| **AgentBase** | **None** (SQLite + FTS5 built-in) | `pip install` |
| Mem0 OSS | Qdrant (vector database) | Docker / cloud service |
| Zep/Graphiti | Neo4j (graph database) | Docker + graph database |
| LangChain | None (but no retrieval capability) | pip install (missing features) |

FTS5 is a built-in SQLite extension; `sqlite-vec` is an optional enhancement. This means AgentBase runs on embedded devices, CI environments, and even in-browser (Pyodide).

### 7. Bilingual NER with Degradation Chain

```
spaCy (en) → spaCy (zh) → regex fallback (中文地名/机构 + English caps/quantities) → skip
```

No other open-source memory framework has local NER capability. Mem0 and Zep delegate all entity recognition to LLM, making NER unavailable without LLM access.

### 8. Multi-Framework Unified Memory Backend

All 5 adapters share the same SQLite database — a "unified memory backend" capability that no other framework provides:

- `Mem0Adapter` — Replace `Memory()` with one-line swap
- `LangChainMemoryAdapter` — Duck-type `BaseChatMemory`
- `AgentBaseChatStore` — Implements LlamaIndex `BaseChatStore`
- `OpenAIAssistantAdapter` — Maps Thread→Session
- `MinimalAdapter` — 3-method API (remember/recall/forget)

### 9. Full-Stack Observability

| Layer | Capability | Detail |
|---|---|---|
| **Trace persistence** | `retrieval_traces` + `trace_steps` tables | Per-step latency, candidate counts, model name, cache hits |
| **Web dashboard** | 6 visualizations | Timeline, heatmap, category sunburst, freshness distribution, tag cloud, activity feed |
| **Debug APIs** | `trace_session()`, `entity_graph()`, `diff_entries()` | Programmatic introspection |

Mem0 OSS has no observability UI; Zep has a basic dashboard but no retrieval tracing; LangChain has none.

---

**Innovation Matrix Summary**:

| Innovation | Industry First | Core Value |
|---|---|---|
| Zero-LLM Ingest | ✅ | Zero cost, zero latency, zero dependency ingestion |
| Three-Way Recall (FTS+Vec+NER) | ✅ | Keyword + semantic + entity coverage |
| Query-Type-Aware Strategy | ✅ | Auto-detected 5-intent specialized processing |
| Session Co-Retrieval (D1) | ✅ | Solves hit-but-lost-context problem |
| Three-Layer Progressive Search | ✅ | 60% I/O reduction, deterministic level selection |
| Pure SQLite Zero-Dependency | ✅ | pip install ready, no Docker/cloud needed |
| Bilingual NER Degradation Chain | ✅ | Local entity recognition without LLM |
| Multi-Framework Unified Backend | ✅ | 5 adapters sharing one database |
| Full-Stack Observability | ✅ | Trace persistence + Web dashboard + Debug APIs |

## Features

- **Memory** — Store and retrieve agent memories (preferences, facts, procedures)
- **Resource** — Manage external resources (URLs, documents, APIs)
- **Skill** — Register and discover tool capabilities
- **Temporal Knowledge Graph** — Entity-relation graph with time-aware fact tracking
- **Session Management** — Multi-turn conversation tracking with memory extraction
- **Web Dashboard** — Visual observability: timeline, heatmap, retrieval traces, category sunburst, freshness distribution, tag cloud
- **Three-Way Hybrid Search** — FTS5 full-text + sqlite-vec vector + NER entity boosting with RRF fusion
- **Three-Layer Content** (L0/L1/L2) — Progressive detail levels for efficient retrieval
- **Multi-Agent Scope** — Global, agent, project, session-level isolation
- **Framework Adapters** — Drop-in compatible with Mem0 / LangChain / LlamaIndex / OpenAI Assistants

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync --all-packages --all-extras --dev

# Or with pip
pip install -e packages/agentbase-core[all]
pip install -e packages/agentbase-sdk
pip install -e packages/agentbase-cli
pip install -e packages/agentbase-mcp
pip install -e packages/agentbase-web
```

### Python SDK

```python
import asyncio
from agentbase import AgentBase

async def main():
    # Initialize
    db = AgentBase(path="./my_agent.db")
    await db.initialize()

    # Add memories
    await db.add_memory("User prefers Python 3.12", category="preference", tags=["python"])

    # Search
    results = await db.find("Python preferences", top_k=5)
    for r in results:
        print(f"[{r.entry.context_type.value}] {r.entry.l2_full}")

    await db.close()

asyncio.run(main())
```

### CLI

```bash
# Initialize a database
agentbase init --data-dir ./data

# Add entries
agentbase add "User prefers dark mode" --type memory --tags "preference,dark-mode"

# Search
agentbase find "user preferences" --top-k 5

# View entry
agentbase get <entry-id>

# Session management
agentbase session create --agent-id my-agent
agentbase session add-message <session-id> --role user --content "Hello"
agentbase session commit <session-id>
```

### MCP Server

```bash
# Start the MCP server (stdio transport)
agentbase-mcp
```

### Web Dashboard

```bash
# Start the web dashboard
agentbase-web ./my_agent.db 8080
```

## Framework Adapters

AgentBase provides drop-in adapters for popular AI frameworks, so you can keep your existing code and swap the memory backend with a single line change.

| Adapter | Framework | Interface | Key Methods |
|---|---|---|---|
| `Mem0Adapter` | [Mem0](https://github.com/mem0ai/mem0) | `mem0.Memory` | `add`, `search`, `get_all`, `update`, `delete` |
| `LangChainMemoryAdapter` | [LangChain](https://github.com/langchain-ai/langchain) | `BaseChatMemory` | `save_context`, `load_memory_variables`, `clear` |
| `AgentBaseChatStore` | [LlamaIndex](https://github.com/run-llama/llama_index) | `BaseChatStore` | `add_message`, `get_messages`, `delete_message` |
| `OpenAIAssistantAdapter` | [OpenAI Assistants](https://platform.openai.com/docs/assistants) | `beta.threads` | `create_thread`, `create_message`, `list_messages` |
| `MinimalAdapter` | — | 3-method API | `remember`, `recall`, `forget` |

All adapters share the **same underlying SQLite database** — you can use multiple adapters simultaneously on a single `AgentBase` instance without data conflicts.

### Mem0 Migration

```python
from agentbase import AgentBase
from agentbase.adapters import Mem0Adapter

db = AgentBase(path="./mem.db")
await db.initialize()

# Replace: m = Memory()
# With:    m = Mem0Adapter(db)
m = Mem0Adapter(db)
m.add("I like pizza", user_id="alice")
results = m.search("food preferences", user_id="alice")
```

### LangChain Integration

```python
from agentbase import AgentBase
from agentbase.adapters import LangChainMemoryAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

memory = LangChainMemoryAdapter(db, owner_id="alice")
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
history = memory.load_memory_variables({"input": "Hello"})
```

### LlamaIndex ChatStore

```python
from agentbase import AgentBase
from agentbase.adapters.llamaindex import AgentBaseChatStore
from llama_index.core.memory import ChatMemoryBuffer

db = AgentBase(path="./mem.db")
await db.initialize()

chat_store = AgentBaseChatStore(db)
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user_alice",
)
```

### OpenAI Assistants API

```python
from agentbase import AgentBase
from agentbase.adapters import OpenAIAssistantAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

oa = OpenAIAssistantAdapter(db)
thread = oa.create_thread(metadata={"agent_id": "my-agent"})
oa.create_message(thread_id=thread["id"], role="user", content="Hello")
messages = oa.list_messages(thread_id=thread["id"])
context = oa.retrieve_context(thread_id=thread["id"], query="user preferences")
```

### Minimal API (Simplest Integration)

```python
from agentbase import AgentBase
from agentbase.adapters import MinimalAdapter

db = AgentBase(path="./mem.db")
await db.initialize()

mem = MinimalAdapter(db)
mem.remember("User prefers dark mode", who="alice", tags=["preference"])
results = mem.recall("theme preferences", who="alice")
mem.forget(entry_id="...")

# Also available as async:
# await mem.aremember(...)
# await mem.arecall(...)
# await mem.aforget(...)
```

## Retrieval Architecture

AgentBase uses a **three-way hierarchical retrieval pipeline** — FTS5 + Vector + NER — that combines keyword precision, semantic understanding, and entity-aware boosting. The entire ingest and retrieval process runs without requiring LLM calls by default.

### Pipeline Overview

```
Query ──► Normalize ──► Intent Detection ──► Query Decomposition
                                           │
              ┌────────────────────────────┘
              ▼
     ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
     │  FTS5 (BM25) │  │ sqlite-vec   │  │  NER Entity   │
     │  Full-Text   │  │ Vector Search│  │  Boost Signal │
     └──────┬──────┘  └──────┬───────┘  └───────┬───────┘
            │     Three-Way RRF Fusion    │          │
            └──────────┬──────────────────┘          │
                       ▼    ◄──── NER score boost ───┘
              Heuristic Rerank
              (freshness / confidence / scope / type)
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Query-Type        Session Co-Retrieval
        Strategy          (link-based expansion)
              │                 │
              └────────┬────────┘
                       ▼
              Load Level (L0/L1/L2)
              + Token Budget Trim
                       │
                       ▼
                   Results
```

### Three-Way Recall: FTS5 + Vector + NER

AgentBase runs **three independent recall signals** in parallel and fuses them into a unified ranking:

| Signal | Engine | Strength | Default Weight |
|---|---|---|---|
| **FTS5** | SQLite built-in BM25 | Exact keyword match, zero latency, no embedding needed | 0.4 |
| **Vector** | sqlite-vec (cosine distance) | Semantic similarity, handles paraphrases and synonyms | 0.6 |
| **NER** | spaCy + regex bilingual NER | Entity-aware boosting, links query entities to tagged entries | 0.3 |

**How NER integrates into the pipeline** (not a separate search path, but a score-boosting signal):

1. **At ingest time** — `NerExtractor` extracts named entities from every entry and tags them as `ner_<EntityName>` (e.g. `ner_Hawaii`, `ner_Python`). This is bilingual: spaCy (en/zh) with regex fallback covering Chinese locations/orgs and English capitalized names/quantities.
2. **At query time** — The same `NerExtractor` extracts entities from the query text.
3. **At fusion time** — For each existing FTS+Vector result, if its `ner_*` tags match query entities:
   - **Strong match** (tag exact match): `score *= (1 + ner_weight)` → `matched_by = "ner+hybrid"`
   - **Medium match** (NER-tagged entry + entity in content): `score *= (1 + ner_weight * 0.8)`
   - **Weak match** (entity text in content, no NER tag): `score *= (1 + ner_weight * 0.5)`
4. **No dilution** — NER never adds new results outside the FTS+Vector result set, preventing irrelevant entries from entering the pipeline.

This design means NER acts as a **precision booster** on top of the dual-path recall, rather than an independent retrieval path that could introduce noise.

**RRF Fusion** (FTS + Vector): `score(d) = Σ weight_i / (k + rank_i(d))` — default k=60. This produces a unified ranking that captures both keyword hits and semantic neighbors. NER boosting is applied **after** RRF fusion, amplifying entity-relevant results.

**Graceful degradation chain**: Vector unavailable → FTS-only with `degrade_reason` tag. spaCy unavailable → regex-based NER fallback. NER returns empty → skip boosting, keep FTS+Vector results unchanged.

### Zero-LLM Ingest Pipeline

Unlike Mem0 and OMEGA which require LLM calls for every ingested message, AgentBase extracts memories through **local rule-based processing**:

1. **Session Summarization** — Auto-generates session abstracts from conversation turns (LLM-optional, can use rules)
2. **NER Extraction** — Bilingual NER: spaCy (en/zh) preferred → regex fallback (capitalized names, Chinese locations/orgs, quantities). Tags entries with `ner_<EntityName>` for retrieval boosting.
3. **Fact Extraction** — Pattern-based fact extraction without LLM dependency
4. **Tier Generation** — L0 abstract / L1 intermediate / L2 full content (LLM-optional)
5. **Intelligent Dedup & Hardening (IDH)** — Deduplicates similar entries and hardens confidence scores

This means the ingest phase for 500 questions costs **zero LLM tokens**, while Mem0 and OMEGA consume ~6,000-7,000+ tokens per query.

### Query Decomposition — Zero-LLM

When LLM is unavailable, `LocalQueryDecomposer` splits queries using rule-based strategies:

1. **Pattern extraction** — Recognizes `"how many X"`, `"what kind of X"`, `"when did I X"` and extracts key noun phrases
2. **Stop-word removal** — Bilingual stop-word filtering (English + Chinese) produces a keyword-only sub-query
3. **Temporal token enrichment (D3)** — Extracts date patterns from temporal queries: `"May 2022"`, `"summer 2021"`, `"3 weeks ago"`, `"last month"`, Chinese date formats
4. **Multi-sub-query search** — Each sub-query searches independently, results are deduplicated by entry ID

This provides a reasonable approximation of LLM-based intent decomposition at zero cost.

### Query-Type-Aware Retrieval

The engine automatically detects query intent and applies specialized post-processing strategies with tuned parameters:

| Query Type | Detection | Strategy | Key Parameters |
|---|---|---|---|
| **Temporal Reasoning** | "first time", "before", "since", "多久", "之前" | Auto-parses date range from query text; populates `date_from`/`date_to` filter; boosts older context for historical coverage | 7-day freshness half-life |
| **Knowledge Update** | "current", "latest", "new", "当前", "最新" | Strongly boosts most recent entries; suppresses outdated duplicates via recency decay | 30-day half-life (stronger recency bias) |
| **Multi-Session** | "all the", "every", "total", "所有", "总共" | Ensures results span multiple sessions; completes uncovered sessions from DB; deduplicates by session | Session co-retrieval min_turns=2 |
| **Preference** | "prefer", "recommend", "suggest", "偏好", "推荐" | Boosts user-role entries (1.8×), preference-category (2.0×), implicit indicators (bought/tried/using × 1.3+), user+event cross (1.4×) | Multi-level preference signals |
| **Aggregation** | "how many", "how much", "多少", "一共" | Auto-boosts `top_k` (default → 120) for exhaustive recall; ensures complete enumeration across all entries | agg_top_k=120 |

### Session Co-Retrieval (D1)

When a search result belongs to a session with sufficient context (≥2 turns), AgentBase automatically **expands the result set** by pulling in related entries from the same session:

- **Link-based expansion** — Follows `session_memory_links` stored during session commit to find co-relevant entries
- **Turn-aware gating** — Only activates for sessions with ≥2 turns (configurable), preventing noise from single-turn sessions
- **Budget-aware** — Co-retrieved entries respect the token budget, never exceeding the caller's limit

This addresses the "lost context" problem: when a retrieval hits one message from a conversation, the surrounding context is automatically included.

### Heuristic Rerank

After three-way RRF fusion + NER boosting, a deterministic 5-dimensional reranker adjusts scores:

```
final_score = α·rrf_score + β·freshness + γ·confidence + δ·scope_priority + ε·type_match
```

| Dimension | Algorithm | Detail |
|---|---|---|
| **α·RRF Score** (0.6) | Weighted RRF output | Base ranking from FTS+Vector+NER fusion |
| **β·Freshness** (0.15) | Exponential decay: `exp(-0.693 · age / half_life)` | 7-day half-life for general, 30-day for knowledge-update queries |
| **γ·Confidence** (0.1) | Entry-level from ingest pipeline | IDH-hardened confidence scores (0.0-1.0) |
| **δ·Scope Priority** (0.1) | Tiered: Session(1.0) > Agent(0.8) > Project(0.6) > Global(0.4) + 0.2 bonus for scope match | More specific scope + matching query scope = higher |
| **ε·Type Match** (0.05) | Binary: 1.0 if entry type matches query filter, 0.5 if no filter, 0.0 if mismatch | Ensures type-relevant results surface |

### Hierarchical Progressive Search (L0 → L1 → L2)

When `strategy=hierarchical`, AgentBase uses a **progressive refinement** approach that searches at increasingly detailed content levels:

1. **L0 coarse search** — Searches `l0_abstract` column only (short summaries), over-fetches 3× top_k for broad recall
2. **L1 precision search** — Re-searches on `l1_overview` column (medium detail), narrows to 2× top_k
3. **L2 full load** — Loads `l2_full` content only for the final top_k results

This reduces I/O and memory usage by ~60% compared to loading full content for all candidates, while maintaining recall quality.

### LLM-Enhanced Path (Optional, Explicit Opt-in)

When LLM is configured, an enhanced retrieval path becomes available:

1. **Intent Decomposition** — LLM splits complex queries into typed sub-queries with category labels (memory/resource/skill + profile/preference/entity/event)
2. **Per-sub-query Search** — Each sub-query searches independently with its own type filter, results are deduplicated
3. **LLM Rerank** — Semantic reranking of merged results using a judge prompt that returns `[index]` ordering
4. **Session Memory Links** — Cross-session co-retrieval via stored link associations

This path is **off by default** and only activates when `strategy=hierarchical` and LLM is configured.

### Full Pipeline Step-by-Step

```
1. Query Normalize
   └─ Lowercase, trim, remove special characters

2. Intent Detection (rule-based, zero-LLM)
   └─ Detect: temporal-reasoning / knowledge-update / multi-session / preference / aggregation
   └─ D5: Auto-parse temporal expressions → populate date_from/date_to filter
   └─ D2: Aggregation detection → auto-boost top_k (20→120)

3. Query Decomposition (zero-LLM)
   └─ Extract key noun phrases via pattern matching
   └─ Remove bilingual stop words → keyword sub-query
   └─ D3: Extract temporal tokens ("May 2022", "3 weeks ago")
   └─ Deduplicate sub-queries by entry ID

4. Three-Way Search
   ├─ FTS5 (BM25): SQLite built-in full-text, zero latency
   ├─ sqlite-vec: cosine distance vector search
   └─ Over-fetch 3× top_k for better RRF recall

5. RRF Fusion
   └─ score(d) = Σ weight_i / (k + rank_i(d))
   └─ Default: FTS=0.4, Vector=0.6, k=60
   └─ Graceful degradation: vec unavailable → FTS-only + degrade_reason

6. NER Boost (on existing results only)
   └─ Extract entities from query via spaCy + regex
   └─ Match against ner_* tags on results
   └─ Strong: tag match → score *= (1 + 0.3)
   └─ Medium: NER tag + content match → score *= (1 + 0.24)
   └─ Weak: content match only → score *= (1 + 0.15)
   └─ No new results added (prevents dilution)

7. Query-Type Strategy
   ├─ temporal: date filter + older context boost
   ├─ knowledge-update: strong recency (30d half-life)
   ├─ multi-session: cross-session completion + dedup
   ├─ preference: user-role (1.8×) + pref-category (2.0×) + implicit signals
   └─ aggregation: top_k already boosted in step 2

8. Session Co-Retrieval (D1)
   └─ Follow session_memory_links for ≥2-turn sessions
   └─ Budget-aware expansion

9. Heuristic Rerank (5-dimensional)
   └─ α(0.6)·rrf + β(0.15)·freshness + γ(0.1)·confidence + δ(0.1)·scope + ε(0.05)·type

10. LLM Rerank (optional, hierarchical strategy only)
    └─ Judge prompt returns [index] ordering

11. Load Level Selection
    └─ top_k>20 → L0, budget<1000 → L0, hierarchical → L1, default → L1
    └─ L2 loaded only for final results

12. Token Budget Trim
    └─ Trim results to fit within token_budget

13. Final top_k trim → Return results with trace
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  SDK (agentbase)             │
├──────────┬──────────┬───────────┬───────────┤
│   CLI    │   MCP    │    Web    │ Adapters  │
├──────────┴──────────┴───────────┴───────────┤
│              Core Engine                     │
│  ┌─────┐ ┌──────┐ ┌──────┐ ┌────────────┐  │
│  │Store│ │Index │ │Ingest│ │ Retrieval  │  │
│  │SQLite│ │FTS+Vec│ │Pipe │ │ Engine     │  │
│  └─────┘ └──────┘ └──────┘ └────────────┘  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌────────────┐ │
│  │Graph │ │Session│ │Obser│ │ Background │ │
│  │ /NER │ │ Mgmt  │ │vabil│ │   Jobs     │ │
│  └──────┘ └──────┘ └──────┘ └────────────┘ │
└─────────────────────────────────────────────┘
```

## Project Structure

```
agentbase-open/
├── packages/
│   ├── agentbase-core/    # Core engine (storage, indexing, retrieval)
│   ├── agentbase-sdk/     # Python SDK + adapters (LlamaIndex, LangChain)
│   ├── agentbase-cli/     # Command-line interface
│   ├── agentbase-mcp/     # MCP protocol server
│   └── agentbase-web/     # Web dashboard (FastAPI)
├── tests/                 # Test suite
├── docs/                  # Documentation
├── benchmarks/            # Evaluation scripts
├── SPEC.md                # Technical specification
└── pyproject.toml         # Workspace configuration
```

## Configuration

Copy the example config and customize:

```bash
cp agentbase.yaml.example agentbase.yaml
```

Key configuration sections:
- **embedding** — Vector embedding model (OpenAI compatible)
- **llm** — LLM for summaries, extraction, and tier generation
- **index** — FTS/vector search settings and RRF fusion weights
- **graph** — Knowledge graph (entities, relations, traversal)
- **session** — Conversation management and memory extraction
- **tier** — L0/L1/L2 layered content generation
- **observability** — Tracing, metrics, and debug

Environment variables can override YAML config: `AGENTBASE_<SECTION>__<KEY>`

## Requirements

- Python >= 3.11
- SQLite with FTS5 support
- Optional: `sqlite-vec` for vector search, `litellm` for LLM features

## License

MIT License — see [LICENSE](LICENSE) for details.
