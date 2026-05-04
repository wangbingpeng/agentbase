# AgentBase User Guide

[中文版](USAGE.md)

> AgentBase — Context Database for AI Agents
> Context DB = Memory + Resource + Skill + Temporal + Session + Observability

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Configuration Reference](#4-configuration-reference)
5. [Python SDK](#5-python-sdk)
6. [CLI Tools](#6-cli-tools)
7. [MCP Integration](#7-mcp-integration)
8. [Framework Adapters](#8-framework-adapters)
9. [Search & Retrieval](#9-search--retrieval)
10. [Knowledge Graph](#10-knowledge-graph)
11. [Session Management](#11-session-management)
12. [Observability](#12-observability)
13. [Maintenance](#13-maintenance)
14. [Production Deployment](#14-production-deployment)
15. [Error Handling](#15-error-handling)
16. [Performance Tuning](#16-performance-tuning)

---

## 1. Quick Start

```python
import asyncio
from agentbase import AgentBase

async def main():
    async with AgentBase(path="./my_agent.db") as db:
        # Write memory
        entry = await db.add_memory(
            "User prefers Python 3.12 and VS Code",
            category="preference",
            tags=["python", "ide"],
            confidence=0.95,
        )

        # Search context
        results = await db.find("Python preference", top_k=5)
        for r in results:
            print(f"[{r.score:.2f}] {r.entry.l2_full}")

        # Create session
        session = await db.create_session(agent_id="my-agent")
        await db.add_message(session.id, "user", "Help me write a Python script")
        await db.add_message(session.id, "assistant", "Sure, let me help...")

        # Commit session, compress and extract memories
        memories = await db.commit_session(session.id)

asyncio.run(main())
```

---

## 2. Installation

### 2.1 From Source (Recommended)

```bash
git clone <repo-url> agentbase
cd agentbase

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2.2 Package Structure

| Package | Description |
|---------|-------------|
| `agentbase-core` | Core engine, storage, index, retrieval |
| `agentbase-sdk` | Python SDK public interface |
| `agentbase-cli` | Command-line tools |
| `agentbase-mcp` | MCP protocol service |
| `agentbase-web` | Web dashboard & observability UI |

### 2.3 Dependencies

These dependencies are automatically installed with `agentbase-core`:

| Dependency | Purpose |
|------------|---------|
| `sqlite-vec` | Vector search engine (controlled by `vector_enabled`) |
| `pyyaml` | YAML config file I/O |
| `pydantic` / `pydantic-settings` | Data models and configuration |
| `aiosqlite` | Async SQLite driver |
| `python-ulid` | ULID ID generation |

> **Zero external database required**: Everything runs on SQLite. No Docker, no cloud services.

---

## 3. Core Concepts

### 3.1 Three Context Types

| Type | Purpose | Exclusive Fields |
|------|---------|-----------------|
| **Memory** | Memories, preferences, experiences | `memory_category` |
| **Resource** | Resources, documents, links | `resource_url`, `resource_format`, `resource_size` |
| **Skill** | Skills, tool definitions | `skill_tool_name`, `skill_api_spec` |

> Field constraint: Each type can only use its corresponding fields. Cross-type fields are rejected at write time.

### 3.2 Four-Level Scope

| Scope | Visibility | owner_id |
|-------|-----------|----------|
| `global` | Visible to all agents | Must be None |
| `project` | Visible within project | Must set (project ID) |
| `agent` | Visible to specific agent | Must set (agent ID) |
| `session` | Visible within session | Must set (session ID) |

### 3.3 Three-Layer Content

| Layer | Length | Purpose |
|-------|--------|---------|
| L0 Abstract | 20-50 chars | Quick filter, title display |
| L1 Overview | 100-300 chars | Summary, preview |
| L2 Full | Unlimited | Detailed content |

### 3.4 Six Memory Categories

`profile` | `preference` | `entity` | `event` | `case` | `pattern`

### 3.5 Feature Flags

| Feature | Config | Default | Notes |
|---------|--------|---------|-------|
| Vector Search | `config.index.vector_enabled` | `True` | Auto-degrades to FTS-only if embedder unavailable |
| Knowledge Graph | `config.graph.enabled` | `True` | CRUD always works; LLM extraction degrades gracefully |
| Session Management | `config.session.enabled` | `True` | Commit always works; LLM extraction degrades gracefully |
| Observability | `config.observability.enabled` | `False` | Enable for debugging/monitoring |

> AgentBase follows **graceful degradation**: if LLM is unavailable, features degrade to local rules instead of crashing.

---

## 4. Configuration Reference

### 4.1 YAML Configuration File

Create `agentbase.yaml`:

```yaml
data_dir: ~/.agentbase
db_filename: agentbase.db

embedding:
  model: text-embedding-3-small
  dimensions: 1536
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  max_concurrent: 10

llm:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1
  max_tokens: 1024

index:
  vector_enabled: true          # Enable FTS + vector hybrid search
  tokenizer: auto               # auto-detect language | jieba | char
  fts_weight: 0.5               # FTS BM25 weight
  vec_weight: 0.5               # Vector cosine weight
  rrf_k: 60                     # RRF smoothing constant
  dedup_threshold: 0.92         # Similarity threshold for dedup

graph:
  enabled: true
  max_traversal_depth: 4
  max_entities: 10000
  max_relations: 50000
  extract_on_ingest: false      # Auto-extract entities on ingest

session:
  enabled: true
  keep_recent_turns: 6
  auto_commit: false
  extract_memories: true
  extract_on_ingest: false      # Auto-create session on ingest_direct

ingest:
  session_summary: true         # LLM session summary (degrades to truncation)
  fact_extraction: true         # LLM fact extraction (degrades to local regex)
  ner_extraction: true          # NER entity extraction (bilingual)

retrieval:
  default_top_k: 20
  default_token_budget: 24000
  freshness_half_life_days: 7.0
  knowledge_update_half_life_days: 14.0
  query_decomposition: true     # Local rule-based query decomposition
  ner_boost: true               # NER-aware query expansion + result boosting
  ner_weight: 0.3               # NER signal weight in three-way fusion
  session_co_retrieval: true    # Session co-retrieval (D1)
  co_retrieve_min_turns: 2      # Min turns before co-retrieval activates
  agg_top_k: 120                # top_k for aggregation queries
  agg_detection: true           # Auto-detect aggregation queries + boost top_k

tier:
  enabled: true
  async_generation: true
  max_concurrent: 5
  fallback_to_truncation: true  # Generate L0/L1 by truncation when no LLM

observability:
  enabled: false
  persist_traces: true
  trace_sample_rate: 1.0
  max_trace_age_days: 30
```

### 4.2 Load from YAML

```python
from agentbase_core.models.config import AgentBaseConfig

# Load from file (env vars take precedence over YAML)
config = AgentBaseConfig.from_yaml("agentbase.yaml")

# Save config to file
config.to_yaml("agentbase.yaml")
```

### 4.3 Environment Variables

All config items support `AGENTBASE_` prefix with `__` for nesting:

```bash
# Basic
export AGENTBASE_DATA_DIR=/data/agentbase
export AGENTBASE_DB_FILENAME=agentbase.db

# Nested
export AGENTBASE_EMBEDDING__MODEL=text-embedding-3-small
export AGENTBASE_EMBEDDING__DIMENSIONS=1536
export AGENTBASE_EMBEDDING__API_KEY=sk-xxx
export AGENTBASE_LLM__MODEL=gpt-4o-mini
export AGENTBASE_LLM__API_KEY=sk-xxx

# Feature flags
export AGENTBASE_INDEX__VECTOR_ENABLED=true
export AGENTBASE_GRAPH__ENABLED=true
export AGENTBASE_SESSION__ENABLED=true
export AGENTBASE_OBSERVABILITY__ENABLED=true

# Retrieval tuning
export AGENTBASE_RETRIEVAL__NER_BOOST=true
export AGENTBASE_RETRIEVAL__NER_WEIGHT=0.3
export AGENTBASE_RETRIEVAL__SESSION_CO_RETRIEVAL=true
export AGENTBASE_RETRIEVAL__AGG_DETECTION=true
```

### 4.4 Configuration Priority

**Environment Variables > YAML File > Code Defaults**

### 4.5 Full Configuration Table

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| — | `data_dir` | Path | `~/.agentbase` | Data storage directory |
| — | `db_filename` | str | `agentbase.db` | SQLite database filename |
| embedding | `model` | str | `text-embedding-3-small` | Embedding model name |
| embedding | `dimensions` | int | `1536` | Vector dimensions |
| embedding | `api_base` | str? | None | API base URL |
| embedding | `api_key` | str? | None | API key |
| embedding | `max_concurrent` | int | `10` | Max concurrent embedding requests |
| llm | `model` | str | `gpt-4o-mini` | LLM model name |
| llm | `api_base` | str? | None | API base URL |
| llm | `api_key` | str? | None | API key |
| llm | `temperature` | float | `0.1` | Generation temperature |
| llm | `max_tokens` | int | `1024` | Max generation tokens |
| index | `vector_enabled` | bool | `true` | Enable vector search |
| index | `tokenizer` | str | `auto` | Tokenizer: auto/jieba/char |
| index | `fts_weight` | float | `0.5` | FTS BM25 weight |
| index | `vec_weight` | float | `0.5` | Vector cosine weight |
| index | `rrf_k` | int | `60` | RRF smoothing constant |
| index | `dedup_threshold` | float | `0.92` | Dedup similarity threshold |
| graph | `enabled` | bool | `true` | Enable knowledge graph |
| graph | `max_traversal_depth` | int | `4` | Max graph traversal depth |
| graph | `max_entities` | int | `10000` | Entity count limit |
| graph | `max_relations` | int | `50000` | Relation count limit |
| graph | `extract_on_ingest` | bool | `false` | Auto-extract on ingest |
| session | `enabled` | bool | `true` | Enable session management |
| session | `keep_recent_turns` | int | `6` | Recent turns to keep |
| session | `auto_commit` | bool | `false` | Auto-commit sessions |
| session | `extract_memories` | bool | `true` | Extract memories on commit |
| session | `extract_on_ingest` | bool | `false` | Auto-create session on ingest |
| ingest | `session_summary` | bool | `true` | Session summary generation |
| ingest | `fact_extraction` | bool | `true` | Fact extraction |
| ingest | `ner_extraction` | bool | `true` | NER entity extraction |
| ingest | `extract_on_direct_ingest` | bool | `false` | Extract on ingest_direct |
| retrieval | `default_top_k` | int | `20` | Default result count |
| retrieval | `default_token_budget` | int | `24000` | Default token budget |
| retrieval | `freshness_half_life_days` | float | `7.0` | Freshness half-life |
| retrieval | `knowledge_update_half_life_days` | float | `14.0` | Knowledge-update half-life |
| retrieval | `query_decomposition` | bool | `true` | Query decomposition |
| retrieval | `ner_boost` | bool | `true` | NER entity boosting |
| retrieval | `ner_weight` | float | `0.3` | NER boost weight |
| retrieval | `session_co_retrieval` | bool | `true` | Session co-retrieval |
| retrieval | `co_retrieve_min_turns` | int | `2` | Min turns for co-retrieval |
| retrieval | `agg_top_k` | int | `120` | Aggregation query top_k |
| retrieval | `agg_detection` | bool | `true` | Aggregation query detection |
| tier | `enabled` | bool | `true` | Enable L0/L1 generation |
| tier | `async_generation` | bool | `true` | Async layer generation |
| tier | `max_concurrent` | int | `5` | Max concurrent generation |
| tier | `fallback_to_truncation` | bool | `true` | Truncation fallback |
| observability | `enabled` | bool | `false` | Enable observability |
| observability | `persist_traces` | bool | `true` | Persist retrieval traces |
| observability | `trace_sample_rate` | float | `1.0` | Trace sampling rate |
| observability | `max_trace_age_days` | int | `30` | Max trace retention days |

---

## 5. Python SDK

### 5.1 Initialization

```python
from agentbase import AgentBase

# Method 1: Specify database path
db = AgentBase(path="./my_agent.db")
await db.initialize()

# Method 2: Use config object
from agentbase_core.models.config import AgentBaseConfig, GraphConfig, SessionConfig

config = AgentBaseConfig(
    data_dir=Path("/data/agentbase"),
    graph=GraphConfig(enabled=True),
    session=SessionConfig(enabled=True),
)
db = AgentBase(config=config)
await db.initialize()

# Method 3: Context manager (auto-init and close)
async with AgentBase(path="./my_agent.db") as db:
    # Use db ...
    pass
```

### 5.2 Write Operations

```python
# Add memory
entry = await db.add_memory(
    content="User prefers dark theme",
    category="preference",        # profile/preference/entity/event/case/pattern
    tags=["ui", "theme"],
    confidence=0.9,
    scope="agent",                # global/agent/project/session
    owner_id="agent-001",
)

# Add resource
entry = await db.add_resource(
    url="https://docs.python.org/3/",
    content="Python Official Documentation",
    format="html",
    tags=["python", "docs"],
    confidence=1.0,
    scope="global",
)

# Add skill
entry = await db.add_skill(
    tool_name="web_search",
    description="Search the internet for information",
    api_spec={"endpoint": "/search", "method": "GET"},
    tags=["search", "web"],
    confidence=1.0,
)
```

### 5.3 Read Operations

```python
# Get by ID
entry = await db.get(entry_id="01HX...")

# Simple search
results = await db.find("Python docs", top_k=5)

# Advanced search
from agentbase import SearchQuery, SearchStrategy, ContextType, EntryStatus

query = SearchQuery(
    text="Python tutorial",
    top_k=10,
    strategy="hybrid",              # fts/vector/hybrid/hierarchical
    context_type=ContextType.RESOURCE,
    scope="global",
    tags=["python"],
    min_confidence=0.7,
    token_budget=4000,
    load_level="auto",              # auto/l0/l1/l2
    include_trace=True,
    include_statuses=[EntryStatus.ACTIVE],
)
results = await db.search(query)

for r in results:
    print(f"Score: {r.score:.3f} | Stage: {r.ranking_stage} | Match: {r.matched_by}")
    print(f"  Degrade: {r.degrade_reason}")  # None/vec_unavailable/embedding_failed
    print(f"  Content: {r.entry.l2_full[:100]}")
```

### 5.4 Delete Operations

```python
# Soft delete (marked as deleted, still queryable)
await db.delete(entry_id="01HX...")

# Hard delete (physical deletion, irreversible)
await db.purge(entry_id="01HX...")
```

### 5.5 List & Count

```python
# List entries
entries = await db.list_entries(
    scope="agent",
    context_type="memory",
    limit=50,
    offset=0,
)

# Count entries
count = await db.count(scope="agent", context_type="memory")
```

### 5.6 Text Ingestion

```python
# Extract structured memories from raw text via LLM
entries = await db.ingest_text(
    text="User uses FastAPI framework in the project...",
    context_type="memory",
    scope="project",
    owner_id="project-001",
    tags=["framework"],
)
```

---

## 6. CLI Tools

### 6.1 Initialize

```bash
agentbase init --path ./my_agent.db
```

### 6.2 Add Entries

```bash
# Add memory
agentbase add "User prefers Python 3.12" --type memory --category preference --tags "python,version" --scope global

# Add resource
agentbase add "Python Official Docs" --type resource --tags "docs,python"

# Add skill
agentbase add "Web Search Tool" --type skill --tags "search"
```

### 6.3 Search

```bash
agentbase find "Python preference" --top-k 5 --type memory
```

### 6.4 Get Entry

```bash
agentbase get <entry_id>
```

### 6.5 Delete Entry

```bash
agentbase delete <entry_id>
```

### 6.6 Session Management

```bash
# Create session
agentbase session create --agent-id my-agent

# Add message
agentbase session add-message <session_id> --role user --content "Hello"

# Show session
agentbase session show <session_id>

# Commit session
agentbase session commit <session_id> --mode full
```

### 6.7 Entity Operations

```bash
# Add entity
agentbase entity add "Python" --type concept --description "Programming language"

# Find entity
agentbase entity find "Python"

# Add relation
agentbase entity relate <source_id> <target_id> --predicate "uses"

# Graph traversal
agentbase entity traverse "Python" --depth 2
```

### 6.8 Statistics

```bash
agentbase stats
```

### 6.9 Maintenance

```bash
# Rebuild index
agentbase reindex

# Clean up data
agentbase cleanup --traces-older-than 30 --deleted-older-than 7

# Debug: explain query
agentbase debug explain "Python preference"

# Debug: view retrieval trace
agentbase debug trace <trace_id>
```

---

## 7. MCP Integration

### 7.1 Available Tools

| Tool | Description | Required Params |
|------|-------------|---------------|
| `add_memory` | Add memory entry | `content` |
| `add_resource` | Add resource entry | `content` |
| `add_skill` | Add skill entry | `tool_name` |
| `find_context` | Search context | `query` |
| `get_context` | Get entry | `entry_id` |
| `delete_context` | Delete entry | `entry_id` |
| `add_entity` | Add entity | `name` |
| `find_entities` | Find entities | `name` |
| `add_relation` | Add relation | `source_id`, `target_id`, `predicate` |
| `graph_traverse` | Graph traversal | `entity_name` |
| `create_session` | Create session | (none) |
| `add_message` | Add message | `session_id`, `role`, `content` |
| `commit_session` | Commit session | `session_id` |
| `get_stats` | Get statistics | (none) |

### 7.2 Use with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agentbase": {
      "command": "python",
      "args": ["-m", "agentbase_mcp"],
      "env": {
        "AGENTBASE_DB_FILENAME": "agentbase.db",
        "AGENTBASE_GRAPH__ENABLED": "true",
        "AGENTBASE_SESSION__ENABLED": "true"
      }
    }
  }
}
```

---

## 8. Framework Adapters

AgentBase provides 5 adapters to serve as a unified memory backend for popular AI frameworks:

### 8.1 Mem0Adapter — Replace Mem0 with One Line

```python
from agentbase import Mem0Adapter

memory = Mem0Adapter(db_path="./agentbase.db")
await memory.add("I prefer dark mode", user_id="user-1")
results = await memory.search("theme preference", user_id="user-1")
all_memories = await memory.get_all(user_id="user-1")
```

### 8.2 LangChainMemoryAdapter — Duck-type BaseChatMemory

```python
from agentbase import LangChainMemoryAdapter

memory = LangChainMemoryAdapter(db_path="./agentbase.db", session_id="session-1")
await memory.save_context(
    inputs={"input": "What is Python?"},
    outputs={"output": "Python is a programming language."}
)
result = await memory.load_memory_variables({"input": "Python"})
```

### 8.3 AgentBaseChatStore — LlamaIndex BaseChatStore

```python
from agentbase import AgentBaseChatStore

store = AgentBaseChatStore(db_path="./agentbase.db")
from llama_index.core.llms import ChatMessage
store.set_messages("chat-key", [ChatMessage(role="user", content="Hello")])
messages = store.get_messages("chat-key")
```

### 8.4 OpenAIAssistantAdapter — Map Thread to Session

```python
from agentbase import OpenAIAssistantAdapter

adapter = OpenAIAssistantAdapter(db_path="./agentbase.db")
thread = await adapter.create_thread(metadata={"assistant_id": "asst-1"})
await adapter.add_message(thread.id, role="user", content="Help me code")
messages = await adapter.list_messages(thread.id)
```

### 8.5 MinimalAdapter — 3-Method API

```python
from agentbase import MinimalAdapter

memory = MinimalAdapter(db_path="./agentbase.db")
await memory.remember("User prefers Python 3.12")
results = await memory.recall("Python preference")
await memory.forget(entry_id="01HX...")
```

---

## 9. Search & Retrieval

### 9.1 Search Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `hybrid` | FTS + Vector + NER + RRF fusion | General search (default) |
| `fts` | Full-text search only | Exact keyword matching |
| `vector` | Vector search only | Semantic similarity |
| `hierarchical` | L0-L1-L2 progressive | Large-scale datasets |

### 9.2 Retrieval Pipeline

```
Query -> Normalization -> Intent Detection -> Strategy Routing
-> Three-Way Search (FTS + Vector + NER) -> RRF Fusion -> NER Boost
-> Heuristic Rerank -> LLM Rerank (optional) -> Layer Loading -> Results
```

### 9.3 Three-Way Hybrid Search (FTS + Vector + NER)

By default, AgentBase uses three-way hybrid search combining:

1. **FTS5 (BM25)**: Full-text keyword matching via SQLite FTS5
2. **sqlite-vec (cosine)**: Vector similarity search
3. **NER entity boosting**: Score amplification for entity-tagged results

All three are **enabled by default** with zero configuration:

```yaml
retrieval:
  ner_boost: true     # Default: true
  ner_weight: 0.3     # Default: 0.3
```

To disable NER boosting:

```yaml
retrieval:
  ner_boost: false
```

### 9.4 RRF Fusion Parameters

```yaml
index:
  fts_weight: 0.5     # FTS BM25 weight
  vec_weight: 0.5     # Vector similarity weight
  rrf_k: 60           # RRF smoothing constant
```

### 9.5 Heuristic Rerank Weights

The 5-dimensional heuristic rerank formula:

```
final_score = a(0.6)*rrf + b(0.15)*freshness + g(0.1)*confidence + d(0.1)*scope + e(0.05)*type
```

| Weight | Dimension | Description |
|--------|-----------|-------------|
| a (0.6) | RRF score | Original hybrid search score |
| b (0.15) | Freshness | Time decay (7-day half-life) |
| g (0.1) | Confidence | Original entry confidence |
| d (0.1) | Scope priority | session > agent > project > global |
| e (0.05) | Type match | Type matching bonus |

### 9.6 Query-Type-Aware Retrieval

AgentBase auto-detects 5 query types and applies specialized strategies:

| Query Type | Auto-Detection | Specialized Strategy |
|-----------|---------------|---------------------|
| `temporal-reasoning` | Date expressions | Auto-populates date_from/date_to filter |
| `knowledge-update` | Recency keywords | Dual half-life (7d / 14d), strong recency bias |
| `multi-session` | Cross-session keywords | Session completion + deduplication |
| `preference` | Preference indicators | User-role (1.8x) + pref-category (2.0x) boost |
| `aggregation` | "how many", "total" | Auto-boost top_k 20->120 |

### 9.7 load_level Auto Rules

| Condition | Level |
|-----------|-------|
| top_k > 20 | L0 |
| token_budget < 1000 | L0 |
| strategy = hierarchical | L1 |
| resource + l2_full > 500 chars | L1 |
| memory + l2_full < 200 chars | L2 |
| Default | L1 |

### 9.8 Vector Search Control

```yaml
index:
  vector_enabled: true   # Enable vector search
```

Or environment variable:

```bash
export AGENTBASE_INDEX__VECTOR_ENABLED=true
```

| State | Search Behavior | Result Marker |
|-------|----------------|-------------|
| `vector_enabled=false` | FTS BM25 keyword matching only | `degrade_reason = "vec_unavailable"` |
| `vector_enabled=true` | FTS + Vector cosine + RRF fusion | `degrade_reason = None` |

---

## 10. Knowledge Graph

> Requires: `config.graph.enabled = True` (default)

### 10.1 Core Models

**Entity**

| Field | Type | Description |
|-------|------|-------------|
| name | str | Entity name |
| entity_type | str | person/project/concept/tool/event/organization |
| description | str | Description |
| properties | dict | Extended properties |

**Relation**

| Field | Type | Description |
|-------|------|-------------|
| source_id | str | Source entity ID |
| target_id | str | Target entity ID |
| predicate | str | Relation predicate |
| confidence | float | Confidence |
| valid_until | datetime? | Expiry time (None = currently valid) |

### 10.2 SDK Operations

```python
entity = await db.add_entity("Python", entity_type="concept", description="Programming language")
await db.add_alias(entity.id, "Python3")
entities = await db.find_entities("Python")
await db.add_relation(source_id=ent1.id, target_id=ent2.id, predicate="uses")
relations = await db.get_current_relations(entity.id)
paths = await db.graph_traversal("Python", depth=2)
await db.add_fact(entity.id, fact="Python 3.12 released in 2023")
facts = await db.get_current_facts(entity.id)
```

---

## 11. Session Management

> Requires: `config.session.enabled = True` (default)

### 11.1 Session Lifecycle

```
Create -> Add Messages -> Commit -> Archive
```

### 11.2 Commit Modes

| Mode | Description |
|------|-------------|
| `full` | Compression + memory extraction (default) |
| `archive_only` | Compression only |
| `extract_only` | Memory extraction only |

### 11.3 SDK Operations

```python
session = await db.create_session(agent_id="my-agent", project="project-001")
await db.add_message(session.id, "user", "Help me write a crawler")
await db.add_message(session.id, "assistant", "Sure, let me write one...")
session = await db.get_session(session.id, load_messages=True)
memories = await db.commit_session(session.id, mode="full")
```

### 11.4 Session Compression

- Keeps the most recent N turns (`keep_recent_turns=6`)
- Older turns are archived as L0/L1 summaries
- Requires LLM for summary generation; uses truncation fallback when LLM is unavailable

---

## 12. Observability

> Requires: `config.observability.enabled = True`

### 12.1 Three Components

| Component | Description |
|-----------|-------------|
| TraceCollector | Retrieval trace collection (configurable sampling rate) |
| ContextMetrics | Quality metrics (query count, latency, P50, etc.) |
| DebugService | Debug tools (query explanation, diff comparison, etc.) |

### 12.2 SDK Operations

```python
metrics = await db.get_metrics()
explanation = await db.explain_query("Python tutorial")
```

### 12.3 DebugService Methods

```python
from agentbase_core.observability.observability_service import DebugService
svc = DebugService(pool)
trace = await svc.get_trace(trace_id="...")
traces = await svc.list_recent_traces(limit=10)
explanation = await svc.explain_query("Python tutorial")
diff = await svc.diff_contexts(id1="...", id2="...")
session_traces = await svc.trace_session(session_id="...")
graph = await svc.entity_graph("Python", depth=2)
```

### 12.4 Web Dashboard

AgentBase includes a web dashboard for visual observability:

- **Timeline**: Chronological view of all context entries
- **Heatmap**: Entry density over time
- **Category Sunburst**: Hierarchical category distribution
- **Freshness Distribution**: Entry age distribution
- **Tag Cloud**: Most frequent tags
- **Activity Feed**: Real-time entry changes

---

## 13. Maintenance

### 13.1 Rebuild Index

```python
result = await engine.reindex()
```

### 13.2 Data Cleanup

```python
result = await engine.cleanup(
    traces_older_than_days=30,
    deleted_older_than_days=7,
    failed_jobs_older_than_days=14,
)
```

### 13.3 VACUUM

```python
await engine.vacuum()
```

### 13.4 Background Job Management

```python
runner = engine.job_runner
jobs = await runner.list_jobs()
await runner.retry_failed()
await runner.resume_pending()
```

---

## 14. Production Deployment

### 14.1 Recommended Configuration

```yaml
data_dir: /data/agentbase
db_filename: agentbase.db

embedding:
  model: text-embedding-3-small
  dimensions: 1536
  api_key: ${OPENAI_API_KEY}
  max_concurrent: 10

llm:
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1
  max_tokens: 1024

index:
  vector_enabled: true
  fts_weight: 0.5
  vec_weight: 0.5
  rrf_k: 60
  dedup_threshold: 0.92

graph:
  enabled: true
  max_traversal_depth: 4

session:
  enabled: true
  keep_recent_turns: 6

retrieval:
  ner_boost: true
  ner_weight: 0.3
  session_co_retrieval: true
  agg_detection: true

observability:
  enabled: true
  trace_sample_rate: 0.1
  max_trace_age_days: 7
```

### 14.2 SQLite Tuning

AgentBase uses WAL mode by default. For high-concurrency scenarios, the ConnectionPool manages read/write connections automatically.

### 14.3 Scheduled Maintenance

```bash
# Daily cleanup
0 3 * * * agentbase cleanup --traces-older-than 7 --deleted-older-than 7 --path /data/agentbase/agentbase.db

# Weekly reindex
0 4 * * 0 agentbase reindex --path /data/agentbase/agentbase.db
```

### 14.4 Database Backup

```bash
sqlite3 /data/agentbase/agentbase.db ".backup /backup/agentbase_$(date +%Y%m%d).db"
```

### 14.5 Custom LLM/Embedder

```python
from agentbase_core.llm.base import AbstractLLM
from agentbase_core.embedding.base import AbstractEmbedder

class MyLLM(AbstractLLM):
    async def complete(self, prompt: str, **kwargs) -> str: ...
    async def complete_json(self, prompt: str, **kwargs) -> dict | list: ...

class MyEmbedder(AbstractEmbedder):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

engine = AgentBaseEngine(config=config, llm=MyLLM(), embedder=MyEmbedder())
```

### 14.6 Multi-Agent Architecture

```python
await db.add_memory("Project uses FastAPI", scope="global")
await db.add_memory("API prefix is /api/v2", scope="project", owner_id="project-001")
await db.add_memory("My role is code review", scope="agent", owner_id="reviewer-agent")
results = await db.find("API config", scope="agent", owner_id="reviewer-agent")
```

---

## 15. Error Handling

### 15.1 Exception Hierarchy

```
AgentBaseError
+-- StorageError        # SQLite operation failed
+-- IndexOpError        # Index operation failed (FTS5/sqlite-vec)
+-- EmbeddingError      # Embedding generation failed
+-- LLMError            # LLM call failed
+-- GraphError          # Graph operation failed
+-- SessionError        # Session operation failed
+-- ConfigError         # Configuration error
+-- ConflictError       # Fact conflict requires manual intervention
+-- BackgroundJobError  # Background job execution failed
+-- ValidationError     # Input validation error
```

### 15.2 Common Error Handling

```python
from agentbase_core.exceptions import AgentBaseError, StorageError, ConfigError, ValidationError

try:
    result = await db.find("query")
except ConfigError as e:
    print(f"Feature needs enabling: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except StorageError as e:
    print(f"Storage error: {e}")
except AgentBaseError as e:
    print(f"Error: {e}")
```

---

## 16. Performance Tuning

### 16.1 Index Weight Adjustment

```yaml
index:
  fts_weight: 0.3    # Lower FTS weight (keyword matching less important)
  vec_weight: 0.7    # Higher vector weight (semantic search matters more)
  rrf_k: 60          # Default, rarely needs adjustment
```

### 16.2 Dedup Threshold

```yaml
index:
  dedup_threshold: 0.92   # Similarity > 0.92 = duplicate
                            # Lower = stricter dedup
                            # Higher = more lenient dedup
```

### 16.3 Layer Generation

```yaml
tier:
  enabled: true
  async_generation: true
  max_concurrent: 5
  fallback_to_truncation: true
```

### 16.4 Session Turn Retention

```yaml
session:
  keep_recent_turns: 6    # Keep last 6 turns
```

### 16.5 Observability Sampling

```yaml
observability:
  trace_sample_rate: 0.1   # Production: 0.01-0.1, Dev/Debug: 1.0
```

### 16.6 Embedding Cache

AgentBase automatically caches computed embedding vectors (`embedding_cache` table). Identical content is never recomputed.

---

> **AgentBase** — Give AI agents persistent, searchable, and evolvable context memory.