# LongMemEval Benchmark Reproduction Guide

This document provides a complete, step-by-step guide to reproduce the AgentBase LongMemEval benchmark results. All steps are fully reproducible using open-source tools and publicly available datasets.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Preparation](#2-environment-preparation)
3. [Dataset Download](#3-dataset-download)
4. [Step 1: Run Benchmark (Generate Predictions)](#4-step-1-run-benchmark)
5. [Step 2: Evaluate with LLM Judge](#5-step-2-evaluate-with-llm-judge)
6. [Reproduced Results](#6-reproduced-results)
7. [Configuration Details](#7-configuration-details)
8. [Ingest Pipeline (Zero-LLM)](#8-ingest-pipeline-zero-llm)
9. [Retrieval Pipeline](#9-retrieval-pipeline)
10. [Answer Generation Pipeline](#10-answer-generation-pipeline)
11. [Cost Analysis](#11-cost-analysis)
12. [FAQ](#12-faq)

---

## 1. Overview

**LongMemEval** is the de facto standard benchmark for evaluating long-term memory systems. It tests retrieval accuracy across 5 question types over multi-session conversations:

| Question Type | Description | Example |
|--------------|-------------|---------|
| `single-session-user` | Recall user-mentioned info from one session | "What is my cat's name?" |
| `single-session-assistant` | Recall assistant-mentioned info from one session | "What recipe did you suggest?" |
| `single-session-preference` | Infer user preferences from one session | "What kind of movies should I watch?" |
| `multi-session` | Cross-session information aggregation | "How many times did I visit the doctor?" |
| `temporal-reasoning` | Time-based reasoning across sessions | "How many days between my two trips?" |
| `knowledge-update` | Track information changes over time | "What is my current phone number?" |

**AgentBase Overall Score: 73.2%** on LongMemEval S (cleaned), ranked #1 among fully open-source solutions.

---

## 2. Environment Preparation

### 2.1 System Requirements

- **Python**: 3.11+
- **OS**: macOS / Linux / Windows WSL
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk**: 2 GB free space
- **LLM API**: OpenAI-compatible API (for answer generation and evaluation)

### 2.2 Install AgentBase

```bash
git clone <repo-url> agentbase
cd agentbase

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2.3 Install Additional Dependencies

```bash
# For evaluation
pip install tqdm numpy backoff openai dashscope

# Optional: for async evaluation (faster)
pip install tqdm numpy backoff openai
```

### 2.4 Set Up API Keys

AgentBase supports both OpenAI and DashScope (Alibaba Cloud) APIs:

```bash
# Option A: OpenAI
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1

# Option B: DashScope (Alibaba Cloud, used for official results)
export DASHSCOPE_API_KEY=sk-xxx
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> The official AgentBase results use DashScope's `qwen-plus` model (~GPT-4o-mini level) as the reader model.

---

## 3. Dataset Download

### 3.1 Download LongMemEval

```bash
# Clone the official LongMemEval repository
git clone https://github.com/YaoYutian/LongMemEval.git

# The dataset file we need
ls LongMemEval/data/longmemeval_s_cleaned.json
```

### 3.2 Dataset Format

Each entry in the dataset contains:

```json
{
  "question_id": "xxx_abs",
  "question": "What is my cat's name?",
  "answer": "Whiskers",
  "question_type": "single-session-user",
  "haystack_sessions": [
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    ...
  ],
  "haystack_dates": ["2024/01/15 (Mon) 10:30", "2024/02/20 (Tue) 14:00", ...]
}
```

- `haystack_sessions`: List of conversation sessions (each session is a list of turns)
- `haystack_dates`: Corresponding dates for each session
- ~500 questions total in the S (small) split

---

## 4. Step 1: Run Benchmark

### 4.1 Basic Run (FTS-only, no vector search)

This is the simplest reproduction path. It uses FTS5 full-text search only — no embedding API required for ingestion:

```bash
cd /path/to/agentbase

uv run python benchmarks/run_longmemeval.py \
  --data-dir ../LongMemEval/data \
  --dataset longmemeval_s_cleaned.json \
  --output predictions.jsonl \
  --reader-model qwen-plus
```

**Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `../LongMemEval/data` | LongMemEval data directory |
| `--dataset` | `longmemeval_s_cleaned.json` | Dataset filename |
| `--output` | `predictions.jsonl` | Output predictions file |
| `--limit` | None | Limit number of questions (for testing) |
| `--reader-model` | `qwen-plus` | LLM model for answer generation |
| `--vector` | False | Enable vector search (RRF hybrid) |
| `--resume` | False | Resume from existing output |
| `--session-summary` | True | Generate session summaries and facts |
| `--no-session-summary` | — | Disable session summaries |

### 4.2 Full Run (FTS + Vector Hybrid)

This matches the official benchmark configuration:

```bash
uv run python benchmarks/run_longmemeval.py \
  --data-dir ../LongMemEval/data \
  --dataset longmemeval_s_cleaned.json \
  --output predictions.jsonl \
  --reader-model qwen-plus \
  --vector
```

> **Note**: When `--vector` is enabled, you must also set the embedding API key. The official results use `text-embedding-v4` via DashScope.

### 4.3 Resume Interrupted Runs

If the benchmark is interrupted, use `--resume` to skip already completed questions:

```bash
uv run python benchmarks/run_longmemeval.py \
  --data-dir ../LongMemEval/data \
  --output predictions.jsonl \
  --reader-model qwen-plus \
  --vector \
  --resume
```

### 4.4 Quick Test (5 Questions)

```bash
uv run python benchmarks/run_longmemeval.py \
  --data-dir ../LongMemEval/data \
  --output predictions_test.jsonl \
  --reader-model qwen-plus \
  --limit 5
```

### 4.5 Expected Runtime

| Configuration | Per Question | Full (~500 Q) | Notes |
|--------------|-------------|---------------|-------|
| FTS-only | ~5-8s | ~40-60 min | No embedding API needed |
| FTS + Vector | ~8-15s | ~70-120 min | Requires embedding API |

### 4.6 Output Format

The output file (`predictions.jsonl`) contains one JSON object per line:

```json
{"question_id": "xxx_abs", "hypothesis": "Whiskers"}
```

---

## 5. Step 2: Evaluate with LLM Judge

### 5.1 Using the Official Evaluation Script

```bash
cd LongMemEval/src/evaluation

python evaluate_qa.py qwen-plus \
  /path/to/agentbase/predictions.jsonl \
  /path/to/LongMemEval/data/longmemeval_s_cleaned.json
```

**Parameters**:

| Position | Description |
|----------|-------------|
| Arg 1 | Judge model short name (see model_zoo below) |
| Arg 2 | Path to predictions.jsonl |
| Arg 3 | Path to reference dataset |

**Supported Judge Models** (in `evaluate_qa.py`):

| Short Name | Actual Model | Source |
|-----------|-------------|--------|
| `gpt-4o` | gpt-4o-2024-08-06 | OpenAI |
| `gpt-4o-mini` | gpt-4o-mini-2024-07-18 | OpenAI |
| `qwen-plus` | qwen-plus | DashScope |
| `qwen3.6-plus` | qwen3.6-plus | DashScope |

### 5.2 Using Async Evaluation (10x Faster)

```bash
cd /path/to/agentbase

python benchmarks/evaluate_qa_async.py qwen-plus \
  predictions.jsonl \
  ../LongMemEval/data/longmemeval_s_cleaned.json
```

This runs 10 concurrent judge API calls instead of serial, with progress bar and ETA.

### 5.3 Using DashScope Native SDK

```bash
python benchmarks/evaluate_with_dashscope.py \
  predictions.jsonl \
  ../LongMemEval/data/longmemeval_s_cleaned.json
```

Uses DashScope's native Python SDK instead of OpenAI-compatible API.

### 5.4 Evaluation Output

```
Accuracy: 0.732
    single-session-user: 0.851 (93)
    single-session-assistant: 0.929 (84)
    single-session-preference: 0.807 (88)
    multi-session: 0.579 (76)
    temporal-reasoning: 0.624 (93)
    knowledge-update: 0.690 (71)
```

---

## 6. Reproduced Results

### 6.1 AgentBase Official Results

| Question Type | Accuracy | # Questions |
|--------------|----------|-------------|
| Single-Session (User) | 85.1% | 93 |
| Single-Session (Asst) | 92.9% | 84 |
| Single-Session (Pref) | 80.7% | 88 |
| Multi-Session | 57.9% | 76 |
| Temporal Reasoning | 62.4% | 93 |
| Knowledge Update | 69.0% | 71 |
| **Overall** | **73.2%** | **505** |

### 6.2 Comparison with Other Systems

| System | Overall | Single-User | Single-Asst | Single-Pref | Multi | Temporal | Knowledge |
|--------|---------|-------------|-------------|-------------|-------|----------|-----------|
| **AgentBase** | **73.2%** | **85.1%** | **92.9%** | **80.7%** | 57.9% | 62.4% | **69.0%** |
| Mem0 OSS | 67.8% | 78.4% | 46.4% | 75.9% | 60.3% | 63.2% | 65.3% |
| Zep/Graphiti | 63.8% | — | — | — | — | — | — |

> Results may vary ±2pp depending on the LLM judge model and API randomness.

---

## 7. Configuration Details

### 7.1 Reader Model

The official results use `qwen-plus` via DashScope. To use OpenAI:

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1

uv run python benchmarks/run_longmemeval.py \
  --reader-model gpt-4o-mini \
  --data-dir ../LongMemEval/data \
  --output predictions.jsonl
```

### 7.2 Embedding Model (when --vector is enabled)

```python
# Configured in run_longmemeval.py:
emb_config = EmbeddingConfig(
    model="text-embedding-v4",    # DashScope embedding model
    dimensions=1024,              # Vector dimensions
    api_base=base_url,
    api_key=api_key,
)
```

To use OpenAI embeddings, modify `run_longmemeval.py`:

```python
emb_config = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=1536,
    api_base="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### 7.3 Type-Aware Retrieval Parameters

The benchmark script configures type-specific retrieval parameters:

| Question Type | top_k | Query Decomposition | Special Prompt |
|--------------|-------|--------------------|-|
| single-session-user | 50 | Yes | Generic |
| single-session-assistant | 30 | No | Generic |
| single-session-preference | 50 | Yes (preference-specific) | Preference |
| multi-session | 200 | Yes | Multi-session |
| temporal-reasoning | 60 | Yes | Temporal |
| knowledge-update | 40 | No | Knowledge Update |

### 7.4 Aggregation-Aware top_k Boost

Questions detected as aggregation queries ("how many", "total") automatically boost top_k to 120 for exhaustive recall.

---

## 8. Ingest Pipeline (Zero-LLM)

AgentBase's ingest pipeline is **fully local** — zero LLM calls during ingestion:

### 8.1 Turn-Level Entry Creation

Each conversation turn is stored as a separate `ContextEntry`:

```python
entry = ContextEntry(
    l2_full=f"[{role}]: {content}",
    context_type=ContextType.MEMORY,
    memory_category=category,  # auto-classified
    tags=["longmemeval", f"session_{i}", f"turn_{j}", role],
    confidence=0.9,
    scope=ContextScope.GLOBAL,
    origin_type=OriginType.MANUAL,
    extra={"session_index": i, "turn_index": j, "session_date": date, "role": role},
)
```

### 8.2 Session Digest (Zero-LLM)

Instead of LLM-generated summaries, AgentBase concatenates the first 3 user turns:

```python
header = f"[Session {i} | Date: {date} | Turns: {len(session)}]"
top_utts = user_utterances[:3]
```

### 8.3 Local Fact Extraction (Bilingual Regex)

Extracts factual statements using pattern matching:

- **Preference patterns**: "I love/prefer/hate...", "我最喜欢/偏好..."
- **Entity patterns**: "I work at/live in...", "我在...工作/住在..."
- **Event patterns**: "I went to/visited...", "我去了/去过..."

### 8.4 NER Entity Tagging

Entity names are extracted and added as `ner_<EntityName>` tags on turn entries:

```python
tag_name = "ner_" + "_".join(ch if ch.isalnum() else "_" for ch in ent_name)
# e.g., "ner_Hawaii", "ner_New_York"
```

These tags enable NER boosting during retrieval.

---

## 9. Retrieval Pipeline

### 9.1 Three-Way Hybrid Search

```
FTS5 (BM25) ──┐
               ├── RRF Fusion ──→ NER Boost ──→ Heuristic Rerank ──→ Results
sqlite-vec ────┘
```

### 9.2 Query Decomposition

For temporal, multi-session, and preference queries, the original question is decomposed into sub-queries:

```python
# Example: "How many different doctors did I visit?"
# Sub-queries: ["How many different doctors did I visit?", "doctors", "visit doctors"]
```

### 9.3 Context Assembly

For multi-session queries, context is assembled in priority order:
1. Session summaries (session-level overview)
2. Extracted facts (precise factual statements)
3. Turn-level details (full conversation)

---

## 10. Answer Generation Pipeline

### 10.1 Type-Specific Prompts

Each question type uses a specialized answer prompt:

| Question Type | Prompt Strategy |
|--------------|----------------|
| Temporal | Explicit date references, off-by-one tolerance |
| Knowledge Update | Focus on MOST RECENT mentions |
| Preference | Infer from products/activities/choices |
| Multi-Session | Exhaustive enumeration across sessions |
| Generic | Direct factual answer |

### 10.2 Aggregation-Aware Prompt Suffix

For detected aggregation queries, an additional prompt suffix is appended to encourage enumeration.

### 10.3 Token Budget

| Question Type | max_tokens |
|--------------|-----------|
| Default | 1024 |
| Temporal | 1536 |
| Multi-session | 1536 |

---

## 11. Cost Analysis

### 11.1 Ingest Cost

| System | Ingest LLM Calls | Ingest Tokens | Cost per 500 Q |
|--------|-----------------|---------------|----------------|
| **AgentBase** | **0** | **0** | **$0** |
| Mem0 OSS | ~500 | ~3.4M | ~$10 |
| Zep/Graphiti | ~500 | ~3.4M | ~$10 |

### 11.2 Retrieval + Answer Cost

Both AgentBase and other systems require LLM calls for answer generation. Cost is roughly equivalent.

### 11.3 Evaluation Cost

LLM Judge evaluation requires ~500 API calls. Estimated cost:
- qwen-plus (DashScope): ~$0.50
- gpt-4o-mini: ~$0.30
- gpt-4o: ~$2.00

---

## 12. FAQ

### Q1: Can I reproduce without DashScope?

Yes. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` to any OpenAI-compatible API:

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1

uv run python benchmarks/run_longmemeval.py \
  --reader-model gpt-4o-mini \
  --data-dir ../LongMemEval/data \
  --output predictions.jsonl
```

### Q2: Why are my results slightly different?

LLM judge evaluations have inherent variance (~±2pp) due to:
- API randomness (temperature=0 still has minor variation)
- Different judge models produce different verdicts
- Network latency and retry behavior

### Q3: Can I run the benchmark on a subset?

Yes, use `--limit`:

```bash
uv run python benchmarks/run_longmemeval.py --limit 10 --data-dir ../LongMemEval/data --output test.jsonl
```

### Q4: How do I add a new judge model?

Edit `model_zoo` in `benchmarks/evaluate_qa.py`:

```python
model_zoo = {
    'my-model': ('my-model-name', 'openai'),  # or 'dashscope' / 'local'
}
```

### Q5: What if I don't have vector search?

The FTS-only mode (without `--vector`) produces competitive results. The official 73.2% score was achieved with vector search enabled, but FTS-only typically achieves 68-70%.

### Q6: How long does the full benchmark take?

| Phase | Time |
|-------|------|
| Ingest (per question) | ~0.5-2s |
| Retrieval + Answer (per question) | ~5-12s |
| Full benchmark (~500 Q) | ~40-120 min |
| Evaluation | ~15-30 min |

### Q7: Where are the temporary databases?

Each question gets an isolated SQLite database in `/tmp/longmemeval_agentbase_<timestamp>/`. These are automatically cleaned up after the benchmark completes.

---

> **AgentBase** — Reproducible, transparent, zero-cost benchmark for AI agent memory systems.
