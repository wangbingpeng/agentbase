# AgentBase Benchmarks

This directory contains benchmark scripts for evaluating AgentBase against public long-context memory benchmarks.

## Available Benchmarks

| Script | Benchmark | Description |
| --- | --- | --- |
| `run_longmemeval.py` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 500-question long-term memory evaluation |
| `run_locomo.py` | [LoCoMo](https://github.com/snap-stanford/locomo) | Long conversation memory evaluation |
| `evaluate_qa.py` | Synchronous LLM judge | Score predictions against gold answers |
| `evaluate_qa_async.py` | Async LLM judge | High-throughput version of `evaluate_qa.py` |
| `evaluate_with_dashscope.py` | DashScope judge | LLM judge via Alibaba Cloud DashScope API |

## Reproduction Guide

For a full step-by-step reproduction guide of the LongMemEval results, see:

- English: [../docs/USAGE_EN.md](../docs/USAGE_EN.md)
- Reproduction: [../docs/BENCHMARK.md](../docs/BENCHMARK.md)

## Prerequisites

```bash
pip install tqdm numpy backoff openai dashscope

# Configure API credentials (choose one)
export OPENAI_API_KEY=sk-xxx
# or
export DASHSCOPE_API_KEY=sk-xxx
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Quick Start (LongMemEval)

```bash
# 1. Run benchmark
python benchmarks/run_longmemeval.py \
  --data-path /path/to/LongMemEval/data/longmemeval_s.json \
  --out predictions.jsonl \
  --reader-model qwen-plus

# 2. Evaluate with LLM judge
python benchmarks/evaluate_qa.py qwen-plus predictions.jsonl \
  /path/to/LongMemEval/data/longmemeval_s.json
```

See [../docs/BENCHMARK.md](../docs/BENCHMARK.md) for complete instructions.
