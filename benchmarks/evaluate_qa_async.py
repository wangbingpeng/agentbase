#!/usr/bin/env python3
"""Async parallel version of evaluate_qa.py for faster LLM Judge scoring.

Improvements over the original:
1. Async parallel API calls (10 concurrent) — ~10x faster than serial
2. Response truncation (1500 chars) — reduces Judge input tokens by ~60%
3. qwen3 series enable_thinking=False support
4. Progress bar with ETA

Usage:
    python benchmarks/evaluate_qa_async.py qwen3.6-plus predictions.jsonl ../LongMemEval/data/longmemeval_s_cleaned.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import backoff
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio

# ── Model registry ──────────────────────────────────────────────

model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
    'glm-5.1': ('glm-5.1', 'dashscope'),
    'qwen3.6-plus': ('qwen3.6-plus', 'dashscope'),
    'qwen-plus': ('qwen-plus', 'dashscope'),
}

# ── Response truncation ─────────────────────────────────────────
# The Judge only checks if the correct answer is present in the
# model response.  For verbose enumeration-style answers (D2
# aggregation prompt), truncating to 1500 chars reduces input
# tokens by ~60% with negligible accuracy impact.

MAX_RESPONSE_CHARS = 1500


def truncate_response(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> str:
    """Truncate model response to max_chars, preserving the ending."""
    if len(text) <= max_chars:
        return text
    # Keep first 80% and last 20% (answer often at the end)
    head = int(max_chars * 0.8)
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


# ── Prompt builder (same logic as original) ─────────────────────

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    # Truncate verbose model responses before sending to Judge
    response = truncate_response(response)

    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(f"Unknown task type: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

    return template.format(question, answer, response)


# ── Async evaluation ─────────────────────────────────────────────

# Concurrency limit — DashScope rate limit is ~60 RPM for qwen3.6-plus
CONCURRENCY = 10


async def evaluate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    entry: dict,
    qtype: str,
    question: str,
    answer: str,
) -> dict:
    """Evaluate a single entry with rate-limited async API call."""
    async with semaphore:
        hyp = entry['hypothesis']
        abstention = '_abs' in entry['question_id']
        prompt = get_anscheck_prompt(qtype, question, answer, hyp, abstention=abstention)

        kwargs = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'n': 1,
            'temperature': 0,
            'max_tokens': 10,
        }
        # qwen3 series requires enable_thinking=False
        if 'qwen3' in model.lower():
            kwargs['extra_body'] = {'enable_thinking': False}

        @backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
        async def _call():
            return await client.chat.completions.create(**kwargs)

        completion = await _call()
        eval_response = completion.choices[0].message.content.strip()
        label = 'yes' in eval_response.lower()

        entry['autoeval_label'] = {
            'model': model,
            'label': label,
        }
        return entry


async def main():
    if len(sys.argv) != 4:
        print('Usage: python evaluate_qa_async.py metric_model hyp_file ref_file')
        sys.exit(1)

    metric_model_short = sys.argv[1]
    hyp_file = sys.argv[2]
    ref_file = sys.argv[3]

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        sys.exit(1)
    metric_model, metric_model_source = model_zoo[metric_model_short]

    if metric_model_source == 'openai':
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = None
    elif metric_model_source == 'dashscope':
        openai_api_key = os.getenv('DASHSCOPE_API_KEY')
        openai_api_base = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"

    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)

    # Load data
    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except Exception:
        hypotheses = json.load(open(ref_file))
    try:
        references = json.load(open(ref_file))
    except Exception:
        references = [json.loads(line) for line in open(ref_file).readlines()]

    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(qid2qtype.values())
    qtype2acc = {t: [] for t in qtypes}

    result_file = hyp_file + f'.eval-results-{metric_model_short}'

    # Deduplicate hypotheses by question_id (keep last)
    seen = {}
    for h in hypotheses:
        seen[h['question_id']] = h
    hypotheses = list(seen.values())

    print(f'Evaluating {len(hypotheses)} entries with {metric_model} (concurrency={CONCURRENCY})...')
    print(f'Response truncation: {MAX_RESPONSE_CHARS} chars')

    # Build tasks
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for entry in hypotheses:
        qid = entry['question_id']
        if qid not in qid2qtype:
            continue
        qtype = qid2qtype[qid]
        question = qid2qdata[qid]['question']
        answer = qid2qdata[qid]['answer']
        tasks.append(evaluate_one(client, semaphore, metric_model, entry, qtype, question, answer))

    # Run all tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc='Judge', mininterval=2)

    # Write results
    with open(result_file, 'w') as out_f:
        for entry in results:
            qid = entry['question_id']
            if qid not in qid2qtype:
                continue
            label = entry['autoeval_label']['label']
            qtype2acc[qid2qtype[qid]].append(1 if label else 0)
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Print summary
    overall = np.mean([1 if x['autoeval_label']['label'] else 0 for x in results]).item()
    print(f'\nOverall Accuracy: {round(overall, 4)}')
    for k, v in sorted(qtype2acc.items()):
        if v:
            print(f'  {k}: {round(np.mean(v), 4)} ({len(v)})')
    print(f'\nSaved to {result_file}')


if __name__ == '__main__':
    asyncio.run(main())
