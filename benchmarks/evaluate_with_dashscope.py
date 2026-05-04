#!/usr/bin/env python3
"""Evaluate LongMemEval predictions using DashScope native SDK (qwen-plus as judge)."""
import os
import sys
import json
import time
from tqdm import tqdm
import dashscope
from dashscope import Generation
import numpy as np

JUDGE_MODEL = "qwen-plus"


def call_judge(prompt, max_retries=5):
    """Call DashScope Generation API with retry."""
    for attempt in range(max_retries):
        try:
            r = Generation.call(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
                result_format="message",
            )
            if r.status_code == 200:
                return r.output.choices[0].message.content.strip()
            else:
                print(f"  API error: {r.code} - {r.message}, retry {attempt+1}/{max_retries}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Exception: {e}, retry {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt)
    return "error"


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            return template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unknown task: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        return template.format(question, answer, response)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python evaluate_with_dashscope.py hyp_file ref_file')
        exit()

    hyp_file = sys.argv[1]
    ref_file = sys.argv[2]
    verbose = True

    result_file = hyp_file + '.eval-results-dashscope-' + JUDGE_MODEL

    api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
    dashscope.api_key = api_key

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except Exception:
        hypotheses = json.load(open(hyp_file))
    try:
        references = json.load(open(ref_file))
    except Exception:
        references = [json.loads(line) for line in open(ref_file).readlines()]

    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}

    # Resume from existing results if any
    existing_qids = set()
    logs = []
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                entry = json.loads(line)
                existing_qids.add(entry['question_id'])
                if 'autoeval_label' in entry:
                    logs.append(entry)
                    qtype2acc[qid2qtype[entry['question_id']]].append(
                        1 if entry['autoeval_label']['label'] else 0
                    )
        print(f"Resuming from {len(existing_qids)} existing results")

    with open(result_file, 'a') as out_f:
        for entry in tqdm(hypotheses, desc="Evaluating"):
            if entry['question_id'] in existing_qids:
                continue
            if entry['question_id'] not in qid2qtype:
                print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
                continue

            qtype = qid2qtype[entry['question_id']]
            q = qid2qdata[entry['question_id']]['question']
            ans = qid2qdata[entry['question_id']]['answer']
            hyp = entry['hypothesis']

            prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
            eval_response = call_judge(prompt)
            label = 'yes' in eval_response.lower()
            entry['autoeval_label'] = {
                'model': JUDGE_MODEL,
                'label': label,
            }
            logs.append(entry)
            if verbose:
                print(json.dumps({
                    'qtype': qtype,
                    'label': label,
                    'judge': eval_response,
                }, ensure_ascii=False), flush=True)
            print(json.dumps(entry, ensure_ascii=False), file=out_f)
            qtype2acc[qid2qtype[entry['question_id']]].append(1 if label else 0)

    print('=' * 60)
    print('Judge model:', JUDGE_MODEL)
    overall = round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4)
    print('Overall Accuracy:', overall)
    for k, v in sorted(qtype2acc.items()):
        acc = round(np.mean(v), 4) if v else 0
        print(f'  {k}: {acc} ({len(v)})')
    print('=' * 60)
    print('Saved to', result_file)
