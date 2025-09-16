import openai
import gzip
import json
import pandas as pd

from datasets import load_dataset

import multiprocessing
from typing import List, Dict, Any
import itertools
from datasets import load_dataset
from codeEvaluation import process_problem


openai.api_key = "YOUR_OPENAI_API_KEY"
openai.api_base = 'https://tb.plus7.plus/v1'
client = openai.OpenAI(
                api_key='your key',
                base_url='https://chatapi.littlewheat.com/v1'
            )

def load_humaneval_local(file_path):
    tasks = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    return tasks


def main(k: int = 1):

    # humaneval_tasks = load_humaneval_local("HumanEval.jsonl.gz")


    splits = {'train': 'full/train-00000-of-00001.parquet', 'test': 'full/test-00000-of-00001.parquet',
              'validation': 'full/validation-00000-of-00001.parquet', 'prompt': 'full/prompt-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/google-research-datasets/mbpp/" + splits["train"])
    testcases = df.code.values

    print(f"processed {len(testcases)} task")
    first_task = testcases[0]
    total_problems = len(testcases)
    passed_count = 0
    passed = process_problem(testcases, k=k)


if __name__ == "__main__":
    score_pass_at_1 = main(k=1)
