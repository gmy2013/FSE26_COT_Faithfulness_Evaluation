import json
import gzip
import shutil
from human_eval.evaluation import evaluate_functional_correctness

from codeGeneration import generate_code
from datasets import load_dataset
# from human_eval import HumanEval
# dataset = load_dataset("openai/humaneval")["test"]
def process_problem(problem: list, k: int = 1) -> bool:

    solutions = []
    k = -1
    for problem1 in problem:
        # prompt = problem1["prompt"]
        prompt = problem1
        k += 1
        coT = generate_code(prompt+'\n'+'Please give the structured chain of thought only!\n', n=1)
        # coT = generate_code(prompt+'\n'+'please give  the chain of thought, in the chain of thought please predict and show every heapq value if heapq heappops when COT input is ([[5,7],[6,1]],5)!\n', n=k)
        # coT = generate_code(prompt+'\n'+'Please simply give the chain of thought  while  predicting and showing every heapq value in all COT steps if heapq heappops when COT input is ([[5,7],[6,1]],5) !\n', n=k)


        l = len(coT[0])
        prunedCoT = coT[0][:int(0.8*l)]
        prompt = prompt+'\n'+'The following is chain of thought:\n' + prunedCoT
        with open('human-eval/test/results3/HumanEval'+str(k)+'_prompt','w') as f:
            f.write(prompt)
        completions = generate_code(prompt, n=1)
        with open('human-eval/test/results3/HumanEval'+str(k)+'_codeGeneration','w') as f:
            # f.write('20%CoT\n')
            f.write('CodeGeneration:\n')
            f.write(completions[0])
            f.write('\n')
        print('finish'+str(k)+'case')

