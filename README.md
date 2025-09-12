# CoT-Faithfulness: Evaluating Chain-of-Thought Faithfulness in Code Generation

## 🧠 Overview

This work presents a comprehensive study on the **faithfulness of Chain-of-Thought (CoT) reasoning** in large language model (LLM)–based code generation. While prior work often evaluates CoTs based on correctness (e.g., pass@k), we investigate whether CoTs truthfully reflect the underlying execution logic, structural planning, and performance behavior of the generated code.

We introduce a unified evaluation framework across **three dimensions**:

- **Execution Faithfulness** — Does the model truly follow the CoT when generating code?
- **Structural Faithfulness** — Are the intermediate variables and control flows aligned with CoT reasoning steps?
- **Non-Functional Faithfulness** — Are the claimed complexities (e.g., time/space) in CoT consistent with actual implementation?

## 📊 Contributions

- A taxonomy and evaluation protocol for CoT faithfulness in code generation
- New metrics:  
  - `Perturb–CoT Slope` (execution reasoning dependency)  
  - `Structural Consistency Score (SCS)`  
  - `Complexity Faithfulness (CF)` and `Differential Performance Score (DPS)`
- Multi-model benchmark comparison across HumanEval, MBPP, CodeElo, and EvalPerf
- A constraint-based CoT repair method using multi-CoT logic reconciliation via SAT solving

## 📁 Repository Structure

```bash
.
├── data/
│   ├── humaneval_cot.json        # CoT-augmented HumanEval benchmark
│   ├── mbpp_cot.json             # CoT-augmented MBPP benchmark
│   ├── codeelo_cot.json          # Competitive tasks with annotated CoTs
│   ├── annotations/
│   │   └── scs_labels.csv        # Human annotation files (Structural Consistency)
│   └── evalperf_outputs/         # Efficiency metrics under stress testing
│
├── src/
│   ├── evaluate_scs.py           # Script to compute Structural Consistency Score
│   ├── evaluate_cf_dps.py        # Metric extraction for CF and DPS
│   ├── perturb_slope_analysis.py # Perturb–CoT Slope computation
│   ├── logic_repair_sat.py       # Constraint-based CoT repair via SAT solver
│   └── utils/                    # Parsing, templating, logging
│
├── figures/
│   └── induced_unfaithful_case.png  # Case study figures for paper
│
├── README.md
└── requirements.txt
