# BidPoison Code Artifacts

This folder contains the code and data artifacts needed to reproduce the experiments reported in the paper.

## Contents

- `src/`: benchmark, simulator, attack, defense, evaluator, and service-workflow modules.
- `experiments/`: experiment entry points used for the reported results.
- `data/`: procurement scenarios, attack templates, and schema-grounding metadata.
- `results/`: reported JSON outputs and visualization helper.
- `run_experiments.py`: top-level runner for the main simulator experiments.
- `requirements.txt`: Python dependencies.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Reproduction Commands

Run the core simulator suite:

```sh
python3 run_experiments.py
```

Run the full defense comparison used in the paper:

```sh
python3 experiments/exp4_defense_comparison.py
```

Run the composed service-workflow validation with the deterministic mock backend:

```sh
python3 experiments/exp5_service_validation.py --backend mock --models mock --guards defensechain guardrails --limit 65
```

Run the simulator-prior sensitivity analysis:

```sh
python3 experiments/exp7_sensitivity_analysis.py
```

Optional live local-model validation, if Ollama and the listed models are installed:

```sh
python3 experiments/exp5_service_validation.py --backend ollama --models llama3.2:3b llama3.2:1b qwen2.5:0.5b --guards defensechain --limit 5
```

## Notes

- Existing outputs are preserved in `results/`.
- The live/Ollama experiment is environment-dependent and is not required for deterministic reproduction.
- The paper manuscript and LaTeX sources are intentionally excluded; this folder is code-only.
