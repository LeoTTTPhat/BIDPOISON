"""
Experiment 1 — Attack Taxonomy Evaluation
==========================================
Evaluates the attack success rate (ASR) of all 5 injection types
against the naive procurement agent across all 5 procurement scenarios.

Outputs (Table 1):
  - Attack Success Rate by attack type (T1 … T5)
  - Attack Success Rate by injection position (header/footer/inline/metadata)
  - Decision deviation from clean baseline

Usage:
    python experiments/exp1_attack_taxonomy.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluator import ProcurementSecurityEvaluator
from src.attack_engine import ATTACK_TAXONOMY, INJECTION_POSITIONS

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    evaluator = ProcurementSecurityEvaluator(model="mock")
    scenarios_path = os.path.join(DATA_DIR, "procurement_scenarios.json")

    print("\n[Exp1] Running attack taxonomy evaluation …")
    summary = evaluator.run_full_evaluation(
        scenarios_path=scenarios_path, n_templates=1
    )

    print("\n" + "="*65)
    print("  TABLE 1 — Attack Success Rate (Naive Agent)")
    print("="*65)
    print(f"  Overall ASR (naive):  {summary['asr_naive']:.2%}")
    print(f"  Overall ASR (struq):  {summary['asr_struq']:.2%}")
    print()
    print(f"  {'Attack Type':<22}  {'ASR naive':>10}  {'DSR struq':>10}  {'SDR struq':>10}")
    print(f"  {'-'*58}")
    for at, m in sorted(summary["by_attack_type"].items()):
        print(f"  {at:<22}  {m['asr_naive']:>10.2%}  "
              f"{m['dsr_struq']:>10.2%}  {m['sdr_struq']:>10.2%}")

    print(f"\n  False Positive Rate:       {summary['false_positive_rate']:.2%}")
    print(f"  Decision Consistency:      {summary['decision_consistency']:.2%}")

    out_path = os.path.join(RESULTS_DIR, "exp1_attack_taxonomy.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Exp1] Results saved → {out_path}")
    return summary


if __name__ == "__main__":
    run()
