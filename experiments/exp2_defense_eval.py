"""
Experiment 2 — Defense Effectiveness Evaluation
=================================================
Compares NaiveProcurementAgent vs. StructuredProcurementAgent (StruQ-inspired)
across all attack types, injection positions, and scenarios.

Outputs (Table 2):
  - Defense Success Rate (DSR) by attack type
  - Suspicious Detection Rate (SDR): true-positive detection
  - False Positive Rate (FPR): clean inputs incorrectly flagged
  - Decision Consistency (DC): benign utility preservation
  - Latency overhead of defense layer

Usage:
    python experiments/exp2_defense_eval.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluator import ProcurementSecurityEvaluator, EvalRecord

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    evaluator = ProcurementSecurityEvaluator(model="mock")
    scenarios_path = os.path.join(DATA_DIR, "procurement_scenarios.json")

    print("\n[Exp2] Running defense effectiveness evaluation …")
    summary = evaluator.run_full_evaluation(
        scenarios_path=scenarios_path, n_templates=2
    )

    print("\n" + "="*70)
    print("  TABLE 2 — Defense Evaluation: Naive vs. StruQ")
    print("="*70)
    print(f"\n  {'Metric':<35s}  {'Value':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Attack Success Rate (Naive)':<35s}  {summary['asr_naive']:>12.2%}")
    print(f"  {'Attack Success Rate (StruQ)':<35s}  {summary['asr_struq']:>12.2%}")
    print(f"  {'Defense Success Rate (StruQ)':<35s}  {summary['dsr_struq']:>12.2%}")
    print(f"  {'Suspicious Detection Rate':<35s}  {summary['sdr_struq']:>12.2%}")
    print(f"  {'False Positive Rate':<35s}  {summary['false_positive_rate']:>12.2%}")
    print(f"  {'Decision Consistency (benign)':<35s}  {summary['decision_consistency']:>12.2%}")

    print(f"\n  {'Attack Type':<22}  {'ASR naive':>10}  {'DSR struq':>10}")
    print(f"  {'-'*46}")
    for at, m in sorted(summary["by_attack_type"].items()):
        print(f"  {at:<22}  {m['asr_naive']:>10.2%}  {m['dsr_struq']:>10.2%}")

    out_path = os.path.join(RESULTS_DIR, "exp2_defense_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Exp2] Results saved → {out_path}")
    return summary


if __name__ == "__main__":
    run()
