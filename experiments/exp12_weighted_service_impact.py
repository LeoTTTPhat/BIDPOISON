"""
Experiment 12 - Weighted Service-Impact Metrics
===============================================

Computes value-weighted attack success for the Tier-1 reference-simulator
defense comparison. This complements ordinary ASR by weighting failures by
estimated procurement service value.

Outputs:
  results/exp12_weighted_service_impact.json
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.eval_utils import extract_service_value_usd  # noqa: E402
from experiments.exp4_defense_comparison import (  # noqa: E402
    RANDOM_SEED,
    evaluate_all_defenses,
)
from src.llm_behavior_simulator import LLMBehaviorSimulator  # noqa: E402


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


DEFENSE_PASS_KEYS = {
    "Naive": "naive_succeeded",
    "StruQ": "struq_passed",
    "GuardrailsPolicy": "guardrails_passed",
    "OutputValidation": "ov_passed",
    "SemanticSimilarity": "sem_passed",
    "DefenseChain": "chain_passed",
}


def weighted_rate(records: List[dict], pass_key: str, weights: Dict[str, float]) -> Dict:
    total = 0.0
    failed = 0.0
    count_failed = 0
    for record in records:
        weight = weights[record["scenario_id"]]
        total += weight
        if record[pass_key]:
            failed += weight
            count_failed += 1
    return {
        "asr": round(sum(1 for r in records if r[pass_key]) / len(records), 4),
        "weighted_asr": round(failed / total, 4) if total else 0.0,
        "total_attacked_value_usd": round(total, 2),
        "compromised_value_usd": round(failed, 2),
        "n_failures": count_failed,
        "n_records": len(records),
    }


def run() -> Dict:
    t0 = time.perf_counter()
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        scenarios = json.load(f)

    weights = {
        scenario["id"]: extract_service_value_usd(scenario)
        for scenario in scenarios
    }
    simulator = LLMBehaviorSimulator(model_profile="ensemble", base_seed=RANDOM_SEED)
    defense_results, records = evaluate_all_defenses(scenarios, simulator)

    by_defense = {
        defense: weighted_rate(records, pass_key, weights)
        for defense, pass_key in DEFENSE_PASS_KEYS.items()
    }

    by_category = {}
    for defense, pass_key in DEFENSE_PASS_KEYS.items():
        grouped = defaultdict(list)
        for record in records:
            grouped[record["category"]].append(record)
        by_category[defense] = {
            category: weighted_rate(rows, pass_key, weights)
            for category, rows in sorted(grouped.items())
        }

    output = {
        "experiment": "exp12_weighted_service_impact",
        "n_scenarios": len(scenarios),
        "n_records": len(records),
        "weight_definition": "Estimated procurement value extracted from scenario vendor_data fields such as contract_value_usd, budget_usd, po_total_usd, invoice_total_usd, or sensitivity-level fallback.",
        "ordinary_defense_results": defense_results,
        "weighted_by_defense": by_defense,
        "weighted_by_category": by_category,
        "scenario_weights_usd": weights,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, "exp12_weighted_service_impact.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(by_defense, indent=2))
    print(f"Saved {out_path}")
    return output


if __name__ == "__main__":
    run()
