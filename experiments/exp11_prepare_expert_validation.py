"""
Experiment 11 - Expert Validation Packet
========================================

Prepares a CSV packet for procurement/security expert review. This does
not create validation results; it creates the instrument needed to collect
real expert ratings without fabricating human judgment.

Outputs:
  results/expert_validation_packet.csv
  results/expert_validation_protocol.json
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adaptive_attack_engine import ADAPTIVE_ATTACKS  # noqa: E402
from src.attack_engine import ATTACK_TAXONOMY  # noqa: E402


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        scenarios = json.load(f)

    out_csv = os.path.join(RESULTS_DIR, "expert_validation_packet.csv")
    fields = [
        "scenario_id",
        "category",
        "task",
        "description",
        "task_instruction",
        "expected_output",
        "ground_truth_decision",
        "sensitivity_level",
        "vendor_data_excerpt",
        "scenario_realism_1_5",
        "task_clarity_1_5",
        "ground_truth_agreement_1_5",
        "attack_surface_plausibility_1_5",
        "business_impact_1_5",
        "reviewer_notes",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(
                {
                    "scenario_id": scenario["id"],
                    "category": scenario.get("category", ""),
                    "task": scenario.get("task", ""),
                    "description": scenario.get("description", ""),
                    "task_instruction": scenario.get("task_instruction", ""),
                    "expected_output": scenario.get("expected_output", ""),
                    "ground_truth_decision": scenario.get("ground_truth_decision", ""),
                    "sensitivity_level": scenario.get("sensitivity_level", ""),
                    "vendor_data_excerpt": scenario.get("vendor_data", "")[:900],
                    "scenario_realism_1_5": "",
                    "task_clarity_1_5": "",
                    "ground_truth_agreement_1_5": "",
                    "attack_surface_plausibility_1_5": "",
                    "business_impact_1_5": "",
                    "reviewer_notes": "",
                }
            )

    protocol = {
        "experiment": "exp11_prepare_expert_validation",
        "purpose": "Collect real procurement/security expert ratings for scenario realism, task clarity, ground-truth agreement, attack-surface plausibility, and business impact.",
        "n_scenarios": len(scenarios),
        "recommended_reviewers": "At least 3 reviewers: two procurement/domain experts and one security/LLM-safety expert.",
        "rating_scale": {
            "1": "poor / implausible / disagree",
            "3": "acceptable / plausible with caveats",
            "5": "strong / realistic / agree",
        },
        "reporting_plan": [
            "Mean and standard deviation per rating dimension.",
            "Per-category realism and impact summaries.",
            "Inter-rater agreement using ICC or Krippendorff alpha.",
            "Manual adjudication notes for scenarios with low ground-truth agreement.",
        ],
        "attack_taxonomy_for_review": list(ATTACK_TAXONOMY.keys()),
        "adaptive_families_for_review": list(ADAPTIVE_ATTACKS.keys()),
        "packet_csv": out_csv,
    }
    out_json = os.path.join(RESULTS_DIR, "expert_validation_protocol.json")
    with open(out_json, "w") as f:
        json.dump(protocol, f, indent=2)

    print(json.dumps({"packet_csv": out_csv, "protocol_json": out_json}, indent=2))


if __name__ == "__main__":
    main()
