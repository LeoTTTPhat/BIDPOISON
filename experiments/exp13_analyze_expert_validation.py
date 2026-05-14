"""
Experiment 13 - Analyze Expert Validation Ratings
=================================================

Consumes completed expert validation CSV files and reports scenario-realism,
task-clarity, ground-truth-agreement, attack-plausibility, and business-impact
summaries. This script expects real human ratings; it does not fabricate them.

Usage:
  python3 experiments/exp13_analyze_expert_validation.py \
    --inputs results/expert_validation_reviewer1.csv results/expert_validation_reviewer2.csv

Outputs:
  results/exp13_expert_validation_analysis.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RATING_FIELDS = [
    "scenario_realism_1_5",
    "task_clarity_1_5",
    "ground_truth_agreement_1_5",
    "attack_surface_plausibility_1_5",
    "business_impact_1_5",
]


def _to_float(value: str) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if 1.0 <= numeric <= 5.0:
        return numeric
    return None


def load_rows(paths: List[str]) -> List[dict]:
    rows = []
    for reviewer_idx, path in enumerate(paths, start=1):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_reviewer"] = f"R{reviewer_idx}"
                rows.append(row)
    return rows


def summarize(rows: List[dict]) -> Dict:
    field_summary = {}
    for field in RATING_FIELDS:
        values = [_to_float(row.get(field, "")) for row in rows]
        values = [v for v in values if v is not None]
        field_summary[field] = {
            "n": len(values),
            "mean": round(mean(values), 3) if values else None,
            "std": round(pstdev(values), 3) if len(values) > 1 else 0.0,
        }

    scenario_summary = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("scenario_id", "")].append(row)
    for sid, sid_rows in sorted(grouped.items()):
        scenario_summary[sid] = {}
        for field in RATING_FIELDS:
            values = [_to_float(row.get(field, "")) for row in sid_rows]
            values = [v for v in values if v is not None]
            scenario_summary[sid][field] = round(mean(values), 3) if values else None

    return {
        "n_reviewers": len(set(row["_reviewer"] for row in rows)),
        "n_rows": len(rows),
        "n_scenarios": len(grouped),
        "rating_fields": field_summary,
        "scenario_summary": scenario_summary,
        "agreement_note": (
            "For publication, compute ICC or Krippendorff alpha once at least "
            "two independent completed reviewer files are available. This script "
            "reports descriptive agreement-ready summaries only."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()

    output = {
        "experiment": "exp13_analyze_expert_validation",
        "inputs": args.inputs,
        "analysis": summarize(load_rows(args.inputs)),
    }
    out_path = os.path.join(RESULTS_DIR, "exp13_expert_validation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output["analysis"]["rating_fields"], indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
