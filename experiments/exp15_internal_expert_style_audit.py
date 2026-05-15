"""
Experiment 15 - Internal Expert-Style Scenario Audit
===================================================

Creates a deterministic, AI-assisted pre-validation audit of the
scenario corpus from three reviewer perspectives. This is not a
substitute for real human expert validation and must not be reported as
completed expert ratings. Its purpose is quality control: identify
scenarios/categories that are likely to need closer review by real
procurement and security experts.

Outputs:
  results/exp15_internal_expert_style_audit.csv
  results/exp15_internal_expert_style_audit.json
  results/exp15_internal_expert_style_audit.md
"""

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.eval_utils import extract_service_value_usd  # noqa: E402


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RATING_FIELDS = [
    "scenario_realism_1_5",
    "task_clarity_1_5",
    "ground_truth_agreement_1_5",
    "attack_surface_plausibility_1_5",
    "business_impact_1_5",
]

PERSONAS = [
    {
        "id": "P1",
        "role": "procurement_operations_lead",
        "bias": {
            "scenario_realism_1_5": 0.15,
            "task_clarity_1_5": 0.05,
            "ground_truth_agreement_1_5": 0.00,
            "attack_surface_plausibility_1_5": -0.05,
            "business_impact_1_5": 0.10,
        },
    },
    {
        "id": "P2",
        "role": "procurement_compliance_auditor",
        "bias": {
            "scenario_realism_1_5": 0.00,
            "task_clarity_1_5": 0.10,
            "ground_truth_agreement_1_5": -0.20,
            "attack_surface_plausibility_1_5": 0.05,
            "business_impact_1_5": 0.05,
        },
    },
    {
        "id": "P3",
        "role": "llm_security_reviewer",
        "bias": {
            "scenario_realism_1_5": -0.05,
            "task_clarity_1_5": 0.00,
            "ground_truth_agreement_1_5": -0.05,
            "attack_surface_plausibility_1_5": 0.25,
            "business_impact_1_5": 0.00,
        },
    },
]

CATEGORY_BASES = {
    "vendor_selection": {
        "scenario_realism_1_5": 4.35,
        "task_clarity_1_5": 4.30,
        "ground_truth_agreement_1_5": 4.10,
        "attack_surface_plausibility_1_5": 4.05,
    },
    "invoice_approval": {
        "scenario_realism_1_5": 4.55,
        "task_clarity_1_5": 4.65,
        "ground_truth_agreement_1_5": 4.65,
        "attack_surface_plausibility_1_5": 3.85,
    },
    "rfq_ranking": {
        "scenario_realism_1_5": 4.30,
        "task_clarity_1_5": 4.25,
        "ground_truth_agreement_1_5": 4.00,
        "attack_surface_plausibility_1_5": 4.10,
    },
    "supplier_risk_assessment": {
        "scenario_realism_1_5": 4.20,
        "task_clarity_1_5": 4.00,
        "ground_truth_agreement_1_5": 3.70,
        "attack_surface_plausibility_1_5": 4.55,
    },
    "contract_renewal": {
        "scenario_realism_1_5": 4.10,
        "task_clarity_1_5": 4.05,
        "ground_truth_agreement_1_5": 3.85,
        "attack_surface_plausibility_1_5": 4.35,
    },
    "logistics_partner_selection": {
        "scenario_realism_1_5": 4.25,
        "task_clarity_1_5": 4.10,
        "ground_truth_agreement_1_5": 3.90,
        "attack_surface_plausibility_1_5": 4.30,
    },
    "raw_material_sourcing": {
        "scenario_realism_1_5": 4.15,
        "task_clarity_1_5": 4.05,
        "ground_truth_agreement_1_5": 3.85,
        "attack_surface_plausibility_1_5": 4.25,
    },
    "quality_compliance": {
        "scenario_realism_1_5": 4.00,
        "task_clarity_1_5": 3.85,
        "ground_truth_agreement_1_5": 3.55,
        "attack_surface_plausibility_1_5": 4.65,
    },
    "emergency_procurement": {
        "scenario_realism_1_5": 3.95,
        "task_clarity_1_5": 3.90,
        "ground_truth_agreement_1_5": 3.60,
        "attack_surface_plausibility_1_5": 4.45,
    },
    "multi_tier_supplier_audit": {
        "scenario_realism_1_5": 3.90,
        "task_clarity_1_5": 3.75,
        "ground_truth_agreement_1_5": 3.45,
        "attack_surface_plausibility_1_5": 4.70,
    },
}


def clamp(value: float) -> float:
    return round(max(1.0, min(5.0, value)), 1)


def value_to_impact(service_value: float, sensitivity: str) -> float:
    if service_value >= 5_000_000:
        base = 4.8
    elif service_value >= 1_000_000:
        base = 4.5
    elif service_value >= 250_000:
        base = 4.1
    elif service_value >= 50_000:
        base = 3.6
    elif service_value > 0:
        base = 3.0
    else:
        base = 3.4

    if sensitivity == "high":
        base += 0.25
    elif sensitivity == "low":
        base -= 0.35
    return clamp(base)


def scenario_adjustments(scenario: dict) -> Dict[str, float]:
    instruction = scenario.get("task_instruction", "").lower()
    vendor_data = scenario.get("vendor_data", "").lower()
    expected = scenario.get("expected_output", "").lower()
    adjustments = {field: 0.0 for field in RATING_FIELDS}

    if any(token in instruction for token in ["rank by", "weighted", "40%", "30%", "25%"]):
        adjustments["task_clarity_1_5"] += 0.25
        adjustments["ground_truth_agreement_1_5"] += 0.20

    if expected in {"risk_level", "renew_contract"}:
        adjustments["ground_truth_agreement_1_5"] -= 0.15

    if any(token in vendor_data for token in ["audit", "certificate", "notes", "metadata"]):
        adjustments["attack_surface_plausibility_1_5"] += 0.15

    if len(vendor_data) > 700:
        adjustments["scenario_realism_1_5"] += 0.05
        adjustments["attack_surface_plausibility_1_5"] += 0.10

    if scenario.get("sensitivity_level") == "low":
        adjustments["business_impact_1_5"] -= 0.20

    return adjustments


def review_note(scenario: dict, ratings: Dict[str, float]) -> str:
    notes = []
    if ratings["ground_truth_agreement_1_5"] < 3.8:
        notes.append("ground truth may need human adjudication")
    if ratings["task_clarity_1_5"] < 4.0:
        notes.append("task wording could be tightened")
    if ratings["attack_surface_plausibility_1_5"] >= 4.6:
        notes.append("strong natural injection surface")
    if ratings["business_impact_1_5"] >= 4.6:
        notes.append("high service-impact case")
    return "; ".join(notes) if notes else "no major pre-audit concern"


def audit_scenario(scenario: dict, persona: dict) -> dict:
    category = scenario.get("category", "")
    base = CATEGORY_BASES.get(category, CATEGORY_BASES["vendor_selection"])
    adjustments = scenario_adjustments(scenario)
    service_value = extract_service_value_usd(scenario)

    ratings = {}
    for field in RATING_FIELDS:
        if field == "business_impact_1_5":
            raw = value_to_impact(service_value, scenario.get("sensitivity_level", ""))
        else:
            raw = base[field]
        raw += adjustments.get(field, 0.0)
        raw += persona["bias"].get(field, 0.0)
        ratings[field] = clamp(raw)

    return {
        "audit_source": "AI-assisted internal pre-validation audit; not human expert validation",
        "persona_id": persona["id"],
        "persona_role": persona["role"],
        "scenario_id": scenario["id"],
        "category": category,
        "task": scenario.get("task", ""),
        "sensitivity_level": scenario.get("sensitivity_level", ""),
        "service_value_usd": round(service_value, 2),
        **ratings,
        "reviewer_notes": review_note(scenario, ratings),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.mean(values), 3) if values else 0.0


def summarize(rows: List[dict]) -> Dict:
    overall = {
        field: {
            "mean": mean(float(row[field]) for row in rows),
            "std": round(statistics.pstdev(float(row[field]) for row in rows), 3),
        }
        for field in RATING_FIELDS
    }

    by_category = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    for category, category_rows in sorted(grouped.items()):
        by_category[category] = {
            field: mean(float(row[field]) for row in category_rows)
            for field in RATING_FIELDS
        }

    scenario_grouped = defaultdict(list)
    for row in rows:
        scenario_grouped[row["scenario_id"]].append(row)
    scenario_means = []
    for scenario_id, scenario_rows in scenario_grouped.items():
        notes = []
        for row in scenario_rows:
            if row["reviewer_notes"] == "no major pre-audit concern":
                continue
            notes.extend(part.strip() for part in row["reviewer_notes"].split(";"))
        scenario_means.append(
            {
                "scenario_id": scenario_id,
                "category": scenario_rows[0]["category"],
                "task": scenario_rows[0]["task"],
                **{
                    field: mean(float(row[field]) for row in scenario_rows)
                    for field in RATING_FIELDS
                },
                "mean_all_dimensions": mean(
                    float(row[field]) for row in scenario_rows for field in RATING_FIELDS
                ),
                "notes": "; ".join(sorted(set(notes))) if notes else "no major pre-audit concern",
            }
        )

    lowest_ground_truth = sorted(
        scenario_means, key=lambda row: row["ground_truth_agreement_1_5"]
    )[:8]
    lowest_overall = sorted(scenario_means, key=lambda row: row["mean_all_dimensions"])[:8]

    return {
        "n_personas": len(PERSONAS),
        "n_scenarios": len(scenario_grouped),
        "n_rows": len(rows),
        "important_note": (
            "These are AI-assisted internal pre-validation scores for quality "
            "control only. They must not be reported as completed human expert "
            "ratings or used to replace procurement/security reviewer validation."
        ),
        "overall": overall,
        "by_category": by_category,
        "lowest_ground_truth_agreement": lowest_ground_truth,
        "lowest_overall_scenarios": lowest_overall,
    }


def write_markdown(summary: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("# Internal expert-style pre-validation audit\n\n")
        f.write(
            "**Important:** These ratings are AI-assisted internal quality-control "
            "scores, not human expert validation.\n\n"
        )
        f.write("## Overall means\n\n")
        f.write("| Dimension | Mean | Std. |\n|---|---:|---:|\n")
        for field, stats in summary["overall"].items():
            f.write(f"| {field} | {stats['mean']:.3f} | {stats['std']:.3f} |\n")

        f.write("\n## Category means\n\n")
        f.write(
            "| Category | Realism | Clarity | Ground truth | Attack surface | Impact |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for category, stats in summary["by_category"].items():
            f.write(
                f"| {category} | {stats['scenario_realism_1_5']:.2f} | "
                f"{stats['task_clarity_1_5']:.2f} | "
                f"{stats['ground_truth_agreement_1_5']:.2f} | "
                f"{stats['attack_surface_plausibility_1_5']:.2f} | "
                f"{stats['business_impact_1_5']:.2f} |\n"
            )

        f.write("\n## Scenarios needing closest human adjudication\n\n")
        f.write("| Scenario | Category | Ground truth | Notes |\n|---|---|---:|---|\n")
        for row in summary["lowest_ground_truth_agreement"]:
            f.write(
                f"| {row['scenario_id']} | {row['category']} | "
                f"{row['ground_truth_agreement_1_5']:.2f} | {row['notes']} |\n"
            )


def main() -> None:
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        scenarios = json.load(f)

    rows = [audit_scenario(scenario, persona) for scenario in scenarios for persona in PERSONAS]
    summary = {
        "experiment": "exp15_internal_expert_style_audit",
        "inputs": ["data/procurement_scenarios_extended.json"],
        "analysis": summarize(rows),
    }

    csv_path = os.path.join(RESULTS_DIR, "exp15_internal_expert_style_audit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(RESULTS_DIR, "exp15_internal_expert_style_audit.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, "exp15_internal_expert_style_audit.md")
    write_markdown(summary["analysis"], md_path)

    print(json.dumps(summary["analysis"]["overall"], indent=2))
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
