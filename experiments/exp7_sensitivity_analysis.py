"""
Experiment 7 — Compact Simulator-Prior Sensitivity Analysis
===========================================================

Varies simulator base ASR priors and position multipliers, then reruns
the full Tier-1 defense comparison. This is not a replacement for
live-model validation; it checks whether the main defense ordering is
fragile to reasonable simulator perturbations.

Output:
  results/exp7_sensitivity_analysis.json
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.llm_behavior_simulator as sim_mod
from src.llm_behavior_simulator import LLMBehaviorSimulator

from experiments.exp4_defense_comparison import evaluate_all_defenses


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


POSITION_SETTINGS = {
    "flat": {
        "header": 1.0,
        "footer": 1.0,
        "inline_comment": 1.0,
        "metadata_field": 1.0,
    },
    "default": {
        "header": 1.25,
        "footer": 0.85,
        "inline_comment": 0.70,
        "metadata_field": 0.95,
    },
    "header_heavy": {
        "header": 1.50,
        "footer": 0.80,
        "inline_comment": 0.60,
        "metadata_field": 0.90,
    },
}


def _scaled_profiles(base_profiles: dict, scale: float) -> dict:
    scaled = copy.deepcopy(base_profiles)
    for profile in scaled.values():
        for attack, value in profile.items():
            profile[attack] = min(0.95, max(0.0, value * scale))
    return scaled


def load_scenarios() -> list:
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        return json.load(f)


def main():
    scenarios = load_scenarios()
    original_profiles = copy.deepcopy(sim_mod.MODEL_PROFILES)
    original_positions = copy.deepcopy(sim_mod.POSITION_MULTIPLIERS)
    rows = []

    try:
        for prior_scale in (0.75, 1.0, 1.25):
            for position_name, position_values in POSITION_SETTINGS.items():
                sim_mod.MODEL_PROFILES.clear()
                sim_mod.MODEL_PROFILES.update(_scaled_profiles(original_profiles, prior_scale))
                sim_mod.POSITION_MULTIPLIERS.clear()
                sim_mod.POSITION_MULTIPLIERS.update(position_values)

                simulator = LLMBehaviorSimulator(model_profile="ensemble", base_seed=42)
                defense_results, _ = evaluate_all_defenses(scenarios, simulator)
                rows.append(
                    {
                        "prior_scale": prior_scale,
                        "position_setting": position_name,
                        "naive_asr": defense_results["Naive"]["overall_asr"],
                        "struq_asr": defense_results["StruQ"]["overall_asr"],
                        "guardrails_asr": defense_results["GuardrailsPolicy"]["overall_asr"],
                        "defensechain_asr": defense_results["DefenseChain"]["overall_asr"],
                        "best_defense": min(
                            defense_results,
                            key=lambda name: defense_results[name]["overall_asr"],
                        ),
                    }
                )
    finally:
        sim_mod.MODEL_PROFILES.clear()
        sim_mod.MODEL_PROFILES.update(original_profiles)
        sim_mod.POSITION_MULTIPLIERS.clear()
        sim_mod.POSITION_MULTIPLIERS.update(original_positions)

    summary = {
        "experiment": "exp7_sensitivity_analysis",
        "n_settings": len(rows),
        "rows": rows,
        "defensechain_asr_min": min(r["defensechain_asr"] for r in rows),
        "defensechain_asr_max": max(r["defensechain_asr"] for r in rows),
        "naive_asr_min": min(r["naive_asr"] for r in rows),
        "naive_asr_max": max(r["naive_asr"] for r in rows),
    }
    out = os.path.join(RESULTS_DIR, "exp7_sensitivity_analysis.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
