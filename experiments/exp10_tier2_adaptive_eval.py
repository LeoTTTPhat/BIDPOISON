"""
Experiment 10 - Systematic Tier-2 Adaptive Evaluation
=====================================================

Evaluates the adaptive Tier-2 attack families across the full scenario
set and multiple reference service guards. The default backend is the
deterministic workflow mock so the experiment is fast and reproducible;
the same script can be pointed at local Ollama models for live stress
testing.

Outputs:
  results/exp10_tier2_adaptive_eval.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.eval_utils import (  # noqa: E402
    clean_correct,
    extract_service_value_usd,
    group_summary,
    summarize_service_records,
    target_success,
)
from src.adaptive_attack_engine import AdaptiveAttackEngine  # noqa: E402
from src.service_workflow import ProcurementServiceWorkflow, ServiceRequest  # noqa: E402


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_scenarios(limit: int | None) -> List[dict]:
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        scenarios = json.load(f)
    return scenarios[:limit] if limit else scenarios


def make_request(scenario: dict, vendor_data: str) -> ServiceRequest:
    return ServiceRequest(
        scenario_id=scenario["id"],
        task_instruction=scenario.get("task_instruction", ""),
        vendor_data=vendor_data,
        expected_output=scenario.get("expected_output", "SELECT_VENDOR"),
        ground_truth_decision=scenario.get("ground_truth_decision", "BaseVendor"),
    )


def run_for_backend_model_guard(
    backend: str,
    model: str,
    guard: str,
    scenarios: List[dict],
) -> Dict:
    workflow = ProcurementServiceWorkflow(backend=backend, model=model, guard=guard)
    engine = AdaptiveAttackEngine()
    records: List[dict] = []
    t0 = time.perf_counter()

    for scenario in scenarios:
        service_value = extract_service_value_usd(scenario)
        clean_resp = workflow.handle(make_request(scenario, scenario["vendor_data"]))
        records.append(
            {
                "backend": backend,
                "model": model,
                "guard": guard,
                "scenario_id": scenario["id"],
                "category": scenario.get("category", "unknown"),
                "attack_family": "clean",
                "attack_id": "clean",
                "decision": clean_resp.decision,
                "flagged": clean_resp.flagged,
                "state": clean_resp.state,
                "latency_ms": clean_resp.latency_ms,
                "target_success": False,
                "clean_correct": clean_correct(
                    clean_resp.decision,
                    scenario.get("ground_truth_decision", ""),
                ),
                "service_value_usd": service_value,
            }
        )

        target = "Adversary Vendor"
        for attack in engine.build_attacks(scenario["vendor_data"], target):
            resp = workflow.handle(make_request(scenario, attack.injected_data))
            records.append(
                {
                    "backend": backend,
                    "model": model,
                    "guard": guard,
                    "scenario_id": scenario["id"],
                    "category": scenario.get("category", "unknown"),
                    "attack_family": attack.family,
                    "attack_id": attack.attack_id,
                    "decision": resp.decision,
                    "flagged": resp.flagged,
                    "state": resp.state,
                    "latency_ms": resp.latency_ms,
                    "target_success": target_success(resp.decision, target),
                    "clean_correct": False,
                    "service_value_usd": service_value,
                }
            )

    attacked = [r for r in records if r["attack_family"] != "clean"]
    return {
        "backend": backend,
        "model": model,
        "guard": guard,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "summary": summarize_service_records(records),
        "by_attack_family": group_summary(attacked, "attack_family"),
        "by_category": group_summary(records, "category"),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--models", nargs="*", default=["mock"])
    parser.add_argument(
        "--guards",
        nargs="+",
        default=["none", "input_sanitization", "defensechain", "guardrails", "policy_chain"],
        choices=[
            "none",
            "defensechain",
            "guardrails",
            "policy_chain",
            "input_sanitization",
            "llm_judge",
            "provenance_oracle",
        ],
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scenarios = load_scenarios(args.limit or None)
    outputs = []
    for model in args.models:
        for guard in args.guards:
            print(
                f"[Exp10] backend={args.backend} model={model} "
                f"guard={guard} scenarios={len(scenarios)}"
            )
            outputs.append(
                run_for_backend_model_guard(
                    backend=args.backend,
                    model=model,
                    guard=guard,
                    scenarios=scenarios,
                )
            )

    result = {
        "experiment": "exp10_tier2_adaptive_eval",
        "backend": args.backend,
        "n_scenarios": len(scenarios),
        "adaptive_families": [
            "paraphrase",
            "obfuscation",
            "split_field",
            "fake_provenance",
            "delimiter_aware",
            "encoded",
        ],
        "model_guard_results": outputs,
    }
    out_path = os.path.join(RESULTS_DIR, "exp10_tier2_adaptive_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({f"{r['model']}::{r['guard']}": r["summary"] for r in outputs}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
