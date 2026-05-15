"""
Experiment 9 - Broad Live Tier-1 Evaluation
===========================================

Runs the Tier-1 attack matrix against live Ollama models through the
procurement service workflow. The script supports the full 65 x 5 x 4
matrix, but defaults should be chosen according to available local
inference time.

Outputs:
  results/exp9_live_tier1_eval.json

Example:
  python3 experiments/exp9_live_tier1_eval.py \
    --models qwen2.5:0.5b llama3.2:1b \
    --guards none guardrails policy_chain \
    --limit 5
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
from src.attack_engine import (  # noqa: E402
    ATTACK_TAXONOMY,
    INJECTION_POSITIONS,
    AttackConfig,
    AttackEngine,
)
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


def run_for_model_guard(
    model: str,
    guard: str,
    scenarios: List[dict],
    attack_types: List[str],
    positions: List[str],
    repeats: int,
) -> Dict:
    workflow = ProcurementServiceWorkflow(backend="ollama", model=model, guard=guard)
    engine = AttackEngine()
    records: List[dict] = []
    t0 = time.perf_counter()

    for scenario in scenarios:
        service_value = extract_service_value_usd(scenario)
        clean_resp = workflow.handle(make_request(scenario, scenario["vendor_data"]))
        records.append(
            {
                "model": model,
                "guard": guard,
                "scenario_id": scenario["id"],
                "category": scenario.get("category", "unknown"),
                "attack_family": "clean",
                "attack_type": "clean",
                "position": "clean",
                "repeat": 0,
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

        for attack_type in attack_types:
            for position in positions:
                for repeat in range(repeats):
                    target = f"Adversary Vendor {repeat + 1}"
                    attacked = engine.inject(
                        scenario["vendor_data"],
                        AttackConfig(
                            attack_type=attack_type,
                            position=position,
                            target_value=target,
                            template_idx=repeat,
                        ),
                    )
                    resp = workflow.handle(make_request(scenario, attacked.injected_data))
                    records.append(
                        {
                            "model": model,
                            "guard": guard,
                            "scenario_id": scenario["id"],
                            "category": scenario.get("category", "unknown"),
                            "attack_family": attack_type,
                            "attack_type": attack_type,
                            "position": position,
                            "repeat": repeat,
                            "decision": resp.decision,
                            "flagged": resp.flagged,
                            "state": resp.state,
                            "latency_ms": resp.latency_ms,
                            "target_success": target_success(resp.decision, target),
                            "clean_correct": False,
                            "service_value_usd": service_value,
                        }
                    )

    return {
        "model": model,
        "guard": guard,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "summary": summarize_service_records(records),
        "by_attack_type": group_summary(
            [r for r in records if r["attack_family"] != "clean"],
            "attack_type",
        ),
        "by_position": group_summary(
            [r for r in records if r["attack_family"] != "clean"],
            "position",
        ),
        "by_category": group_summary(records, "category"),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--guards",
        nargs="+",
        default=["none", "guardrails", "policy_chain"],
        choices=[
            "none",
            "defensechain",
            "guardrails",
            "policy_chain",
            "input_sanitization",
            "llm_judge",
            "llama_guard",
            "provenance_oracle",
            "strict_tool_call",
        ],
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--attack-types", nargs="*", default=list(ATTACK_TAXONOMY.keys()))
    parser.add_argument("--positions", nargs="*", default=INJECTION_POSITIONS)
    args = parser.parse_args()

    scenarios = load_scenarios(args.limit)
    outputs = []
    for model in args.models:
        for guard in args.guards:
            print(f"[Exp9] model={model} guard={guard} scenarios={len(scenarios)}")
            outputs.append(
                run_for_model_guard(
                    model=model,
                    guard=guard,
                    scenarios=scenarios,
                    attack_types=args.attack_types,
                    positions=args.positions,
                    repeats=args.repeats,
                )
            )

    result = {
        "experiment": "exp9_live_tier1_eval",
        "backend": "ollama",
        "n_scenarios": len(scenarios),
        "full_matrix_requested": len(scenarios) == 65,
        "attack_types": args.attack_types,
        "positions": args.positions,
        "repeats": args.repeats,
        "model_guard_results": outputs,
    }
    out_path = os.path.join(RESULTS_DIR, "exp9_live_tier1_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({f"{r['model']}::{r['guard']}": r["summary"] for r in outputs}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
