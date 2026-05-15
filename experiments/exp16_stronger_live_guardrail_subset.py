"""
Experiment 16 - Stronger Live Model + External Guardrail Subset
===============================================================

Runs a compact Tier-1 and Tier-2 live subset on a stronger local model
with an external Llama Guard baseline. This experiment is intentionally
reported as a stronger live-model subset, not as a population-level
estimate of deployed LLM vulnerability.

Outputs:
  results/exp16_stronger_live_guardrail_subset.json
  results/exp16_stronger_live_guardrail_subset.md
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp9_live_tier1_eval import run_for_model_guard as run_tier1  # noqa: E402
from experiments.exp10_tier2_adaptive_eval import (  # noqa: E402
    run_for_backend_model_guard as run_tier2,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def model_available(name: str) -> bool:
    return shutil.which("ollama") is not None and name in os.popen("ollama list").read()


def load_scenarios(limit: int) -> List[dict]:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "procurement_scenarios_extended.json")) as f:
        return json.load(f)[:limit]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_markdown(result: Dict, path: str) -> None:
    with open(path, "w") as f:
        f.write("# Stronger live-model subset with external Llama Guard\n\n")
        f.write(
            "This compact run is reported as workflow evidence on a stronger local "
            "model, not as a population-level vulnerability estimate.\n\n"
        )
        f.write(f"- Decision model: `{result['decision_model']}`\n")
        f.write(f"- External guard model: `{result['external_guard_model']}`\n")
        f.write(f"- Tier 1 scenarios: {result['tier1']['n_scenarios']}\n")
        f.write(f"- Tier 2 scenarios: {result['tier2']['n_scenarios']}\n\n")

        for tier_name in ("tier1", "tier2"):
            f.write(f"## {tier_name.upper()} summary\n\n")
            f.write("| Guard | Attacks | Silent ASR | W-Silent | Attack Esc. | Clean Esc. | Latency |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in result[tier_name]["model_guard_results"]:
                summary = row["summary"]
                f.write(
                    f"| {row['guard']} | {int(summary['n_attacked'])} | "
                    f"{pct(summary['silent_target_success_rate'])} | "
                    f"{pct(summary['weighted_silent_target_success_rate'])} | "
                    f"{pct(summary['attack_escalation_rate'])} | "
                    f"{pct(summary['clean_escalation_rate'])} | "
                    f"{summary['mean_latency_ms']:.1f} ms |\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral:7b")
    parser.add_argument("--guard-model", default="llama-guard3:1b")
    parser.add_argument("--tier1-limit", type=int, default=10)
    parser.add_argument("--tier2-limit", type=int, default=5)
    parser.add_argument(
        "--guards",
        nargs="+",
        default=["none", "guardrails", "llama_guard", "strict_tool_call"],
        choices=["none", "guardrails", "llama_guard", "strict_tool_call", "policy_chain"],
    )
    parser.add_argument("--tier1-attack-types", nargs="+", default=["T1", "T3", "T5"])
    parser.add_argument("--tier1-positions", nargs="+", default=["header", "metadata"])
    args = parser.parse_args()

    if not model_available(args.model):
        raise SystemExit(
            f"Decision model {args.model!r} is not installed in Ollama. "
            f"Run: ollama pull {args.model}"
        )
    if "llama_guard" in args.guards and not model_available(args.guard_model):
        raise SystemExit(
            f"External guard model {args.guard_model!r} is not installed in Ollama. "
            f"Run: ollama pull {args.guard_model}"
        )

    tier1_scenarios = load_scenarios(args.tier1_limit)
    tier2_scenarios = load_scenarios(args.tier2_limit)

    tier1_outputs = []
    for guard in args.guards:
        print(f"[Exp16/Tier1] model={args.model} guard={guard} scenarios={len(tier1_scenarios)}")
        tier1_outputs.append(
            run_tier1(
                model=args.model,
                guard=guard,
                scenarios=tier1_scenarios,
                attack_types=args.tier1_attack_types,
                positions=args.tier1_positions,
                repeats=1,
            )
        )

    tier2_outputs = []
    for guard in args.guards:
        print(f"[Exp16/Tier2] model={args.model} guard={guard} scenarios={len(tier2_scenarios)}")
        tier2_outputs.append(
            run_tier2(
                backend="ollama",
                model=args.model,
                guard=guard,
                scenarios=tier2_scenarios,
            )
        )

    result = {
        "experiment": "exp16_stronger_live_guardrail_subset",
        "decision_model": args.model,
        "external_guard_model": args.guard_model,
        "scope_note": (
            "Compact stronger live-model subset for workflow validation and guard "
            "comparison; not a population-level production-model estimate."
        ),
        "tier1": {
            "n_scenarios": len(tier1_scenarios),
            "attack_types": args.tier1_attack_types,
            "positions": args.tier1_positions,
            "model_guard_results": tier1_outputs,
        },
        "tier2": {
            "n_scenarios": len(tier2_scenarios),
            "adaptive_families": [
                "paraphrase",
                "obfuscation",
                "split_field",
                "fake_provenance",
                "delimiter_aware",
                "encoded",
            ],
            "model_guard_results": tier2_outputs,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "exp16_stronger_live_guardrail_subset.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    md_path = os.path.join(RESULTS_DIR, "exp16_stronger_live_guardrail_subset.md")
    write_markdown(result, md_path)

    print(json.dumps({
        "tier1": {row["guard"]: row["summary"] for row in tier1_outputs},
        "tier2": {row["guard"]: row["summary"] for row in tier2_outputs},
    }, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
