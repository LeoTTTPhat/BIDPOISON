"""
Experiment 14 - Live Run Artifact Summary
=========================================

Builds a compact artifact table from the released live-model result files.
The table is intended for appendices or artifact review packets.

Outputs:
  results/exp14_live_run_summary.csv
  results/exp14_live_run_summary.md
  results/exp14_live_run_summary.json
"""

import csv
import json
import os
from typing import Dict, List


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

LIVE_FILES = [
    ("E9 Qwen Tier 1 (10 scenarios)", "exp9_live_tier1_eval_qwen10.json"),
    ("E9 Llama 3B Tier 1 (10 scenarios)", "exp9_live_tier1_eval_llama3b10.json"),
    ("E10 Qwen Tier 2 adaptive subset", "exp10_tier2_adaptive_eval_qwen_live_subset.json"),
    ("E5 three-model workflow smoke", "exp9_live_tier1_eval_qwen_llama_paired.json"),
    ("E5 Llama 3B smoke", "exp9_live_tier1_eval_llama3b_smoke.json"),
    ("E9 Qwen broad subset", "exp9_live_tier1_eval_qwen_broad.json"),
]

FIELDS = [
    "subset",
    "source_file",
    "backend",
    "model",
    "guard",
    "n_scenarios",
    "n_attacked",
    "n_clean",
    "silent_asr_pct",
    "weighted_silent_asr_pct",
    "attack_escalation_pct",
    "clean_escalation_pct",
    "clean_accuracy_pct",
    "mean_latency_ms",
    "throughput_req_s",
    "silent_compromised_value_usd",
]


def pct(value: float) -> float:
    return round(100.0 * float(value), 1)


def load_rows() -> List[Dict]:
    rows: List[Dict] = []
    for subset, filename in LIVE_FILES:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        for result in payload.get("model_guard_results", []):
            summary = result.get("summary", {})
            rows.append(
                {
                    "subset": subset,
                    "source_file": filename,
                    "backend": result.get("backend", payload.get("backend", "ollama")),
                    "model": result.get("model", ""),
                    "guard": result.get("guard", ""),
                    "n_scenarios": payload.get("n_scenarios", ""),
                    "n_attacked": summary.get("n_attacked", 0),
                    "n_clean": summary.get("n_clean", 0),
                    "silent_asr_pct": pct(summary.get("silent_target_success_rate", 0.0)),
                    "weighted_silent_asr_pct": pct(
                        summary.get("weighted_silent_target_success_rate", 0.0)
                    ),
                    "attack_escalation_pct": pct(summary.get("attack_escalation_rate", 0.0)),
                    "clean_escalation_pct": pct(summary.get("clean_escalation_rate", 0.0)),
                    "clean_accuracy_pct": pct(summary.get("clean_accuracy", 0.0)),
                    "mean_latency_ms": round(float(summary.get("mean_latency_ms", 0.0)), 1),
                    "throughput_req_s": round(float(summary.get("throughput_req_s", 0.0)), 3),
                    "silent_compromised_value_usd": round(
                        float(summary.get("silent_compromised_value_usd", 0.0)), 2
                    ),
                }
            )
    return rows


def write_markdown(rows: List[Dict], path: str) -> None:
    headers = [
        "Subset",
        "Model",
        "Guard",
        "Atk.",
        "Silent ASR",
        "W-Silent",
        "Atk. Esc.",
        "Clean Esc.",
        "Latency",
    ]
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["subset"]),
                        str(row["model"]),
                        str(row["guard"]),
                        str(row["n_attacked"]),
                        f"{row['silent_asr_pct']}%",
                        f"{row['weighted_silent_asr_pct']}%",
                        f"{row['attack_escalation_pct']}%",
                        f"{row['clean_escalation_pct']}%",
                        f"{row['mean_latency_ms']} ms",
                    ]
                )
                + " |\n"
            )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = load_rows()
    csv_path = os.path.join(RESULTS_DIR, "exp14_live_run_summary.csv")
    md_path = os.path.join(RESULTS_DIR, "exp14_live_run_summary.md")
    json_path = os.path.join(RESULTS_DIR, "exp14_live_run_summary.json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)
    with open(json_path, "w") as f:
        json.dump({"experiment": "exp14_live_run_summary", "rows": rows}, f, indent=2)
    print(json.dumps({"n_rows": len(rows), "csv": csv_path, "markdown": md_path}, indent=2))


if __name__ == "__main__":
    main()
