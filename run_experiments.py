#!/usr/bin/env python3
"""
BidPoison — One-Click Experiment Runner
========================================
Runs all four experiments and prints a consolidated result summary.

Experiments:
  Exp 1 — Attack Taxonomy        (5 scenarios, naive agent, mock LLM)
  Exp 2 — Defense Effectiveness  (StruQ vs Naive, 5 scenarios)
  Exp 3 — Extended Analysis      (60 scenarios, ensemble LLM simulator,
                                   bootstrap CIs, sensitivity + category breakdown)
  Exp 4 — Full Defense Comparison (5 defenses × 5 attacks × 60 scenarios,
                                   statistical significance, LaTeX table)

Usage:
    python run_experiments.py
"""

import json
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def header(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


def main():
    print("""
  ██████╗ ██╗██████╗ ██████╗  ██████╗ ██╗███████╗ ██████╗ ███╗   ██╗
  ██╔══██╗██║██╔══██╗██╔══██╗██╔═══██╗██║██╔════╝██╔═══██╗████╗  ██║
  ██████╔╝██║██║  ██║██████╔╝██║   ██║██║███████╗██║   ██║██╔██╗ ██║
  ██╔══██╗██║██║  ██║██╔═══╝ ██║   ██║██║╚════██║██║   ██║██║╚██╗██║
  ██████╔╝██║██████╔╝██║      ╚██████╔╝██║███████║╚██████╔╝██║ ╚████║
  ╚═════╝ ╚═╝╚═════╝ ╚═╝       ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
    """)
    print("  Prompt Injection Attacks in LLM-Powered Procurement Agents")
    print("  Full Evaluation Suite — BidPoison v2\n")

    t0 = time.perf_counter()
    sys.path.insert(0, os.path.dirname(__file__))

    from experiments.exp1_attack_taxonomy  import run as run_exp1
    from experiments.exp2_defense_eval     import run as run_exp2
    from experiments.exp3_extended_analysis import run as run_exp3
    from experiments.exp4_defense_comparison import run as run_exp4

    # ── Experiment 1 ──────────────────────────────────────
    header("Experiment 1 — Attack Taxonomy (Naive Agent, 5 scenarios)")
    t1 = time.perf_counter()
    r1 = run_exp1()
    print(f"\n  Done in {time.perf_counter() - t1:.1f}s")

    # ── Experiment 2 ──────────────────────────────────────
    header("Experiment 2 — Defense Effectiveness (StruQ vs Naive)")
    t2 = time.perf_counter()
    r2 = run_exp2()
    print(f"\n  Done in {time.perf_counter() - t2:.1f}s")

    # ── Experiment 3 ──────────────────────────────────────
    header("Experiment 3 — Extended Analysis (60 scenarios, Ensemble LLM)")
    t3 = time.perf_counter()
    r3 = run_exp3()
    print(f"\n  Done in {time.perf_counter() - t3:.1f}s")

    # ── Experiment 4 ──────────────────────────────────────
    header("Experiment 4 — Full Defense Comparison (5 defenses × 60 scenarios)")
    t4 = time.perf_counter()
    r4 = run_exp4()
    print(f"\n  Done in {time.perf_counter() - t4:.1f}s")

    # ── Consolidated summary ──────────────────────────────
    # Determine best attack (highest ensemble ASR from Exp3 bootstrap)
    best_attack = "N/A"
    best_attack_asr = 0.0
    if r3 and "bootstrap_ci_by_attack_type" in r3:
        for at, b in r3["bootstrap_ci_by_attack_type"].items():
            if b["mean"] > best_attack_asr:
                best_attack_asr = b["mean"]
                best_attack = at

    # Determine best defense (highest DSR from Exp4)
    best_defense = "N/A"
    best_defense_dsr = 0.0
    if r4 and "defense_results" in r4:
        for name, dr in r4["defense_results"].items():
            if name == "Naive":
                continue
            if dr.get("overall_dsr", 0.0) > best_defense_dsr:
                best_defense_dsr = dr["overall_dsr"]
                best_defense = name

    # Vulnerability score: mean ASR across all attacks and defenses (Exp4 naive baseline)
    vuln_score = r4["defense_results"]["Naive"]["overall_asr"] if r4 else r2.get("asr_naive", 0.0)

    print(f"""
{'='*65}
  CONSOLIDATED RESULTS — BidPoison Full Suite
{'='*65}

  --- Baseline (Exp1/2, 5 scenarios, mock LLM) ---
  Naive Agent ASR:             {r2['asr_naive']:.2%}
  StruQ Agent ASR:             {r2['asr_struq']:.2%}
  StruQ Defense Success Rate:  {r2['dsr_struq']:.2%}
  Suspicious Detection Rate:   {r2['sdr_struq']:.2%}
  False Positive Rate:         {r2['false_positive_rate']:.2%}
  Decision Consistency:        {r2['decision_consistency']:.2%}

  --- Extended Analysis (Exp3, 60 scenarios, Ensemble LLM) ---
  Best Attack (by ASR):        {best_attack}  (ASR={best_attack_asr:.3f})
  Sensitivity — High:          {r3['sensitivity_analysis'].get('high', {}).get('asr', 0):.3f}
  Sensitivity — Medium:        {r3['sensitivity_analysis'].get('medium', {}).get('asr', 0):.3f}
  Sensitivity — Low:           {r3['sensitivity_analysis'].get('low', {}).get('asr', 0):.3f}
  Most Vulnerable Category:    {max(r3['category_breakdown'].items(), key=lambda x: x[1]['asr'])[0] if r3 and 'category_breakdown' in r3 else 'N/A'}

  --- Defense Comparison (Exp4, 5 defenses, 60 scenarios) ---
  Best Defense (by DSR):       {best_defense}  (DSR={best_defense_dsr:.3f})
  Overall Vulnerability Score: {vuln_score:.3f}

  ASR by Defense:
    Naive ASR:                 {r4['defense_results']['Naive']['overall_asr']:.3f}
    StruQ ASR:                 {r4['defense_results']['StruQ']['overall_asr']:.3f}
    OutputValidation ASR:      {r4['defense_results']['OutputValidation']['overall_asr']:.3f}
    SemanticSimilarity ASR:    {r4['defense_results']['SemanticSimilarity']['overall_asr']:.3f}
    DefenseChain ASR:          {r4['defense_results']['DefenseChain']['overall_asr']:.3f}

  Total runtime: {time.perf_counter() - t0:.1f}s
  Results  → {RESULTS_DIR}/
{'='*65}
""")

    log = {
        "exp1": r1,
        "exp2": r2,
        "exp3_summary": {
            "n_scenarios":    r3.get("n_scenarios"),
            "model_profile":  r3.get("model_profile"),
            "best_attack":    best_attack,
            "best_attack_asr": round(best_attack_asr, 4),
            "sensitivity":    r3.get("sensitivity_analysis"),
        },
        "exp4_summary": {
            "n_scenarios":    r4.get("n_scenarios"),
            "best_defense":   best_defense,
            "best_defense_dsr": round(best_defense_dsr, 4),
            "vuln_score":     round(vuln_score, 4),
            "defense_asr":    {
                name: dr.get("overall_asr")
                for name, dr in r4["defense_results"].items()
            },
        },
    }
    with open(os.path.join(RESULTS_DIR, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Full log saved → {RESULTS_DIR}/run_log.json")


if __name__ == "__main__":
    main()
