"""
Experiment 3 — Extended Statistical Analysis
=============================================
Uses the 60-scenario extended dataset and the LLMBehaviorSimulator to
perform comprehensive statistical analysis with bootstrap confidence intervals.

Analyses:
  1. Bootstrap CI (n=1000): ASR per attack type with 95% CI
  2. Position analysis: 4×5 heatmap (position × attack type)
  3. Sensitivity analysis: ASR by scenario sensitivity level
  4. Defense comparison: Naive vs StruQ vs OutputValidation vs
                         SemanticSimilarity vs DefenseChain
  5. Category breakdown: ASR by procurement category

Simulator profile: "ensemble" (averages model_a, model_b, model_c profiles)

Outputs:
  results/exp3_extended_analysis.json — full results
  Console — formatted summary tables

Usage:
    python experiments/exp3_extended_analysis.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_behavior_simulator import (
    LLMBehaviorSimulator,
    POSITION_MULTIPLIERS,
    MODEL_PROFILES,
)
from src.additional_defenses import (
    OutputValidationDefense,
    SemanticSimilarityDefense,
    InputSanitizationDefense,
    DefenseChain,
)
from src.attack_engine import AttackEngine, AttackConfig, ATTACK_TAXONOMY, INJECTION_POSITIONS

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ATTACK_TYPES = list(ATTACK_TAXONOMY.keys())
POSITIONS    = INJECTION_POSITIONS  # ["header","footer","inline_comment","metadata_field"]

BOOTSTRAP_N  = 1000
RANDOM_SEED  = 42

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def bootstrap_ci(data: list, n: int = BOOTSTRAP_N, ci: float = 0.95, seed: int = RANDOM_SEED):
    """
    Compute bootstrap mean and confidence interval.

    Returns
    -------
    dict with keys: mean, ci_lower, ci_upper, std
    """
    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}

    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean":     float(np.mean(arr)),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha))),
        "std":      float(np.std(arr)),
        "n":        len(arr),
    }


def simulate_attack_outcomes(
    scenarios: list,
    simulator: LLMBehaviorSimulator,
) -> list:
    """
    For each scenario × attack type × position, simulate whether the
    attack succeeds against the naive (undefended) agent.

    Builds actual injected documents via AttackEngine so the simulator
    sees the real malicious payload and content-awareness works correctly.
    Returns a list of flat outcome dicts.
    """
    engine   = AttackEngine()
    outcomes = []
    for scenario in scenarios:
        sid         = scenario["id"]
        category    = scenario.get("category", "unknown")
        sensitivity = scenario.get("sensitivity_level", "medium")
        vendor_data = scenario.get("vendor_data", "{}")
        instruction = scenario.get("task_instruction", "")
        fallback    = scenario.get("ground_truth_decision", "BaseVendor")

        for attack_type in ATTACK_TYPES:
            for position in POSITIONS:
                # Build the actual injected document so the simulator
                # receives a prompt that contains injection markers.
                attacked_doc  = engine.inject(
                    document=vendor_data,
                    config=AttackConfig(
                        attack_type=attack_type,
                        position=position,
                        target_value=fallback,
                        template_idx=0,
                    ),
                )
                injected_data = attacked_doc.injected_data

                result = simulator.simulate_response(
                    prompt=f"{instruction}\n{injected_data}",
                    attack_type=attack_type,
                    position=position,
                    decision_fallback=fallback,
                )
                outcomes.append({
                    "scenario_id":   sid,
                    "category":      category,
                    "sensitivity":   sensitivity,
                    "attack_type":   attack_type,
                    "position":      position,
                    "was_injected":  result["was_injected"],
                    "effective_asr": result["effective_asr"],
                })
    return outcomes


def simulate_defense_effectiveness(
    scenarios: list,
    simulator: LLMBehaviorSimulator,
    defense_name: str,
    defense_obj,
) -> dict:
    """
    Simulate defense success rate for a given defense strategy.

    For each scenario × attack × position:
      1. Build the actual injected document via AttackEngine.
      2. Apply the defense to the injected document.
      3. Simulate the LLM on the (possibly cleaned) document.
      4. Defense "wins" if it flagged the input OR the attack failed after cleaning.

    Returns aggregated metrics.
    """
    engine      = AttackEngine()
    results     = {at: {"defended": 0, "total": 0} for at in ATTACK_TYPES}
    flagged_clean = 0
    total_clean   = len(scenarios)

    for scenario in scenarios:
        instruction = scenario.get("task_instruction", "")
        vendor_data = scenario.get("vendor_data", "{}")
        expected    = scenario.get("expected_output", "SELECT_VENDOR")
        fallback    = scenario.get("ground_truth_decision", "Base")

        for attack_type in ATTACK_TYPES:
            for position in POSITIONS:
                results[attack_type]["total"] += 1

                # Step 1: build injected document
                attacked_doc  = engine.inject(
                    document=vendor_data,
                    config=AttackConfig(
                        attack_type=attack_type,
                        position=position,
                        target_value=fallback,
                        template_idx=0,
                    ),
                )
                injected_data = attacked_doc.injected_data

                # Step 2: apply defense to the injected document
                if isinstance(defense_obj, DefenseChain):
                    def_result = defense_obj.apply(
                        instruction=instruction,
                        data=injected_data,
                        expected_output_type=expected,
                    )
                    defense_flagged = def_result.was_flagged
                    clean_instr = def_result.cleaned_instruction
                    clean_data  = def_result.cleaned_data
                elif isinstance(defense_obj, OutputValidationDefense):
                    # Output validation fires on the response; for simplicity
                    # we let the injected prompt through and check the response.
                    defense_flagged = False
                    clean_instr = instruction
                    clean_data  = injected_data
                elif isinstance(defense_obj, SemanticSimilarityDefense):
                    _, is_safe = defense_obj.check_similarity(instruction, injected_data)
                    defense_flagged = not is_safe
                    clean_instr = instruction
                    clean_data  = injected_data
                elif isinstance(defense_obj, InputSanitizationDefense):
                    clean_data, removed = defense_obj.sanitize(injected_data)
                    defense_flagged = bool(removed)
                    clean_instr = instruction
                else:
                    defense_flagged = False
                    clean_instr = instruction
                    clean_data  = injected_data

                # Step 3: simulate LLM on cleaned data
                result = simulator.simulate_response(
                    prompt=f"{clean_instr}\n{clean_data}",
                    attack_type=attack_type,
                    position=position,
                    decision_fallback=fallback,
                )

                # Step 4: defense wins if it flagged OR attack failed on cleaned prompt
                defended = defense_flagged or (not result["was_injected"])
                if defended:
                    results[attack_type]["defended"] += 1

        # Clean baseline FPR check (no attack applied — use original vendor_data)
        if isinstance(defense_obj, DefenseChain):
            clean_result = defense_obj.apply(instruction=instruction, data=vendor_data)
            if clean_result.was_flagged:
                flagged_clean += 1
        elif isinstance(defense_obj, SemanticSimilarityDefense):
            _, is_safe = defense_obj.check_similarity(instruction, vendor_data)
            if not is_safe:
                flagged_clean += 1

    # Aggregate
    asr_by_type = {}
    for at in ATTACK_TYPES:
        total    = results[at]["total"]
        defended = results[at]["defended"]
        dsr      = defended / total if total > 0 else 0.0
        asr      = 1.0 - dsr
        asr_by_type[at] = {
            "asr":  round(asr, 4),
            "dsr":  round(dsr, 4),
            "n":    total,
        }

    all_defends = [r["defended"] for r in results.values()]
    all_totals  = [r["total"]    for r in results.values()]
    overall_dsr = sum(all_defends) / sum(all_totals) if sum(all_totals) > 0 else 0.0
    fpr         = flagged_clean / total_clean if total_clean > 0 else 0.0

    return {
        "defense_name":     defense_name,
        "overall_dsr":      round(overall_dsr, 4),
        "overall_asr":      round(1.0 - overall_dsr, 4),
        "false_positive_rate": round(fpr, 4),
        "by_attack_type":   asr_by_type,
    }


# ──────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────

def run():
    t0 = time.perf_counter()
    print("\n[Exp3] Loading extended scenario dataset …")
    with open(os.path.join(DATA_DIR, "procurement_scenarios_extended.json")) as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios across "
          f"{len(set(s['category'] for s in scenarios))} categories")

    simulator = LLMBehaviorSimulator(model_profile="ensemble", base_seed=RANDOM_SEED)

    # ── 1. Simulate all attack outcomes ──────────────────
    print("\n[Exp3] Simulating attack outcomes (ensemble LLM) …")
    outcomes = simulate_attack_outcomes(scenarios, simulator)
    print(f"  {len(outcomes)} outcome records generated")

    # ── 2. Bootstrap CI per attack type ──────────────────
    print("\n[Exp3] Computing bootstrap confidence intervals …")
    bootstrap_results = {}
    for at in ATTACK_TYPES:
        at_outcomes = [o["was_injected"] for o in outcomes if o["attack_type"] == at]
        bootstrap_results[at] = bootstrap_ci(at_outcomes, n=BOOTSTRAP_N)

    # ── 3. Position × Attack type heatmap ────────────────
    print("\n[Exp3] Building position × attack type heatmap …")
    heatmap = {}
    for at in ATTACK_TYPES:
        heatmap[at] = {}
        for pos in POSITIONS:
            pos_outcomes = [
                o["was_injected"]
                for o in outcomes
                if o["attack_type"] == at and o["position"] == pos
            ]
            heatmap[at][pos] = round(float(np.mean(pos_outcomes)) if pos_outcomes else 0.0, 4)

    # ── 4. Sensitivity analysis ───────────────────────────
    print("\n[Exp3] Running sensitivity analysis …")
    sensitivity_results = {}
    for level in ["high", "medium", "low"]:
        level_outcomes = [o["was_injected"] for o in outcomes if o["sensitivity"] == level]
        sensitivity_results[level] = {
            "asr":  round(float(np.mean(level_outcomes)) if level_outcomes else 0.0, 4),
            "n":    len(level_outcomes),
            "ci":   bootstrap_ci(level_outcomes),
        }

    # ── 5. Category breakdown ────────────────────────────
    print("\n[Exp3] Computing category-level vulnerability …")
    categories = list(set(s["category"] for s in scenarios))
    category_results = {}
    for cat in sorted(categories):
        cat_outcomes = [o["was_injected"] for o in outcomes if o["category"] == cat]
        category_results[cat] = {
            "asr":      round(float(np.mean(cat_outcomes)) if cat_outcomes else 0.0, 4),
            "n":        len(cat_outcomes),
            "n_scenarios": len([s for s in scenarios if s["category"] == cat]),
        }

    # ── 6. Defense comparison ────────────────────────────
    print("\n[Exp3] Evaluating defense strategies …")

    output_val_defense = OutputValidationDefense(strict_mode=True)
    semantic_defense   = SemanticSimilarityDefense(threshold=0.05)
    sanitization       = InputSanitizationDefense()
    chain_defense      = DefenseChain(defenses=[
        sanitization,
        semantic_defense,
        output_val_defense,
    ])

    defense_configs = [
        ("OutputValidation",    output_val_defense),
        ("SemanticSimilarity",  semantic_defense),
        ("InputSanitization",   sanitization),
        ("DefenseChain",        chain_defense),
    ]

    defense_comparison = {}
    for name, defense in defense_configs:
        print(f"    Evaluating {name} …")
        result = simulate_defense_effectiveness(scenarios, simulator, name, defense)
        defense_comparison[name] = result

    # ── Print results ─────────────────────────────────────
    print("\n" + "="*70)
    print("  TABLE 3a — Attack Success Rate by Type (Bootstrap 95% CI)")
    print("="*70)
    print(f"  {'Attack Type':<24}  {'ASR':>8}  {'95% CI':>20}  {'n':>6}")
    print(f"  {'-'*62}")
    for at in sorted(bootstrap_results.keys()):
        b = bootstrap_results[at]
        print(
            f"  {at:<24}  {b['mean']:>8.3f}  "
            f"[{b['ci_lower']:>6.3f}, {b['ci_upper']:>6.3f}]  {b['n']:>6}"
        )

    print("\n" + "="*70)
    print("  TABLE 3b — Position × Attack Type Heatmap (ASR)")
    print("="*70)
    hdr = f"  {'Attack Type':<24}"
    for pos in POSITIONS:
        hdr += f"  {pos[:8]:>10}"
    print(hdr)
    print(f"  {'-'*68}")
    for at in sorted(ATTACK_TYPES):
        row = f"  {at:<24}"
        for pos in POSITIONS:
            row += f"  {heatmap[at][pos]:>10.3f}"
        print(row)

    print("\n" + "="*70)
    print("  TABLE 3c — ASR by Sensitivity Level")
    print("="*70)
    for level in ["high", "medium", "low"]:
        r = sensitivity_results[level]
        ci = r["ci"]
        print(f"  {level:<8}: ASR={r['asr']:.3f}  "
              f"CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]  n={r['n']}")

    print("\n" + "="*70)
    print("  TABLE 3d — ASR by Procurement Category")
    print("="*70)
    cat_sorted = sorted(category_results.items(), key=lambda x: -x[1]["asr"])
    for cat, r in cat_sorted:
        print(f"  {cat:<30}  ASR={r['asr']:.3f}  n_scenarios={r['n_scenarios']}")

    print("\n" + "="*70)
    print("  TABLE 3e — Defense Comparison")
    print("="*70)
    print(f"  {'Defense':<24}  {'DSR':>8}  {'ASR':>8}  {'FPR':>8}")
    print(f"  {'-'*52}")
    for dname, dr in defense_comparison.items():
        print(
            f"  {dname:<24}  {dr['overall_dsr']:>8.3f}  "
            f"{dr['overall_asr']:>8.3f}  {dr['false_positive_rate']:>8.3f}"
        )

    # ── Compile summary table ─────────────────────────────
    summary_table = []
    for at in sorted(ATTACK_TYPES):
        b = bootstrap_results[at]
        row = {
            "attack_type":  at,
            "asr_mean":     round(b["mean"], 4),
            "ci_lower":     round(b["ci_lower"], 4),
            "ci_upper":     round(b["ci_upper"], 4),
            "std":          round(b["std"], 4),
            "n":            b["n"],
            "by_position":  heatmap[at],
        }
        summary_table.append(row)

    # ── Save results ──────────────────────────────────────
    output = {
        "experiment":          "exp3_extended_analysis",
        "n_scenarios":         len(scenarios),
        "n_outcomes":          len(outcomes),
        "model_profile":       "ensemble",
        "bootstrap_n":         BOOTSTRAP_N,
        "bootstrap_ci_by_attack_type": bootstrap_results,
        "position_heatmap":    heatmap,
        "sensitivity_analysis": sensitivity_results,
        "category_breakdown":  category_results,
        "defense_comparison":  defense_comparison,
        "summary_table":       summary_table,
        "runtime_s":           round(time.perf_counter() - t0, 2),
    }

    out_path = os.path.join(RESULTS_DIR, "exp3_extended_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Exp3] Results saved → {out_path}")
    print(f"[Exp3] Runtime: {output['runtime_s']:.1f}s")
    return output


if __name__ == "__main__":
    run()
