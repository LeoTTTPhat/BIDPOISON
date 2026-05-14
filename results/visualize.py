"""
BidPoison — Paper Figure Generator
====================================
Generates publication-ready figures from experiment results.

  Figure 1 — Attack taxonomy ASR with bootstrap CI (Exp3)
  Figure 2 — Defense comparison heatmap (Exp4)
  Figure 3 — Position × attack type vulnerability heatmap (Exp3)
  Figure 4 — Category breakdown bar chart (Exp3)

Usage:
    python results/visualize.py
    → figures/*.pdf
"""

import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[visualize] matplotlib not found; skipping figure generation")

RESULTS_DIR = os.path.dirname(__file__)
FIGS_DIR    = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGS_DIR, exist_ok=True)


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        print(f"  [warn] {name} not found, skipping.")
        return {}
    with open(p) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────
# Figure 1: Attack Taxonomy ASR with Bootstrap CI
# ──────────────────────────────────────────────────────────

def fig1_attack_taxonomy():
    if not MATPLOTLIB_OK: return
    data = _load("exp3_extended_analysis.json")
    if not data: return

    bci = data.get("bootstrap_ci_by_attack_type", {})
    if not bci: return

    attacks = list(bci.keys())
    means   = [bci[a]["mean"] * 100 for a in attacks]
    lowers  = [bci[a]["ci_lower"] * 100 for a in attacks]
    uppers  = [bci[a]["ci_upper"] * 100 for a in attacks]
    yerr_lo = [m - l for m, l in zip(means, lowers)]
    yerr_hi = [u - m for m, u in zip(means, uppers)]

    labels = [a.replace("_", "\n") for a in attacks]
    colors = ["#F44336", "#FF9800", "#9C27B0", "#2196F3", "#FF5722"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(attacks)), means, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.errorbar(range(len(attacks)), means,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="black", capsize=6, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attack Success Rate (%) with 95% CI", fontsize=11)
    ax.set_title("BidPoison: Attack Taxonomy — ASR by Type\n(65 Scenarios, Ensemble LLM Simulator)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + yerr_hi[i] + 1.5, f"{m:.1f}%", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "fig1_attack_taxonomy_asr.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────
# Figure 2: Defense Comparison
# ──────────────────────────────────────────────────────────

def fig2_defense_comparison():
    if not MATPLOTLIB_OK: return
    data = _load("exp4_defense_comparison.json")
    if not data: return

    dr = data.get("defense_results", {})
    if not dr: return

    defenses = list(dr.keys())
    asr_vals = [dr[d]["overall_asr"] * 100 for d in defenses]
    dsr_vals = [dr[d]["overall_dsr"] * 100 for d in defenses]
    fpr_vals = [dr[d]["fpr"] * 100 for d in defenses]

    x = np.arange(len(defenses))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width, asr_vals, width, label="Attack Success Rate (ASR ↓)",
                color="#F44336", edgecolor="black", linewidth=0.5, alpha=0.85)
    b2 = ax.bar(x,          dsr_vals, width, label="Defense Success Rate (DSR ↑)",
                color="#4CAF50", edgecolor="black", linewidth=0.5, alpha=0.85)
    b3 = ax.bar(x + width,  fpr_vals, width, label="False Positive Rate (FPR ↓)",
                color="#FF9800", edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(defenses, fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("BidPoison: Defense Comparison Across 5 Configurations",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    # Annotate DSR bars
    for i, v in enumerate(dsr_vals):
        ax.text(x[i], v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold",
                color="#1B5E20")

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "fig2_defense_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────
# Figure 3: Position × Attack Heatmap
# ──────────────────────────────────────────────────────────

def fig3_position_heatmap():
    if not MATPLOTLIB_OK: return
    data = _load("exp3_extended_analysis.json")
    if not data: return

    hm = data.get("position_heatmap", {})
    if not hm: return

    positions = list(hm.keys())
    attacks   = list(hm[positions[0]].keys()) if positions else []
    if not attacks: return

    matrix = np.array([[hm[pos][att] for att in attacks] for pos in positions])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([a.replace("_", "\n") for a in attacks], fontsize=8.5)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=9)
    ax.set_title("ASR by Injection Position × Attack Type", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Attack Success Rate")

    for i in range(len(positions)):
        for j in range(len(attacks)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if matrix[i,j] > 0.35 else "black")

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "fig3_position_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────
# Figure 4: Vulnerability by Procurement Category
# ──────────────────────────────────────────────────────────

def fig4_category_breakdown():
    if not MATPLOTLIB_OK: return
    data = _load("exp3_extended_analysis.json")
    if not data: return

    cb = data.get("category_breakdown", {})
    if not cb: return

    # cb values may be dicts with 'asr' key or plain floats
    def _asr(v):
        return v["asr"] if isinstance(v, dict) else v
    categories = sorted(cb.keys(), key=lambda k: _asr(cb[k]), reverse=True)
    values     = [_asr(cb[c]) * 100 for c in categories]
    labels     = [c.replace("_", "\n") for c in categories]
    colors     = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(categories)))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(categories)), values, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean ASR (%)", fontsize=11)
    ax.set_title("Vulnerability by Procurement Category (Ensemble LLM, 65 Scenarios)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "fig4_category_vulnerability.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def run_all():
    print("\n[visualize] Generating BidPoison paper figures …")
    fig1_attack_taxonomy()
    fig2_defense_comparison()
    fig3_position_heatmap()
    fig4_category_breakdown()
    print(f"\n[visualize] Figures written to: {FIGS_DIR}/")


if __name__ == "__main__":
    run_all()
