"""Generate scaling plots for Qwen3 baseline vs Staged-Competence across sizes.

Two PDFs in same style as existing figures:
  documents/images/qwen_scaling_ood_safety.pdf  -- avg OOD violation rate vs model size
  documents/images/qwen_scaling_attacks.pdf     -- avg attack ASR vs model size
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/user_data/sandeep3/curriculum-safety")
OUT = ROOT / "documents" / "images"

# Sizes
SIZES = ["1.7B", "4B", "8B"]
X = np.arange(len(SIZES))

# Per-benchmark numbers (% violation rate), [baseline, curri-pacing]
OOD = {
    "Sorry": [(8.22, 5.33),  (27.56, 14.44), (29.56, 8.89)],
    "Adv":   [(0.19, 0.19),  (14.04, 3.27),  (38.46, 0.38)],
    "HEx":   [(2.67, 0.67),  (20.00, 5.67),  (30.67, 2.67)],
}
# Avg per size
ood_baseline = [np.mean([OOD[k][i][0] for k in OOD]) for i in range(3)]
ood_curri    = [np.mean([OOD[k][i][1] for k in OOD]) for i in range(3)]

# Attack ASR (%)
ATTACKS = {
    "Prefill": [(19.25, 13.25), (50.00, 18.00), (51.74, 16.20)],
    "GCG":     [(5.78,  2.01),  (25.13, 5.03),  (26.88, 8.29)],
}
atk_baseline = [np.mean([ATTACKS[k][i][0] for k in ATTACKS]) for i in range(3)]
atk_curri    = [np.mean([ATTACKS[k][i][1] for k in ATTACKS]) for i in range(3)]


def plot(baseline, curri, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(X, baseline, color="#1f77b4", linestyle="-", linewidth=2.2,
            marker="o", markersize=8, label="Baseline DPO")
    ax.plot(X, curri,    color="#d62728", linestyle="-", linewidth=2.2,
            marker="s", markersize=8, label="Staged-Competence")
    # Shade the gap
    ax.fill_between(X, baseline, curri, color="#d62728", alpha=0.08)
    # Per-point delta annotations. The 1.7B point is special-cased because the
    # baseline line rises steeply from there and the curri line stays low, so
    # naive placement causes overlap with the lines/markers at LaTeX render size.
    # Last-point label is right-anchored so it doesn't run off the plot edge.
    ymax = max(max(baseline), max(curri))
    for i in range(len(SIZES)):
        delta = baseline[i] - curri[i]
        if i == 0 and delta < 3.0:
            # 1.7B with tiny gap: place label well above baseline so the
            # steeply-rising line does not cross through it.
            ha, dx = "left", 0.04
            y = baseline[i] + 0.16 * ymax
            va = "bottom"
        elif i == 0:
            # 1.7B with moderate gap: place between lines but far enough right
            # that the diverging lines have separated comfortably. Manually set
            # y so the label sits clearly above the curri line.
            ha, dx = "left", 0.18
            y = baseline[i] + 0.5
            va = "center"
        elif i == len(SIZES) - 1:
            ha, dx = "right", -0.05
            y = (baseline[i] + curri[i]) / 2
            va = "center"
        else:
            ha, dx = "center", 0.0
            y = (baseline[i] + curri[i]) / 2
            va = "center"
        ax.annotate(f"$-${delta:.1f}", xy=(X[i] + dx, y),
                    ha=ha, va=va, fontsize=9, fontweight="bold",
                    color="#7d1212")
    ax.set_xticks(X)
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.2, 2.2)  # extra breathing room on left/right for end-point labels
    ax.set_xlabel("Qwen3 model size", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    # Add headroom on top so the highest line doesn't touch the upper border
    ymax = max(max(baseline), max(curri))
    ax.set_ylim(bottom=0, top=ymax * 1.22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.17, top=0.96)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot(ood_baseline, ood_curri,
         "Avg OOD safety violation rate (%)",
         OUT / "qwen_scaling_ood_safety.pdf")
    plot(atk_baseline, atk_curri,
         "Avg attack success rate (%)",
         OUT / "qwen_scaling_attacks.pdf")
    print("\n=== Numbers used ===")
    print(f"OOD baseline: {[round(x,2) for x in ood_baseline]}")
    print(f"OOD curri   : {[round(x,2) for x in ood_curri]}")
    print(f"Atk baseline: {[round(x,2) for x in atk_baseline]}")
    print(f"Atk curri   : {[round(x,2) for x in atk_curri]}")


if __name__ == "__main__":
    main()
