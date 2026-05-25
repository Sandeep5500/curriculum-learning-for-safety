"""Generate 6 individual PDF plots for the learning_curves figure.

Each plot is a single subplot (no title, no embedded model name) so they can be
arranged in a 3-col x 2-row LaTeX table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/data/user_data/sandeep3/curriculum-safety")
OUT = ROOT / "documents" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# (model_short, baseline_exp, curri_pacing_exp, model_dir_name)
MODELS = [
    ("llama3", "exp_171", "exp_175", "llama3_v2"),
    ("qwen3",  "exp_163", "exp_164", "qwen3-8b-v3"),
    ("gemma3", "exp_129", "exp_148", "gemma3_4b"),
    ("yi",     "exp_191", "exp_205", "yi15_9b"),
]


def load_jsonl(path: Path):
    steps, accs, margins, offsets = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps.append(int(d["global_step"]))
            accs.append(float(d["accuracy"]))
            margins.append(float(d["mean_margin"]))
            off = int(d.get("step_offset", 0))
            if off > 0 and off not in offsets:
                offsets.append(off)
    offsets.sort()
    return steps, accs, margins, offsets


def plot_one(steps_b, vals_b, steps_c, vals_c, boundaries, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(steps_b, vals_b, color="#1f77b4", linestyle="-",  linewidth=2,
            marker="o", markersize=3.5, label="Baseline DPO")
    ax.plot(steps_c, vals_c, color="#d62728", linestyle="-",  linewidth=2,
            marker="s", markersize=3.5, label="Staged-Competence")
    for b in boundaries:
        ax.axvline(x=b, color="#d62728", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    # Symmetric horizontal padding so the data area is centered within the saved PDF
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.96)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    for short, b_exp, c_exp, mdir in MODELS:
        b_path = ROOT / "outputs" / b_exp / "combined_pku_hh" / mdir / "dpo_baseline" / "eval_margin.jsonl"
        c_path = ROOT / "outputs" / c_exp / "combined_pku_hh" / mdir / "dpo_curriculum" / "eval_margin.jsonl"
        print(f"\n{short}: baseline={b_path.parent.parent.name}, curri={c_path.parent.parent.name}")

        steps_b, accs_b, margs_b, _ = load_jsonl(b_path)
        steps_c, accs_c, margs_c, boundaries = load_jsonl(c_path)

        # Accuracy plot (as percentage)
        plot_one(
            steps_b, [a * 100 for a in accs_b],
            steps_c, [a * 100 for a in accs_c],
            boundaries,
            "Test Reward Accuracy (%)",
            OUT / f"{short}_accuracy.pdf",
        )
        # Margin plot
        plot_one(
            steps_b, margs_b,
            steps_c, margs_c,
            boundaries,
            "Mean Test Margin",
            OUT / f"{short}_margin.pdf",
        )


if __name__ == "__main__":
    main()
