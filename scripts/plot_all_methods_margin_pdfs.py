"""Generate 5-line individual PDF plots for the learning_curves figure.

Each plot is a single subplot (no title) with all 5 methods overlaid:
Standard DPO, Sequential, Sqrt-Pacing, Curri-DPO, and Staged-Competence.
Output paths match the existing 2-line plots (overwriting them).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/data/user_data/sandeep3/curriculum-safety")
OUT = ROOT / "documents" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Per-model: (short_name, model_dir_name, exps_by_method)
# Each method's exp ID points to the run with eval_margin.jsonl.
MODELS = [
    {
        "short": "llama3",
        "mdir": "llama3_v2",
        "exps": {
            "Standard DPO":      ("exp_171", "dpo_baseline"),
            "Sequential":        ("exp_178", "dpo_curriculum"),
            "Sqrt-Competence":   ("exp_179", "dpo_curriculum"),
            "Curri-DPO":         ("exp_180", "dpo_curriculum"),
            "Staged-Competence": ("exp_175", "dpo_curriculum"),
        },
    },
    {
        "short": "qwen3",
        "mdir": "qwen3-8b-v3",
        "exps": {
            "Standard DPO":      ("exp_163", "dpo_baseline"),
            "Sequential":        ("exp_181", "dpo_curriculum"),
            "Sqrt-Competence":   ("exp_182", "dpo_curriculum"),
            "Curri-DPO":         ("exp_183", "dpo_curriculum"),
            "Staged-Competence": ("exp_164", "dpo_curriculum"),
        },
    },
    {
        "short": "gemma3",
        "mdir": "gemma3_4b",
        "exps": {
            "Standard DPO":      ("exp_129", "dpo_baseline"),
            "Sequential":        ("exp_184", "dpo_curriculum"),
            "Sqrt-Competence":   ("exp_185", "dpo_curriculum"),
            "Curri-DPO":         ("exp_186", "dpo_curriculum"),
            "Staged-Competence": ("exp_148", "dpo_curriculum"),
        },
    },
    {
        "short": "yi",
        "mdir": "yi15_9b",
        "exps": {
            "Standard DPO":      ("exp_191", "dpo_baseline"),
            "Sequential":        ("exp_200", "dpo_curriculum"),
            "Sqrt-Competence":   ("exp_201", "dpo_curriculum"),
            "Curri-DPO":         ("exp_192", "dpo_curriculum"),
            "Staged-Competence": ("exp_204", "dpo_curriculum"),
        },
    },
]

# Plot styling per method (color / linestyle / marker)
STYLES = {
    "Standard DPO":      dict(color="#1f77b4", linestyle="-",  marker="o"),
    "Sequential":        dict(color="#2ca02c", linestyle="--", marker="^"),
    "Sqrt-Competence":   dict(color="#ff7f0e", linestyle="-.", marker="D"),
    "Curri-DPO":         dict(color="#9467bd", linestyle="--", marker="x"),
    "Staged-Competence": dict(color="#d62728", linestyle="-",  marker="s"),
}

METHOD_ORDER = ["Standard DPO", "Sequential", "Sqrt-Competence", "Curri-DPO", "Staged-Competence"]


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


def plot_one(series_by_method, boundaries, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for method in METHOD_ORDER:
        if method not in series_by_method:
            continue
        steps, vals = series_by_method[method]
        style = STYLES[method]
        ax.plot(steps, vals,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=3.2,
                label=method,
                alpha=0.95)
    for b in boundaries:
        ax.axvline(x=b, color="#d62728", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True, ncol=1)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.96)
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    for model in MODELS:
        short = model["short"]
        mdir = model["mdir"]
        print(f"\n=== {short} ===")

        acc_series = {}
        margin_series = {}
        boundaries: list[int] = []

        for method, (exp_id, subdir) in model["exps"].items():
            path = ROOT / "outputs" / exp_id / "combined_pku_hh" / mdir / subdir / "eval_margin.jsonl"
            if not path.exists():
                print(f"  [WARN] missing: {path}")
                continue
            steps, accs, margs, offs = load_jsonl(path)
            acc_series[method] = (steps, [a * 100 for a in accs])
            margin_series[method] = (steps, margs)
            print(f"  {method:20s} ({exp_id}): {len(steps)} points")
            # Use Staged-Competence's stage boundaries for the vertical lines
            if method == "Staged-Competence" and offs:
                boundaries = offs

        plot_one(acc_series, boundaries,
                 "Test Reward Accuracy (%)",
                 OUT / f"{short}_accuracy.pdf")
        plot_one(margin_series, boundaries,
                 "Mean Test Margin",
                 OUT / f"{short}_margin.pdf")


if __name__ == "__main__":
    main()
