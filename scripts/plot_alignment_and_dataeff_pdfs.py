"""Generate individual PDF plots for alignment depth (3) and data efficiency (4).

No titles, no embedded model names. Sized for a 3-col / 2-col LaTeX tabular grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/user_data/sandeep3/curriculum-safety")
OUT = ROOT / "documents" / "images"

# === Alignment depth: (model_short, exp_dir, model_dir) ===
ALIGN = [
    ("llama3", "exp_175_exp_171",  "llama3_v2"),
    ("qwen3",  "exp_164_exp_163",  "qwen3-8b-v3"),
    ("yi",     "exp_205_exp_191",  "yi15_9b"),
]

# === Data efficiency: (model_short, baseline_exp, curri_50, curri_75, curri_100, model_dir) ===
DATAEFF = [
    ("llama3", "exp_171", "exp_176", "exp_177", "exp_175", "llama3_v2"),
    ("qwen3",  "exp_163", "exp_168", "exp_169", "exp_164", "qwen3-8b-v3"),
]


def load_eval_margin(path: Path):
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


def plot_alignment(short, exp_dir, mdir):
    p = ROOT / "results" / exp_dir / "combined_pku_hh" / mdir / "alignment_depth" / "alignment_depth_results.json"
    d = json.load(open(p))
    base = np.array(d["baseline"]["mean_delta"])
    staged = np.array(d["curri_pacing"]["mean_delta"])
    # Apply same 5-window moving average smoothing as the original alignment_depth.py
    window = 5
    base = np.convolve(base, np.ones(window) / window, mode="same")
    staged = np.convolve(staged, np.ones(window) / window, mode="same")
    positions = np.arange(len(base))

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(positions, base,   color="#1f77b4", linewidth=2, label="Baseline DPO")
    ax.plot(positions, staged, color="#d62728", linewidth=2, label="Staged-Competence")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Token Position in Response", fontsize=11)
    ax.set_ylabel(r"Suppression $\delta(t)$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.96)
    out = OUT / f"{short}_alignment_depth.pdf"
    plt.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  -> {out}")


def plot_data_eff(short, b_exp, c50_exp, c75_exp, c100_exp, mdir):
    paths = {
        "Baseline DPO (100%)":          ROOT / "outputs" / b_exp   / "combined_pku_hh" / mdir / "dpo_baseline"   / "eval_margin.jsonl",
        "Staged-Competence (50%)":      ROOT / "outputs" / c50_exp / "combined_pku_hh" / mdir / "dpo_curriculum" / "eval_margin.jsonl",
        "Staged-Competence (75%)":      ROOT / "outputs" / c75_exp / "combined_pku_hh" / mdir / "dpo_curriculum" / "eval_margin.jsonl",
        "Staged-Competence (100%)":     ROOT / "outputs" / c100_exp/ "combined_pku_hh" / mdir / "dpo_curriculum" / "eval_margin.jsonl",
    }
    style = {
        "Baseline DPO (100%)":      ("#1f77b4", "-",  "o"),
        "Staged-Competence (50%)":  ("#ff7f0e", "-",  "s"),
        "Staged-Competence (75%)":  ("#2ca02c", "-",  "^"),
        "Staged-Competence (100%)": ("#d62728", "-",  "D"),
    }

    series = {label: load_eval_margin(p) for label, p in paths.items()}

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for label, (steps, accs, _, _) in series.items():
        c, ls, m = style[label]
        ax.plot(steps, [a * 100 for a in accs], color=c, linestyle=ls, linewidth=1.8,
                marker=m, markersize=3, label=label)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Test Reward Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.96)
    out = OUT / f"{short}_dataeff_accuracy.pdf"
    plt.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  -> {out}")

    # Margin plot
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for label, (steps, _, margins, _) in series.items():
        c, ls, m = style[label]
        ax.plot(steps, margins, color=c, linestyle=ls, linewidth=1.8,
                marker=m, markersize=3, label=label)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Test Margin", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.17, top=0.96)
    out = OUT / f"{short}_dataeff_margin.pdf"
    plt.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("Alignment depth:")
    for short, exp_dir, mdir in ALIGN:
        plot_alignment(short, exp_dir, mdir)
    print("\nData efficiency:")
    for short, b, c50, c75, c100, mdir in DATAEFF:
        plot_data_eff(short, b, c50, c75, c100, mdir)


if __name__ == "__main__":
    main()
