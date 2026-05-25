"""
Generate DPO training loss curves for comparison across curriculum strategies.

Loads trainer_state.json from multiple model checkpoints and plots loss over steps.

Usage:
  python -m src.eval.plot_training_loss \
    --baseline outputs/exp_004/dpo_baseline \
    --curriculum outputs/exp_004/dpo_curriculum \
    --curriculum-ratio outputs/exp_004_ratio/dpo_curriculum \
    --output results/exp_004/training_loss_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    """Load trainer_state.json from the latest checkpoint inside a model directory.

    HuggingFace Trainer saves trainer_state.json inside numbered checkpoint
    subdirectories (e.g. checkpoint-1986/). We find the highest-numbered one
    to get the full training history.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Look for checkpoint-N subdirectories and pick the highest step
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )

    for ckpt in reversed(checkpoints):
        candidate = ckpt / "trainer_state.json"
        if candidate.exists():
            print(f"  Using trainer_state from {ckpt.name}")
            with open(candidate) as f:
                return json.load(f)

    raise FileNotFoundError(f"No trainer_state.json found in any checkpoint under {checkpoint_dir}")


def load_curri_dpo_trainer_states(checkpoint_dir: Path) -> tuple[list[dict], list[int]]:
    """Load trainer states from a Curri-DPO output directory (stage_0/, stage_1/, ...).

    Returns (list of trainer_state dicts in stage order, list of stage boundary steps).
    """
    checkpoint_dir = Path(checkpoint_dir)
    stage_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("stage_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not stage_dirs:
        raise FileNotFoundError(f"No stage_N/ subdirectories found under {checkpoint_dir}")

    states = []
    for stage_dir in stage_dirs:
        state = load_trainer_state(stage_dir)
        states.append(state)
        print(f"  Loaded stage {stage_dir.name}: {len(state.get('log_history', []))} log entries")

    # Boundary steps: cumulative step count at end of each stage except last
    boundaries = []
    cumulative = 0
    for state in states[:-1]:
        logs = [e for e in state.get("log_history", []) if "loss" in e]
        if logs:
            cumulative += max(e["step"] for e in logs)
        boundaries.append(cumulative)

    return states, boundaries


def extract_loss_curve_stitched(states: list[dict], boundaries: list[int]) -> tuple[list[int], list[float]]:
    """Stitch per-stage loss curves into a single curve with globally increasing steps."""
    all_steps: list[int] = []
    all_losses: list[float] = []
    offset = 0

    for i, state in enumerate(states):
        logs = [e for e in state.get("log_history", []) if "loss" in e]
        if not logs:
            continue
        stage_steps = [e["step"] for e in logs]
        stage_losses = [e["loss"] for e in logs]
        all_steps.extend(s + offset for s in stage_steps)
        all_losses.extend(stage_losses)
        offset += max(stage_steps)  # next stage steps start after this stage's last step

    return all_steps, all_losses


def extract_loss_curve(trainer_state: dict) -> tuple[list[int], list[float]]:
    """Extract (step, loss) pairs from trainer state."""
    log_history = trainer_state.get("log_history", [])

    steps = []
    losses = []

    for log in log_history:
        if "loss" in log:
            steps.append(log.get("step", len(steps)))
            losses.append(log["loss"])

    return steps, losses


def plot_loss_curves(
    models: dict[str, Path],
    output_path: Path,
) -> None:
    """Plot loss curves for multiple models."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"baseline": "#1f77b4", "curriculum": "#ff7f0e", "curriculum_ratio": "#2ca02c"}
    line_styles = {"baseline": "-", "curriculum": "--", "curriculum_ratio": ":"}

    for model_name, checkpoint_dir in models.items():
        print(f"Loading {model_name} from {checkpoint_dir}...")
        try:
            checkpoint_dir = Path(checkpoint_dir)
            # Detect Curri-DPO: contains stage_N/ subdirectories
            stage_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("stage_")]
            if stage_dirs:
                states, boundaries = load_curri_dpo_trainer_states(checkpoint_dir)
                steps, losses = extract_loss_curve_stitched(states, boundaries)
            else:
                trainer_state = load_trainer_state(checkpoint_dir)
                steps, losses = extract_loss_curve(trainer_state)
                boundaries = []

            if not steps:
                print(f"  WARNING: No loss history found for {model_name}")
                continue

            print(f"  Loaded {len(steps)} training steps")
            ax.plot(
                steps,
                losses,
                label=model_name,
                color=colors.get(model_name, "#555"),
                linestyle=line_styles.get(model_name, "-"),
                linewidth=2,
            )
            # Draw vertical dashed lines at stage boundaries for curri-dpo
            for b in boundaries:
                ax.axvline(x=b, color=colors.get(model_name, "#555"), linestyle=":", alpha=0.5, linewidth=1)

        except Exception as e:
            print(f"  ERROR: {e}")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("DPO Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    plt.close()


def load_eval_margin_jsonl(path: Path) -> tuple[list[int], list[float], list[float], list[int]]:
    """Read eval_margin.jsonl produced by TestMarginCallback.

    Returns (steps, accuracies, mean_margins, stage_boundaries) where stage
    boundaries are the global_step values where step_offset changes (curri-pacing).
    """
    if not path.exists():
        return [], [], [], []
    steps: list[int] = []
    accs: list[float] = []
    margins: list[float] = []
    offsets_seen: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps.append(int(d["global_step"]))
            accs.append(float(d["accuracy"]))
            margins.append(float(d["mean_margin"]))
            offset = int(d.get("step_offset", 0))
            if offset > 0 and offset not in offsets_seen:
                offsets_seen.append(offset)
    offsets_seen.sort()
    return steps, accs, margins, offsets_seen


def plot_test_margin_curves(models: dict[str, Path], output_path: Path) -> None:
    """Plot test-set reward accuracy + mean margin from eval_margin.jsonl files.

    `models` maps display name → adapter directory (the parent of eval_margin.jsonl).
    Produces a 1×2 subplot figure: left=accuracy, right=mean margin.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"baseline": "#1f77b4", "curriculum": "#ff7f0e", "curriculum_ratio": "#2ca02c"}
    line_styles = {"baseline": "-", "curriculum": "--", "curriculum_ratio": ":"}

    any_data = False
    for model_name, ckpt_dir in models.items():
        jsonl_path = Path(ckpt_dir) / "eval_margin.jsonl"
        steps, accs, margins, boundaries = load_eval_margin_jsonl(jsonl_path)
        if not steps:
            print(f"  [test_margin] no eval_margin.jsonl for {model_name} at {jsonl_path}")
            continue
        any_data = True
        print(f"  [test_margin] loaded {len(steps)} eval points for {model_name}")
        color = colors.get(model_name, "#555")
        ls = line_styles.get(model_name, "-")
        axes[0].plot(steps, accs, label=model_name, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4)
        axes[1].plot(steps, margins, label=model_name, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4)
        for b in boundaries:
            for ax in axes:
                ax.axvline(x=b, color=color, linestyle=":", alpha=0.5, linewidth=1)

    if not any_data:
        plt.close(fig)
        print(f"  [test_margin] no eval_margin.jsonl found in any model dir; skipping plot")
        return

    axes[0].set_xlabel("Training Step", fontsize=12)
    axes[0].set_ylabel("Test Reward Accuracy", fontsize=12)
    axes[0].set_title("Test-set Reward Accuracy", fontsize=13, fontweight="bold")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=10)

    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("Mean Margin (log p chosen − log p rejected)", fontsize=12)
    axes[1].set_title("Test-set Mean Reward Margin", fontsize=13, fontweight="bold")
    axes[1].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved test margin plot to {output_path}")
    plt.close(fig)


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Plot DPO training loss curves for curriculum comparison.")
    p.add_argument(
        "--baseline",
        type=Path,
        default=root / "outputs" / "exp_004" / "dpo_baseline",
        help="Path to baseline DPO checkpoint",
    )
    p.add_argument(
        "--curriculum",
        type=Path,
        default=root / "outputs" / "exp_004" / "dpo_curriculum",
        help="Path to curriculum (similarity) DPO checkpoint",
    )
    p.add_argument(
        "--curriculum-ratio",
        type=Path,
        default=None,
        help="Path to curriculum (ratio) DPO checkpoint (optional)",
    )
    p.add_argument("--output", type=Path, default=root / "results" / "exp_004" / "training_loss_curves.png", help="Output PNG path")
    p.add_argument(
        "--baseline-margin-dir",
        type=Path,
        default=None,
        help="Path to a *different* exp's dpo_baseline adapter dir whose eval_margin.jsonl "
             "should be used for the test-margin plot (use when baseline and curriculum are in separate exps).",
    )
    args = p.parse_args()

    # Build model dict
    models = {}
    if args.baseline.exists():
        models["baseline"] = args.baseline
    if args.curriculum.exists():
        models["curriculum"] = args.curriculum
    if args.curriculum_ratio and args.curriculum_ratio.exists():
        models["curriculum_ratio"] = args.curriculum_ratio

    if not models:
        raise ValueError(f"No valid checkpoint directories found. Check paths.")

    plot_loss_curves(models, args.output)

    # Build margin model dict — optionally override baseline path for test-margin plot
    margin_models = dict(models)
    if args.baseline_margin_dir is not None and args.baseline_margin_dir.exists():
        margin_models["baseline"] = args.baseline_margin_dir
    elif args.baseline_margin_dir is not None:
        print(f"  [test_margin] WARNING: --baseline-margin-dir {args.baseline_margin_dir} not found, skipping baseline margin curve")
        margin_models.pop("baseline", None)

    margin_output = args.output.with_name("test_margin_curves.png")
    plot_test_margin_curves(margin_models, margin_output)


if __name__ == "__main__":
    cli()
