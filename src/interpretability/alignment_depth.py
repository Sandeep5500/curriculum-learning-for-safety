"""Alignment Depth Analysis.

Measures whether curriculum training produces "deeper" safety alignment —
i.e., whether the safety signal is distributed across more output tokens
rather than concentrated in the first few.

Inspired by "Safety Alignment Should Be Made More Than Just a Few Tokens Deep"
(ICLR 2025 Outstanding Paper).

For each model variant (baseline DPO, curri-pacing), computes:
  delta(t) = logP_ref(token_t) - logP_aligned(token_t)

over the rejected (unsafe) response tokens. Positive delta means the aligned
model suppresses that token relative to the unaligned base. If curriculum
training spreads this suppression signal deeper into the sequence, that's
a mechanistic explanation for improved prefill robustness.

Run:
    python -m src.interpretability.alignment_depth \
      --base-model informatiker/Llama-3-8B-Instruct-abliterated \
      --baseline-adapter outputs/exp_127/.../dpo_baseline \
      --curri-adapter outputs/exp_122/.../dpo_curriculum \
      --curri-base outputs/exp_122/.../dpo_curriculum/stage_1/merged \
      --test-path data/processed/.../curriculum_dataset_test.jsonl \
      --output results/.../alignment_depth \
      --n-samples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(base_path: str, adapter_path: str | None, device: str):
    """Load base model, optionally with LoRA adapter merged in."""
    is_local = Path(base_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=is_local,
    )
    if adapter_path is not None:
        import json as _json
        import tempfile
        from peft import PeftModel

        cfg_path = Path(adapter_path) / "adapter_config.json"
        cfg = _json.loads(cfg_path.read_text())
        cfg["base_model_name_or_path"] = base_path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "adapter_config.json").write_text(_json.dumps(cfg))
            for f in Path(adapter_path).iterdir():
                if f.name != "adapter_config.json":
                    (tmp_path / f.name).symlink_to(f.resolve())
            model = PeftModel.from_pretrained(model, str(tmp_path))
        model = model.merge_and_unload()

    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def compute_per_token_logprobs(model, tokenizer, prompt: str, response: str,
                                device: str, max_length: int = 512):
    """Compute per-token log-probabilities for the response, conditioned on prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Truncate if needed
    total_max = max_length
    if len(prompt_ids) + len(response_ids) > total_max:
        response_ids = response_ids[:total_max - len(prompt_ids)]

    if len(response_ids) == 0:
        return np.array([])

    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    outputs = model(input_ids)
    logits = outputs.logits[0].float()  # (seq_len, vocab)

    # Extract logprobs for each response token
    # logits[t] predicts token at position t+1
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    logprobs = []

    for i in range(response_len):
        pos = prompt_len + i - 1  # logits position that predicts this token
        if pos < 0:
            continue
        token_id = response_ids[i]
        log_p = torch.log_softmax(logits[pos], dim=-1)[token_id].item()
        logprobs.append(log_p)

    return np.array(logprobs)


@torch.no_grad()
def compute_alignment_depth(
    ref_model, aligned_model, tokenizer, prompts, rejected_responses,
    device: str, max_length: int = 512, max_positions: int = 128,
):
    """
    Compute per-position safety suppression signal.

    delta(t) = logP_ref(token_t) - logP_aligned(token_t)

    Positive = aligned model suppresses this token relative to reference.
    """
    # Accumulate per-position deltas
    delta_sum = np.zeros(max_positions, dtype=np.float64)
    delta_count = np.zeros(max_positions, dtype=np.float64)
    all_deltas = []

    for idx, (prompt, rejected) in enumerate(zip(prompts, rejected_responses)):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(prompts)}...")

        ref_logprobs = compute_per_token_logprobs(
            ref_model, tokenizer, prompt, rejected, device, max_length)
        aligned_logprobs = compute_per_token_logprobs(
            aligned_model, tokenizer, prompt, rejected, device, max_length)

        n = min(len(ref_logprobs), len(aligned_logprobs), max_positions)
        if n == 0:
            continue

        delta = ref_logprobs[:n] - aligned_logprobs[:n]
        delta_sum[:n] += delta
        delta_count[:n] += 1
        all_deltas.append(delta)

    # Average
    safe_count = np.where(delta_count > 0, delta_count, 1)
    mean_delta = delta_sum / safe_count
    valid_positions = int(np.max(np.where(delta_count > 0)[0]) + 1) if delta_count.any() else 0

    return {
        "mean_delta": mean_delta[:valid_positions].tolist(),
        "n_samples_per_position": delta_count[:valid_positions].tolist(),
        "n_prompts": len(prompts),
        "n_used": len(all_deltas),
    }


def plot_results(baseline_depth: dict, curri_depth: dict, output_dir: Path):
    bl = np.array(baseline_depth["mean_delta"])
    cu = np.array(curri_depth["mean_delta"])
    max_pos = max(len(bl), len(cu))

    # Pad shorter one
    if len(bl) < max_pos:
        bl = np.pad(bl, (0, max_pos - len(bl)))
    if len(cu) < max_pos:
        cu = np.pad(cu, (0, max_pos - len(cu)))

    positions = np.arange(max_pos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Alignment Depth: Per-Token Safety Suppression Signal",
                 fontsize=13, fontweight="bold")

    # --- Left: raw per-position delta ---
    ax = axes[0]
    window = 5
    bl_smooth = np.convolve(bl, np.ones(window)/window, mode="same")
    cu_smooth = np.convolve(cu, np.ones(window)/window, mode="same")
    ax.plot(positions, bl_smooth, "b-", label="Baseline DPO", alpha=0.9, linewidth=1.5)
    ax.plot(positions, cu_smooth, "r-", label="Curri-Pacing", alpha=0.9, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Token Position in Response")
    ax.set_ylabel("Suppression Signal\n(logP_ref - logP_aligned)")
    ax.set_title("Per-Token Suppression\n(smoothed, window=5)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Middle: cumulative suppression ---
    ax = axes[1]
    bl_cum = np.cumsum(bl)
    cu_cum = np.cumsum(cu)
    ax.plot(positions, bl_cum, "b-", label="Baseline DPO", linewidth=1.8)
    ax.plot(positions, cu_cum, "r-", label="Curri-Pacing", linewidth=1.8)
    ax.set_xlabel("Token Position in Response")
    ax.set_ylabel("Cumulative Suppression")
    ax.set_title("Cumulative Safety Signal\n(higher = more total suppression)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Right: fraction of total suppression by position ---
    ax = axes[2]
    bl_total = np.sum(np.maximum(bl, 0)) or 1
    cu_total = np.sum(np.maximum(cu, 0)) or 1
    bl_frac = np.cumsum(np.maximum(bl, 0)) / bl_total * 100
    cu_frac = np.cumsum(np.maximum(cu, 0)) / cu_total * 100
    ax.plot(positions, bl_frac, "b-", label="Baseline DPO", linewidth=1.8)
    ax.plot(positions, cu_frac, "r-", label="Curri-Pacing", linewidth=1.8)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(90, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    # Find 50% and 90% depth for each
    bl_50 = np.searchsorted(bl_frac, 50)
    cu_50 = np.searchsorted(cu_frac, 50)
    bl_90 = np.searchsorted(bl_frac, 90)
    cu_90 = np.searchsorted(cu_frac, 90)
    ax.annotate(f"BL 50%: token {bl_50}", xy=(bl_50, 50), fontsize=8, color="blue",
                xytext=(bl_50 + 5, 45), arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))
    ax.annotate(f"CP 50%: token {cu_50}", xy=(cu_50, 50), fontsize=8, color="red",
                xytext=(cu_50 + 5, 55), arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_xlabel("Token Position in Response")
    ax.set_ylabel("% of Total Suppression Reached")
    ax.set_title("Alignment Depth\n(slower rise = deeper alignment)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 105)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "alignment_depth.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    # Summary stats
    print(f"\n=== Alignment Depth Summary ===")
    print(f"Baseline  — 50% suppression by token {bl_50}, 90% by token {bl_90}, total={bl_total:.1f}")
    print(f"Curri     — 50% suppression by token {cu_50}, 90% by token {cu_90}, total={cu_total:.1f}")
    if cu_50 > bl_50:
        print(f"Curri-Pacing alignment is DEEPER: 50% point shifted by +{cu_50 - bl_50} tokens")
    elif cu_50 < bl_50:
        print(f"Baseline alignment is deeper: 50% point shifted by +{bl_50 - cu_50} tokens")
    else:
        print(f"50% depth is equal at token {bl_50}")


def main():
    parser = argparse.ArgumentParser(description="Alignment Depth Analysis")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--baseline-adapter", required=True)
    parser.add_argument("--curri-adapter", required=True)
    parser.add_argument("--curri-base", default=None,
                        help="Base model for curri adapter (if different from --base-model)")
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-positions", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output)

    # Load test data — use rejected (unsafe) responses
    rows = []
    with open(args.test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    samples = rows[:args.n_samples]
    prompts = [r["prompt"] for r in samples]
    rejected = [r["rejected"] for r in samples]
    print(f"Loaded {len(prompts)} test pairs from {args.test_path}")

    results = {}

    # --- Reference model (no adapter) ---
    print("\n" + "=" * 50)
    print("Loading reference model (no adapter)...")
    ref_model, tokenizer = load_model(args.base_model, None, device)

    for label, adapter_path, base_path in [
        ("baseline", args.baseline_adapter, args.base_model),
        ("curri_pacing", args.curri_adapter, args.curri_base or args.base_model),
    ]:
        print("\n" + "=" * 50)
        print(f"Loading {label}: base={base_path}, adapter={adapter_path}")
        aligned_model, _ = load_model(base_path, adapter_path, device)

        print(f"Computing per-token suppression on {len(prompts)} rejected responses...")
        depth = compute_alignment_depth(
            ref_model, aligned_model, tokenizer, prompts, rejected,
            device, args.max_length, args.max_positions,
        )
        results[label] = depth
        print(f"  Used {depth['n_used']}/{depth['n_prompts']} prompts, "
              f"max position = {len(depth['mean_delta'])}")

        # Free GPU memory
        del aligned_model
        torch.cuda.empty_cache()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "alignment_depth_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    # Plot
    plot_results(results["baseline"], results["curri_pacing"], output_dir)

    # Free ref model
    del ref_model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
