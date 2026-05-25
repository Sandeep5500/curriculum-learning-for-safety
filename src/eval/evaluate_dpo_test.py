"""
Phase 3b: Evaluate DPO reward accuracy on held-out test split.

For each (prompt, chosen, rejected) pair in the test set, computes the implicit
DPO reward: log p(chosen|prompt) - log p(rejected|prompt).

Metrics:
  reward_accuracy : fraction of pairs where the model prefers chosen over rejected
  reward_margin   : mean log-prob margin (chosen - rejected), higher = more confident

This is the in-distribution complement to AdvBench (which is out-of-distribution).

Usage:
  python -m src.eval.evaluate_dpo_test \
    --test-path data/processed/exp_004/pku/llama3/curriculum_dataset_test.jsonl \
    --base-model informatiker/Llama-3-8B-Instruct-abliterated \
    --adapter-path outputs/exp_004/llama3/dpo_curriculum \
    --output results/exp_004/llama3/dpo_test_curriculum.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def sequence_log_prob(model, tokenizer, prompt: str, response: str, device: str) -> tuple[float, int]:
    """Compute (sum_log_p, n_tokens) for log p(response tokens | prompt)."""
    import torch

    # Tokenize full sequence and prompt-only to find response token boundary
    full_ids = tokenizer(prompt + response, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, seq_len, vocab)

    # Log-probs of each token given the previous context
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

    # Response tokens are at positions prompt_len..end (shifted by 1 for next-token prediction)
    response_token_ids = full_ids[0, prompt_len:]           # tokens to predict
    response_log_probs = log_probs[prompt_len - 1 : -1]     # context positions

    n_tokens = int(response_token_ids.shape[0])
    if n_tokens == 0:
        return 0.0, 0

    token_log_probs = response_log_probs.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum().item(), n_tokens


def run(
    test_path: Path,
    base_model: str,
    adapter_path: str | None,
    output_path: Path,
    batch_size: int = 8,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    print(f"\nLoading test set: {test_path}")
    rows = load_jsonl(test_path)
    print(f"  {len(rows)} test pairs")

    margins = []
    norm_margins = []
    correct = 0
    correct_normalized = 0
    samples = []

    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            acc_so_far = correct / (i + 1)
            print(f"  [{i+1}/{len(rows)}]  running accuracy={acc_so_far:.3f}")

        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        lp_chosen, n_chosen = sequence_log_prob(model, tokenizer, prompt, chosen, device)
        lp_rejected, n_rejected = sequence_log_prob(model, tokenizer, prompt, rejected, device)

        margin = lp_chosen - lp_rejected
        margins.append(margin)
        is_correct = margin > 0
        if is_correct:
            correct += 1

        # Length-normalized margin: avg per-token log-prob difference
        avg_lp_chosen = lp_chosen / n_chosen if n_chosen > 0 else 0.0
        avg_lp_rejected = lp_rejected / n_rejected if n_rejected > 0 else 0.0
        norm_margin = avg_lp_chosen - avg_lp_rejected
        norm_margins.append(norm_margin)
        is_correct_normalized = norm_margin > 0
        if is_correct_normalized:
            correct_normalized += 1

        sample: dict = {
            "margin": round(margin, 4),
            "correct": is_correct,
            "lp_chosen": round(lp_chosen, 4),
            "lp_rejected": round(lp_rejected, 4),
            "n_chosen_tokens": n_chosen,
            "n_rejected_tokens": n_rejected,
            "norm_margin": round(norm_margin, 6),
            "correct_normalized": is_correct_normalized,
        }
        # Include difficulty margin if available (from scored dataset)
        if "similarity_score" in row and "similarity_rejected" in row:
            sample["difficulty_margin"] = round(
                row["similarity_score"] - row["similarity_rejected"], 4
            )
        samples.append(sample)

    reward_accuracy = correct / len(rows)
    reward_margin = sum(margins) / len(margins)
    reward_accuracy_normalized = correct_normalized / len(rows)
    reward_margin_normalized = sum(norm_margins) / len(norm_margins)

    # Per-quartile accuracy breakdown (requires difficulty_margin in samples)
    quartile_accs: dict = {}
    if all("difficulty_margin" in s for s in samples):
        import numpy as np
        diff_margins = [s["difficulty_margin"] for s in samples]
        q25, q50, q75 = float(np.percentile(diff_margins, 25)), float(np.percentile(diff_margins, 50)), float(np.percentile(diff_margins, 75))
        buckets: dict[str, list[bool]] = {"Q1_easy": [], "Q2": [], "Q3": [], "Q4_hard": []}
        for s in samples:
            dm = s["difficulty_margin"]
            if dm <= q25:
                buckets["Q1_easy"].append(s["correct"])
            elif dm <= q50:
                buckets["Q2"].append(s["correct"])
            elif dm <= q75:
                buckets["Q3"].append(s["correct"])
            else:
                buckets["Q4_hard"].append(s["correct"])
        quartile_accs = {k: round(sum(v) / len(v), 4) if v else None for k, v in buckets.items()}
        print(f"  Per-quartile accuracy: {quartile_accs}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "adapter_path": adapter_path or "none",
        "test_path": str(test_path),
        "n_pairs": len(rows),
        "reward_accuracy": round(reward_accuracy, 4),
        "reward_margin": round(reward_margin, 4),
        "reward_accuracy_normalized": round(reward_accuracy_normalized, 4),
        "reward_margin_normalized": round(reward_margin_normalized, 6),
        "n_correct": correct,
        "n_correct_normalized": correct_normalized,
        "quartile_accuracy": quartile_accs,
        "samples": samples,
    }

    save_json(output_path, results)
    print(f"\n=== Results ===")
    print(f"Reward accuracy            : {100*reward_accuracy:.1f}%  ({correct}/{len(rows)})  [raw, length-biased]")
    print(f"Reward accuracy (norm.)    : {100*reward_accuracy_normalized:.1f}%  ({correct_normalized}/{len(rows)})  [length-normalized]")
    print(f"Reward margin              : {reward_margin:.4f}")
    print(f"Reward margin (per-token)  : {reward_margin_normalized:.6f}")
    print(f"Saved to {output_path}")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Evaluate DPO reward accuracy on held-out test split.")
    p.add_argument("--test-path", type=Path, required=True, help="JSONL test file (prompt/chosen/rejected)")
    p.add_argument("--base-model", type=str, default="informatiker/Llama-3-8B-Instruct-abliterated")
    p.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter path (omit for raw model)")
    p.add_argument("--output", type=Path, default=root / "results" / "dpo_test_eval.json")
    args = p.parse_args()

    run(
        test_path=args.test_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    cli()
