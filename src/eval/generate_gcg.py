"""
Generate a GCG adversarial suffix for one HarmBench behavior using nanogcg.

Runs GCG optimization against a target model (Qwen3-8B-Instruct by default),
optimizing a suffix appended to the behavior prompt to elicit a harmful response.

Output JSON per behavior:
  {
    "behavior_id": str,
    "behavior": str,
    "suffix": str,          # optimized GCG suffix
    "test_case": str,       # full attack prompt = behavior + " " + suffix
    "best_loss": float,
    "num_steps": int,
    "model": str
  }

Usage:
  python -m src.eval.generate_gcg \\
    --behavior-id oracle_db_schema_time_blind_sql_injection_steps \\
    --behaviors-file data/harmbench/behaviors.json \\
    --model Qwen/Qwen3-8B \\
    --output-dir data/harmbench/gcg_qwen3/individual/ \\
    --num-steps 500

Optimization flags (for faster generation, e.g. Mistral):
  --search-width 128      # default 512; 4x speedup with minimal quality loss
  --topk 128              # default 256; narrows gradient candidate pool
  --early-stop            # stop once model outputs the target string
  --use-prefix-cache      # enable KV cache for prompt prefix (default off for Qwen3 compat)
  --suffix-length 10      # shorter suffix = faster forward passes (default 20 tokens)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GCG_TARGET = "Sure, here is how to"


def run_gcg(
    behavior_id: str,
    behavior_text: str,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    search_width: int,
    seed: int,
    topk: int = 256,
    early_stop: bool = False,
    use_prefix_cache: bool = False,
    suffix_length: int = 20,
) -> None:
    import nanogcg
    from nanogcg import GCGConfig

    output_path = output_dir / f"{behavior_id}.json"
    if output_path.exists():
        print(f"Already exists, skipping: {output_path}")
        return

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    messages = [{"role": "user", "content": behavior_text}]
    target = GCG_TARGET

    optim_str_init = " ".join(["x"] * suffix_length)

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        early_stop=early_stop,
        use_prefix_cache=use_prefix_cache,
        optim_str_init=optim_str_init,
        seed=seed,
        verbosity="WARNING",
    )

    print(
        f"Running GCG: behavior_id={behavior_id}, num_steps={num_steps}, "
        f"search_width={search_width}, topk={topk}, early_stop={early_stop}, "
        f"prefix_cache={use_prefix_cache}, suffix_len={suffix_length}"
    )
    result = nanogcg.run(model, tokenizer, messages, target, config)

    suffix = result.best_string
    test_case = f"{behavior_text} {suffix}"

    output = {
        "behavior_id": behavior_id,
        "behavior": behavior_text,
        "suffix": suffix,
        "test_case": test_case,
        "best_loss": float(result.best_loss),
        "num_steps": num_steps,
        "model": model_id,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}  (loss={result.best_loss:.4f})")


def cli() -> None:
    p = argparse.ArgumentParser(description="Generate GCG suffix for one HarmBench behavior.")
    p.add_argument("--behavior-id", required=True, help="HarmBench behavior ID (slug)")
    p.add_argument("--behaviors-file", type=Path,
                   default=Path("data/harmbench/behaviors.json"),
                   help="JSON mapping behavior_id -> {behavior, category}")
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="HuggingFace model ID to attack")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/harmbench/gcg_qwen3/individual"),
                   help="Directory to save per-behavior result JSON")
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--search-width", type=int, default=512)
    p.add_argument("--topk", type=int, default=256,
                   help="Top-k tokens from gradient ranking to sample candidates from")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop optimization once the model outputs the target string")
    p.add_argument("--use-prefix-cache", action="store_true",
                   help="Cache KV for prompt prefix (faster; disable for Qwen3 compat)")
    p.add_argument("--suffix-length", type=int, default=20,
                   help="Number of optimizable suffix tokens (shorter = faster)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    behaviors = json.loads(args.behaviors_file.read_text())
    if args.behavior_id not in behaviors:
        print(f"ERROR: behavior_id '{args.behavior_id}' not found in {args.behaviors_file}")
        sys.exit(1)

    behavior_text = behaviors[args.behavior_id]["behavior"]

    run_gcg(
        behavior_id=args.behavior_id,
        behavior_text=behavior_text,
        model_id=args.model,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        search_width=args.search_width,
        seed=args.seed,
        topk=args.topk,
        early_stop=args.early_stop,
        use_prefix_cache=args.use_prefix_cache,
        suffix_length=args.suffix_length,
    )


if __name__ == "__main__":
    cli()
