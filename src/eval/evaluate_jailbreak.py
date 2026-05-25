"""
Jailbreak robustness evaluation — Phase 1: Inference.

Loads pre-computed GCG or AutoDAN attack prompts (from wicai24/Jailbreak-Datasets),
runs each through a model (base + optional LoRA adapter), and saves responses to JSONL.

Test cases JSON format:
  { "behavior_id": ["attack_prompt_string"], ... }

Output JSONL (one JSON line per behavior):
  {"behavior_id": "...", "attack_type": "gcg", "test_case": "...", "response": "..."}

Usage:
  python -m src.eval.evaluate_jailbreak \\
    --base-model informatiker/Llama-3-8B-Instruct-abliterated \\
    [--adapter-path outputs/exp_007/llama3/dpo_baseline] \\
    --test-cases data/harmbench/gcg_test_cases.json \\
    --attack-type gcg \\
    --output results/exp_007/llama3/jailbreak/gcg_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(base_model_id: str, adapter_path: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    if has_chat_template:
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
    else:
        formatted = prompts

    responses: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(model.device)
        # Some tokenizers (e.g. Mistral) return token_type_ids which certain
        # model architectures reject in generate(). Strip it if present.
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for seq in out:
            decoded = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            # Strip think tags (Qwen3 and other CoT models)
            decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.DOTALL)
            decoded = re.sub(r"<think>.*$", "", decoded, flags=re.DOTALL)
            responses.append(decoded.strip())
    return responses


def run(
    base_model: str,
    adapter_path: Path | None,
    test_cases_path: Path,
    attack_type: str,
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if output_path.exists():
        existing = output_path.read_text().strip().splitlines()
        print(f"Output already exists with {len(existing)} lines, skipping: {output_path}")
        return

    # Load test cases
    test_cases: dict[str, list[str]] = json.loads(test_cases_path.read_text())
    behavior_ids = list(test_cases.keys())
    attack_prompts = [test_cases[bid][0] for bid in behavior_ids]
    print(f"Loaded {len(behavior_ids)} {attack_type.upper()} attack prompts")

    # Load model
    label = base_model + (f" + {adapter_path}" if adapter_path else " (raw)")
    print(f"Model: {label}")
    model, tokenizer = load_model(base_model, adapter_path)

    # Generate
    responses = generate_responses(model, tokenizer, attack_prompts, batch_size, max_new_tokens)

    # Save JSONL
    with output_path.open("w") as f:
        for bid, test_case, response in zip(behavior_ids, attack_prompts, responses):
            record = {
                "behavior_id": bid,
                "attack_type": attack_type,
                "test_case": test_case,
                "response": response,
                "meta": {
                    "base_model": base_model,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(responses)} responses to {output_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Jailbreak inference: generate responses to attack prompts.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--test-cases", type=Path, required=True, help="JSON file: {behavior_id: [attack_prompt]}")
    p.add_argument("--attack-type", required=True, choices=["gcg", "autodan"], help="Attack method label")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    run(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        test_cases_path=args.test_cases,
        attack_type=args.attack_type,
        output_path=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    cli()
