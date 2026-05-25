"""
Prefill attack evaluation — Phase 1: Inference.

For each record in the DOOR dataset, takes the first k tokens of the harmful ('rejected')
response, decodes them as text, appends that prefix to the chat-templated user prompt, and
generates a continuation. This simulates an attack where the model is forced to begin its
response with a harmful prefix and we measure whether it continues harmfully.

Dataset format (door_full_parsed.jsonl):
  {"prompt": "harmful intent...", "chosen": "safe refusal...", "rejected": "harmful response..."}

Output JSONL (one line per record):
  {"intent": "...", "harmful_prefix": "...", "response_generated": "prefix + continuation...",
   "k": 20, "meta": {...}}

Usage:
  python -m src.eval.evaluate_prefill \\
    --base-model mlabonne/Qwen3-8B-abliterated \\
    [--adapter-path outputs/exp_009/qwen3/dpo_curriculum] \\
    --eval-data data/raw/door_full_parsed.jsonl \\
    --k 20 \\
    --output results/exp_009/qwen3/prefill/k_20/prefill_curriculum.jsonl \\
    [--batch-size 8] [--max-new-tokens 256]
"""

from __future__ import annotations

import argparse
import json
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


def build_prefill_prompts(
    records: list[dict],
    tokenizer,
    k: int,
) -> tuple[list[str], list[str]]:
    """
    For each record, derive the harmful_prefix (first k tokens of 'rejected' decoded back to text)
    and build the full input text (chat-templated intent + harmful_prefix).

    Returns (full_input_texts, harmful_prefixes).
    """
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    full_texts: list[str] = []
    prefixes: list[str] = []

    for rec in records:
        intent = rec["prompt"]
        harmful_response = rec["rejected"]

        # Tokenize the harmful response and take first k tokens
        resp_tokens = tokenizer.encode(harmful_response, add_special_tokens=False)
        prefix_tokens = resp_tokens[:k]
        harmful_prefix = tokenizer.decode(prefix_tokens, skip_special_tokens=True)

        if has_chat_template:
            base = tokenizer.apply_chat_template(
                [{"role": "user", "content": intent}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            base = intent + "\n\n"

        full_texts.append(base + harmful_prefix)
        prefixes.append(harmful_prefix)

    return full_texts, prefixes


def generate_continuations(
    model,
    tokenizer,
    full_texts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    continuations: list[str] = []
    for i in tqdm(range(0, len(full_texts), batch_size), desc="Generating"):
        batch = full_texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(model.device)
        inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
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
            continuations.append(decoded.strip())
    return continuations


def run(
    base_model: str,
    adapter_path: Path | None,
    eval_data_path: Path,
    k: int,
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = output_path.read_text().strip().splitlines()
        print(f"Output already exists with {len(existing)} lines, skipping: {output_path}")
        return

    # Load dataset
    records = [json.loads(line) for line in eval_data_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} records from {eval_data_path}")

    # Load model
    label = base_model + (f" + {adapter_path}" if adapter_path else " (raw)")
    print(f"Model: {label}")
    model, tokenizer = load_model(base_model, adapter_path)

    # Build prefill prompts
    full_texts, harmful_prefixes = build_prefill_prompts(records, tokenizer, k)
    print(f"Built prefill prompts (k={k} tokens). Example prefix: {repr(harmful_prefixes[0][:80])}")

    # Generate continuations
    continuations = generate_continuations(model, tokenizer, full_texts, batch_size, max_new_tokens)

    # Save JSONL
    with output_path.open("w") as f:
        for rec, prefix, cont in zip(records, harmful_prefixes, continuations):
            response_generated = prefix + cont
            record = {
                "intent": rec["prompt"],
                "harmful_prefix": prefix,
                "response_generated": response_generated,
                "k": k,
                "meta": {
                    "base_model": base_model,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {output_path}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Prefill attack inference: generate continuations from harmful prefixes.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--eval-data", type=Path, default=Path("data/raw/door_full_parsed.jsonl"),
                   help="JSONL with prompt/rejected fields")
    p.add_argument("--k", type=int, default=20, help="Number of tokens to use as harmful prefix")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    run(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        eval_data_path=args.eval_data,
        k=args.k,
        output_path=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    cli()
