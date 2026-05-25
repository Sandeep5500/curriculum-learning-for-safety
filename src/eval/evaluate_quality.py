"""
Phase 3b: General quality evaluation on MMLU and HellaSwag benchmarks.

Evaluates model capability preservation after DPO safety tuning.
Tests on MMLU (57k multiple choice general knowledge) and HellaSwag (10k commonsense).

Usage:
  python -m src.eval.evaluate_quality --adapter-path outputs/exp_001/dpo_curriculum
  python -m src.eval.evaluate_quality --adapter-path outputs/exp_001/dpo_baseline --benchmarks mmlu
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_mmlu(split: str = "test") -> list[dict]:
    """Load MMLU from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
    processed = []
    for item in dataset:
        processed.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": chr(65 + item["answer"]),  # Convert 0-3 to A-D
            "subject": item.get("subject", "unknown"),
        })
    return processed


def load_hellaswag(split: str = "validation") -> list[dict]:
    """Load HellaSwag from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset("Rowan/hellaswag", split=split)
    processed = []
    for item in dataset:
        # HellaSwag endings are pre-shuffled, correct answer is in label
        processed.append({
            "context": item["ctx"],
            "endings": item["endings"],
            "answer": chr(65 + int(item["label"])),  # Convert 0-3 to A-D
            "activity": item.get("activity_label", "unknown"),
        })
    return processed


def format_mmlu_prompt(item: dict, num_shots: int = 5) -> str:
    """Format MMLU multiple choice question."""
    question = item["question"]
    choices = item["choices"]

    # Format: Q: {question}
    # A) {choice_a}
    # B) {choice_b}
    # C) {choice_c}
    # D) {choice_d}
    # Answer:

    prompt = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}) {choice}\n"
    prompt += "Answer:"
    return prompt


def format_hellaswag_prompt(item: dict) -> str:
    """Format HellaSwag commonsense completion."""
    context = item["context"]
    endings = item["endings"]

    # Format: {context}
    # (A) {ending_a}
    # (B) {ending_b}
    # (C) {ending_c}
    # (D) {ending_d}
    # Answer:

    prompt = f"{context}\n"
    for i, ending in enumerate(endings):
        prompt += f"({chr(65 + i)}) {ending}\n"
    prompt += "Answer:"
    return prompt


def extract_answer_token(text: str) -> str | None:
    """Extract A, B, C, or D from model output."""
    import re
    import unicodedata
    # Normalize Unicode (e.g. fullwidth letters -> ASCII)
    text = unicodedata.normalize("NFKC", text)
    # Strip <think>...</think> blocks before extracting
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip common special tokens that leak from chat templates
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    # Normalize "Xassistant" artifact (Llama3 chat template leakage: "C" + "assistant")
    text = re.sub(r"^(\s*[ABCD])assistant\b", r"\1 ", text, flags=re.IGNORECASE)
    # 1. Generation starts with a letter (most reliable)
    m = re.match(r"^\s*([ABCD])[^a-zA-Z]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Handle single character or letter at very start
    m = re.match(r"^\s*([ABCD])\s*$", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2. Explicit answer patterns: "answer is X", "answer: X", "correct answer is X"
    m = re.search(r"(?:correct\s+)?answer(?:\s+is)?[:\s]+\**\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3. "option X" patterns
    m = re.search(r"option\s+\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4. Standalone letter on its own line
    m = re.search(r"(?:^|\n)\s*([ABCD])\s*(?:\)|\.|\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 5. Numeric answer (0-3 mapped to A-D)
    m = re.match(r"^\s*([0-3])\s*$", text)
    if m:
        return chr(65 + int(m.group(1)))
    # 6. Last resort: first standalone A/B/C/D (case-insensitive, but skip
    #    lowercase 'a' to avoid matching articles)
    m = re.search(r"\b([BCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(A)\b", text)  # uppercase only for 'A' to avoid articles
    if m:
        return m.group(1).upper()
    return None


def _build_prompt_token_ids(tokenizer, prompts: list[str]) -> list[list[int]] | None:
    """Apply chat template and return token ID lists, or None if no template."""
    if tokenizer is None:
        return None
    if not (hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None):
        return None
    print("  Applying chat template to prompts (tokenized to avoid double-BOS).")
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]


def _run_vllm(llm, prompt_token_ids, prompts, sampling_params, lora_request):
    if prompt_token_ids is not None:
        return llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
    return llm.generate(prompts, sampling_params, lora_request=lora_request)


def extract_cot_answer(text: str) -> str | None:
    """Extract answer letter from a chain-of-thought generation.

    Looks at the last non-empty line first (model should conclude with the
    letter), then falls back to the standard extractor on the full text.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m = extract_answer_token(last)
        if m:
            return m
    return extract_answer_token(text)


def evaluate_benchmark(
    llm,
    items: list[dict],
    format_fn,
    benchmark_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    lora_request=None,
    tokenizer=None,
    use_logprobs: bool = False,
    use_cot: bool = False,
) -> dict[str, Any]:
    """Evaluate model on a benchmark.

    use_logprobs=True: score by argmax over P(A/B/C/D) at the first token
    position — immune to generation position bias.  Faster (max_tokens=1).

    use_cot=True: ask the model to reason step-by-step before answering;
    extract the final letter from the last line of the generation.

    Default: generate text and extract the answer letter with regex.
    """
    from vllm import SamplingParams

    if use_logprobs:
        scoring_mode = "logprobs"
    elif use_cot:
        scoring_mode = "cot"
    else:
        scoring_mode = "generation"
    print(f"\nEvaluating {benchmark_name} ({len(items)} items) [{scoring_mode}]...")

    if use_logprobs:
        # Prompts end with "Answer:" — no "respond with only" instruction needed.
        prompts = [format_fn(item) for item in items]
        prompt_token_ids = _build_prompt_token_ids(tokenizer, prompts)
        # Generate exactly 1 token; request top-50 logprobs so A/B/C/D are always present.
        sampling_params = SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)
        outputs = _run_vllm(llm, prompt_token_ids, prompts, sampling_params, lora_request)

        correct = 0
        detailed_results = []
        for item, output, prompt in zip(items, outputs, prompts):
            expected = item["answer"]
            token_logprobs = output.outputs[0].logprobs[0]  # dict[token_id -> Logprob]

            # Collect the highest logprob seen for each of A B C D.
            # decoded_token may be " A", "A", "▁A", etc. — strip and upper.
            letter_scores: dict[str, float] = {}
            for logprob_obj in token_logprobs.values():
                decoded = logprob_obj.decoded_token.strip().upper()
                if decoded in ("A", "B", "C", "D"):
                    if decoded not in letter_scores or logprob_obj.logprob > letter_scores[decoded]:
                        letter_scores[decoded] = logprob_obj.logprob

            predicted = max(letter_scores, key=letter_scores.get) if letter_scores else None
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            detailed_results.append({
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "prompt": prompt[:200],
                "letter_logprobs": {k: round(v, 4) for k, v in sorted(letter_scores.items())},
                "expected_answer": expected,
                "predicted_answer": predicted,
                "correct": is_correct,
            })

    elif use_cot:
        cot_suffix = (
            "\nThink through the options step by step, then give your final answer "
            "as a single letter (A, B, C, or D) on the last line."
        )
        prompts = [format_fn(item) + cot_suffix for item in items]
        prompt_token_ids = _build_prompt_token_ids(tokenizer, prompts)
        sampling_params = SamplingParams(max_tokens=256, temperature=temperature)
        outputs = _run_vllm(llm, prompt_token_ids, prompts, sampling_params, lora_request)
        generations = [o.outputs[0].text.strip() for o in outputs]

        correct = 0
        detailed_results = []
        for item, generation, prompt in zip(items, generations, prompts):
            predicted = extract_cot_answer(generation)
            expected = item["answer"]
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            detailed_results.append({
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "prompt": prompt[:200],
                "generation": generation[:400],
                "expected_answer": expected,
                "predicted_answer": predicted,
                "correct": is_correct,
            })

    else:
        prompts = [
            format_fn(item) + "\nRespond with only the letter of the correct answer: A, B, C, or D. Do not explain."
            for item in items
        ]
        prompt_token_ids = _build_prompt_token_ids(tokenizer, prompts)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = _run_vllm(llm, prompt_token_ids, prompts, sampling_params, lora_request)
        generations = [o.outputs[0].text.strip() for o in outputs]

        correct = 0
        detailed_results = []
        for item, generation, prompt in zip(items, generations, prompts):
            predicted = extract_answer_token(generation)
            expected = item["answer"]
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            detailed_results.append({
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "prompt": prompt[:200],
                "generation": generation[:200],
                "expected_answer": expected,
                "predicted_answer": predicted,
                "correct": is_correct,
            })

    accuracy = correct / len(items) if items else 0.0
    result = {
        "benchmark": benchmark_name,
        "scoring": scoring_mode,
        "n_total": len(items),
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "detailed_results": detailed_results,
    }
    print(f"  {benchmark_name}: {correct}/{len(items)} correct ({100*accuracy:.1f}%)")
    return result


def run(
    base_model: str,
    adapter_path: str | None,
    benchmarks: list[str],
    output_path: Path,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    gen_batch_size: int = 16,
    use_logprobs: bool = False,
    use_cot: bool = False,
) -> None:
    """Evaluate model on quality benchmarks."""
    from vllm import LLM

    print(f"Loading base model: {base_model}")
    if adapter_path:
        print(f"  with LoRA adapter: {adapter_path}")
        lora_path = Path(adapter_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Initialize vLLM with LoRA support if needed
    llm = LLM(
        model=base_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enable_lora=bool(adapter_path),
        max_lora_rank=64,
    )

    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, str(Path(adapter_path).resolve()))

    # Load benchmarks
    results = {"timestamp": datetime.now().isoformat(), "model": base_model, "adapter_path": adapter_path or "none"}

    if "mmlu" in benchmarks:
        print("\nLoading MMLU (57k questions)...")
        mmlu_data = load_mmlu(split="test")
        mmlu_result = evaluate_benchmark(
            llm,
            mmlu_data,
            format_mmlu_prompt,
            "MMLU",
            max_tokens=max_tokens,
            temperature=temperature,
            lora_request=lora_request,
            tokenizer=tokenizer,
            use_logprobs=use_logprobs,
            use_cot=use_cot,
        )
        results["mmlu"] = mmlu_result

    if "hellaswag" in benchmarks:
        print("\nLoading HellaSwag (10k examples)...")
        hellaswag_data = load_hellaswag(split="validation")
        hellaswag_result = evaluate_benchmark(
            llm,
            hellaswag_data,
            format_hellaswag_prompt,
            "HellaSwag",
            max_tokens=max_tokens,
            temperature=temperature,
            lora_request=lora_request,
            tokenizer=tokenizer,
            use_logprobs=use_logprobs,
            use_cot=use_cot,
        )
        results["hellaswag"] = hellaswag_result

    # Compute average accuracy
    accuracies = []
    if "mmlu" in results and "mmlu" in benchmarks:
        accuracies.append(results["mmlu"]["accuracy"])
    if "hellaswag" in results and "hellaswag" in benchmarks:
        accuracies.append(results["hellaswag"]["accuracy"])

    if accuracies:
        results["average_accuracy"] = round(sum(accuracies) / len(accuracies), 4)

    # Save results
    save_json(output_path, results)
    print(f"\nSaved results to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    if "mmlu" in results:
        print(f"MMLU: {results['mmlu']['n_correct']}/{results['mmlu']['n_total']} ({100*results['mmlu']['accuracy']:.1f}%)")
    if "hellaswag" in results:
        print(f"HellaSwag: {results['hellaswag']['n_correct']}/{results['hellaswag']['n_total']} ({100*results['hellaswag']['accuracy']:.1f}%)")
    if "average_accuracy" in results:
        print(f"Average: {100*results['average_accuracy']:.1f}%")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Phase 3b: Evaluate model quality on MMLU and HellaSwag.")
    p.add_argument("--base-model", default="informatiker/Llama-3-8B-Instruct-abliterated", help="Base model ID")
    p.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter (optional)")
    p.add_argument("--benchmarks", nargs="+", default=["mmlu", "hellaswag"], help="Benchmarks to run: mmlu, hellaswag, or both")
    p.add_argument("--output", type=Path, default=root / "results" / "quality_metrics.json", help="Output JSON path")
    p.add_argument("--max-tokens", type=int, default=32, help="Max generation tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    p.add_argument("--gen-batch-size", type=int, default=16, help="Batch size for vLLM")
    p.add_argument("--logprobs", action="store_true", default=False,
                   help="Score by argmax over P(A/B/C/D) at first token position instead of generating text. "
                        "Immune to position bias; recommended for most models.")
    p.add_argument("--cot", action="store_true", default=False,
                   help="Chain-of-thought prompting: ask model to reason step-by-step before answering. "
                        "Extracts letter from the last line of the generation.")
    args = p.parse_args()

    # Normalize benchmark names
    benchmarks = [b.lower() for b in args.benchmarks]
    if not all(b in ["mmlu", "hellaswag"] for b in benchmarks):
        raise ValueError(f"Unknown benchmarks: {benchmarks}. Use 'mmlu' and/or 'hellaswag'")

    run(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        benchmarks=benchmarks,
        output_path=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gen_batch_size=args.gen_batch_size,
        use_logprobs=args.logprobs,
        use_cot=args.cot,
    )


if __name__ == "__main__":
    cli()
