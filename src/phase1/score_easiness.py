"""
Phase 1: Easiness scoring pipeline.

Easiness = how semantically similar the model's zero-shot output is to the
dataset's safe (chosen) response. We generate with the target model via
HuggingFace transformers, then score (generated, chosen) with either sentence-transformer
cosine similarity or BERTScore F1.

Supports stratified train/test split (default 80/20) with seed=42 for reproducibility.
Split is stratified by difficulty (margin quartiles) to guarantee equal representation of
easy and hard examples in both train and test. Train split is used for curriculum sorting;
test split is held constant across all experiments.

Usage:
  python -m src.phase1.score_easiness --scorer sentence-transformer
  python -m src.phase1.score_easiness --scorer bertscore --input data/raw/debug_subset.jsonl
  python -m src.phase1.score_easiness --split-train-ratio 0.8 --split-seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol


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


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split_train_test(
    rows: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
    n_strata: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/test split ensuring equal difficulty representation.

    Divides examples into n_strata equal-size buckets by margin score
    (similarity_score - similarity_rejected), then samples independently
    within each bucket. Guarantees ~train_ratio examples per difficulty quartile.

    Rows must already have 'similarity_score' and 'similarity_rejected' fields.
    """
    import random

    rng = random.Random(seed)
    margins = [r["similarity_score"] - r["similarity_rejected"] for r in rows]
    sorted_indices = sorted(range(len(rows)), key=lambda i: margins[i])

    n = len(sorted_indices)
    stratum_size = n // n_strata
    train_indices: list[int] = []
    test_indices: list[int] = []

    for s in range(n_strata):
        start = s * stratum_size
        end = (s + 1) * stratum_size if s < n_strata - 1 else n
        bucket = sorted_indices[start:end]
        rng.shuffle(bucket)
        n_train = round(len(bucket) * train_ratio)
        train_indices.extend(bucket[:n_train])
        test_indices.extend(bucket[n_train:])

    return [rows[i] for i in train_indices], [rows[i] for i in test_indices]


def generate_responses_vllm(prompts: list[str], model_id: str, max_tokens: int, temperature: float) -> list[str]:
    """Fast batch-generate with vLLM. Raises ImportError if model arch unsupported."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, trust_remote_code=True, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def generate_responses_hf(prompts: list[str], model_id: str, max_tokens: int, temperature: float, batch_size: int = 4) -> list[str]:
    """Fallback batch-generate using HuggingFace transformers."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from tqdm import tqdm

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, TypeError):
        # Some models have tokenizer issues (e.g. Mistral v0.3 uses tokenizer.model.v3,
        # Ministral uses TokenizersBackend). PreTrainedTokenizerFast loads tokenizer.json directly.
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = config.architectures[0] if config.architectures else None
    model_cls = getattr(transformers, arch) if arch and hasattr(transformers, arch) else transformers.AutoModelForCausalLM
    print(f"Using model class: {model_cls.__name__}")

    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    if has_chat_template:
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
    else:
        formatted = prompts

    responses = []
    for i in tqdm(range(0, len(formatted), batch_size), desc="Generating"):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, pad_token_id=tokenizer.eos_token_id,
            )
        for j in range(len(batch)):
            prompt_len = inputs["input_ids"].shape[1]
            responses.append(tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True).strip())
    return responses


def generate_responses(prompts: list[str], model_id: str, max_tokens: int, temperature: float, batch_size: int = 4) -> list[str]:
    """Generate responses — tries vLLM first (fast), falls back to HF transformers."""
    try:
        print(f"Attempting vLLM generation (model: {model_id})...")
        return generate_responses_vllm(prompts, model_id, max_tokens, temperature)
    except Exception as e:
        print(f"vLLM failed ({e}), falling back to HF transformers (batch_size={batch_size})...")
        return generate_responses_hf(prompts, model_id, max_tokens, temperature, batch_size=batch_size)


class SimilarityScorer(Protocol):
    def score(self, generated: list[str], chosen: list[str]) -> list[float]:
        ...


class SentenceTransformerScorer:
    """Cosine similarity between embeddings from sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id)

    def score(self, generated: list[str], chosen: list[str]) -> list[float]:
        import numpy as np
        gen_emb = self.model.encode(generated, show_progress_bar=True)
        ref_emb = self.model.encode(chosen, show_progress_bar=True)
        # Cosine similarity per pair
        gen_norm = gen_emb / (np.linalg.norm(gen_emb, axis=1, keepdims=True) + 1e-9)
        ref_norm = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-9)
        sim = (gen_norm * ref_norm).sum(axis=1)
        return sim.tolist()


class BERTScoreScorer:
    """BERTScore F1 via Hugging Face evaluate."""

    def __init__(self):
        import evaluate
        self.metric = evaluate.load("bertscore")

    def score(self, generated: list[str], chosen: list[str]) -> list[float]:
        result = self.metric.compute(
            predictions=generated,
            references=chosen,
            lang="en",
        )
        return result["f1"]


def get_scorer(name: str) -> SimilarityScorer:
    if name == "sentence-transformer":
        return SentenceTransformerScorer()
    if name == "bertscore":
        return BERTScoreScorer()
    raise ValueError(f"Unknown scorer: {name}. Use 'sentence-transformer' or 'bertscore'.")


def run(
    input_path: Path,
    output_path: Path,
    scorer_name: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    max_samples: int | None = None,
    gen_cache: Path | None = None,
    split_train_ratio: float = 0.8,
    split_seed: int = 42,
    batch_size: int = 4,
) -> None:
    rows = load_jsonl(input_path)
    if not rows:
        raise SystemExit(f"No rows in {input_path}")
    if max_samples:
        rows = rows[:max_samples]

    # Score all rows first; stratified split happens after scoring
    # (we need difficulty scores to stratify by them)
    all_rows = rows
    prompts = [r["prompt"] for r in all_rows]

    # Derive default cache path alongside the output file
    if gen_cache is None:
        gen_cache = output_path.parent / (output_path.stem + ".gen_cache.jsonl")

    if gen_cache.exists():
        cached = load_jsonl(gen_cache)
        if len(cached) == len(prompts):
            print(f"Loading {len(cached)} cached generations from {gen_cache}")
            all_generated = [c["generated"] for c in cached]
        else:
            print(f"Cache size mismatch ({len(cached)} vs {len(prompts)}), re-generating...")
            gen_cache.unlink()
            all_generated = None
    else:
        all_generated = None

    if all_generated is None:
        # vLLM initializes the model once and batches all prompts internally.
        print(f"Generating {len(prompts)} responses with HF transformers (model: {model_id}, batch_size={batch_size})...")
        all_generated = generate_responses(prompts, model_id, max_tokens, temperature, batch_size=batch_size)
        save_jsonl(gen_cache, [{"generated": g} for g in all_generated])
        print(f"Cached {len(all_generated)} generations to {gen_cache}")

    chosen = [r["chosen"] for r in all_rows]
    rejected = [r["rejected"] for r in all_rows]
    print(f"Scoring with {scorer_name}...")
    scorer = get_scorer(scorer_name)
    scores_chosen = scorer.score(all_generated, chosen)
    print(f"Scoring similarity to rejected responses...")
    scores_rejected = scorer.score(all_generated, rejected)

    epsilon = 1e-6
    out_rows = []
    for r, gen, sc, sr in zip(all_rows, all_generated, scores_chosen, scores_rejected):
        # Shift cosine sim from [-1,1] to [0,2] so ratio is always well-defined
        ratio = (sc + 1) / (sr + 1 + epsilon)
        out = {
            **r,
            "generated": gen,
            "similarity_score": round(sc, 6),
            "similarity_rejected": round(sr, 6),
            "ratio_score": round(ratio, 6),
        }
        out_rows.append(out)

    # Stratified split by difficulty: ensures equal representation of easy/hard
    # examples in both train and test sets.
    train_out_rows, test_out_rows = stratified_split_train_test(
        out_rows, train_ratio=split_train_ratio, seed=split_seed
    )
    print(
        f"Stratified split: {len(train_out_rows)} train ({100*split_train_ratio:.0f}%) "
        f"and {len(test_out_rows)} test ({100*(1-split_train_ratio):.0f}%) "
        f"— equal difficulty distribution per quartile"
    )

    # Save with _train and _test suffixes
    train_path = output_path.parent / (output_path.stem + "_train.jsonl")
    test_path = output_path.parent / (output_path.stem + "_test.jsonl")

    save_jsonl(train_path, train_out_rows)
    print(f"Wrote {len(train_out_rows)} scored train rows to {train_path}")

    save_jsonl(test_path, test_out_rows)
    print(f"Wrote {len(test_out_rows)} scored test rows to {test_path}")


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Phase 1: Generate and score easiness (similarity to chosen) with train/test split.")
    p.add_argument("--input", type=Path, default=root / "data" / "processed" / "clean_parsed.jsonl", help="Input JSONL (prompt/chosen/rejected)")
    p.add_argument("--output", type=Path, default=root / "data" / "processed" / "scored_dataset.jsonl", help="Output base path (outputs _train and _test variants)")
    p.add_argument("--scorer", choices=["sentence-transformer", "bertscore"], default="sentence-transformer", help="Similarity: sentence-transformer (default) or bertscore")
    p.add_argument("--model", default="informatiker/Llama-3-8B-Instruct-abliterated", help="HF model for generation")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per generation")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for HF generation (reduce for larger models)")
    p.add_argument("--max-samples", type=int, default=None, help="Cap number of examples to process (default: all)")
    p.add_argument("--gen-cache", type=Path, default=None, help="Path to cache generated responses (avoids re-running on retry)")
    p.add_argument("--split-train-ratio", type=float, default=0.8, help="Proportion for training split (default 0.8 for 80/20)")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for reproducible split (default 42)")
    args = p.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        scorer_name=args.scorer,
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        gen_cache=args.gen_cache,
        split_train_ratio=args.split_train_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    cli()
