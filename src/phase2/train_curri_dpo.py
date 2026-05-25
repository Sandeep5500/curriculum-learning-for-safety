"""Curri-DPO: Multi-stage DPO with reference model updates between stages.

Splits the curriculum-sorted dataset into N equal difficulty buckets (easy → hard),
trains DPO on each bucket sequentially, and updates the reference model after each
stage to the model just trained. The final stage's LoRA adapter is saved for eval.

Inspired by "Enhancing Alignment using Curriculum Learning & Ranked Preferences"
(EMNLP 2024 Findings).
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def _fix_pad_token(tok) -> None:
    """Ensure pad_token_id is in-vocab (Llama tokenizers can set it OOB)."""
    if tok.pad_token is None or tok.pad_token_id is None or tok.pad_token_id >= tok.vocab_size:
        tok.pad_token_id = tok.eos_token_id


STAGE_LABELS = ["EASY", "MEDIUM", "HARD", "STAGE_3", "STAGE_4"]


def run(
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    n_stages: int = 3,
    epochs_per_stage: int = 2,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    attn_implementation: str | None = None,
    bf16: bool | None = None,
    eval_test_path: Path | None = None,
    eval_every_steps: int = 50,
    eval_output_jsonl: Path | None = None,
) -> None:
    # Auto-detect bf16: use it when a CUDA GPU is available, fall back to fp32 on CPU.
    if bf16 is None:
        bf16 = torch.cuda.is_available()
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ── Load dataset and split into difficulty buckets ────────────────────────
    dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")

    if "similarity_score" not in dataset.column_names or "similarity_rejected" not in dataset.column_names:
        raise ValueError(
            "Curri-DPO requires 'similarity_score' and 'similarity_rejected' fields. "
            "Use a curriculum_dataset_train.jsonl from score_easiness."
        )

    margins = np.array(dataset["similarity_score"]) - np.array(dataset["similarity_rejected"])
    sorted_indices = np.argsort(-margins)  # highest margin (easiest) first

    N = len(dataset)
    bucket_size = N // n_stages
    buckets = []
    for i in range(n_stages):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_stages - 1 else N
        bucket_indices = sorted_indices[start:end].tolist()
        buckets.append(dataset.select(bucket_indices))

    print(f"=== Curri-DPO: {n_stages} stages × {epochs_per_stage} epochs/stage ===")
    for i, b in enumerate(buckets):
        label = STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"STAGE_{i}"
        margin_slice = margins[sorted_indices[i * bucket_size:(i + 1) * bucket_size if i < n_stages - 1 else N]]
        print(f"  Stage {i} ({label}): {len(b)} examples, "
              f"margin range [{margin_slice.min():.4f}, {margin_slice.max():.4f}]")

    # ── Load tokenizer (reused across all stages) ────────────────────────────
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch = (config.architectures or [""])[0]

    text_only_models = ["gemma-3-4b", "gemma-3", "gemma3"]
    is_text_only = any(m in model_name.lower() for m in text_only_models)
    if "ConditionalGeneration" in arch and not is_text_only:
        processing_class = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _fix_pad_token(processing_class.tokenizer)
    else:
        processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _fix_pad_token(processing_class)

    tokenizer = getattr(processing_class, "tokenizer", processing_class)

    # ── Common model loading kwargs ──────────────────────────────────────────
    from_pretrained_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    # For text-only models, override model_type to prevent TRL from treating as vision model
    if is_text_only:
        from_pretrained_kwargs["model_type"] = "llama"  # Neutral text-only type

    # ── Stage loop ───────────────────────────────────────────────────────────
    current_base_path = model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve eval JSONL path once so it's stitched across all stages.
    eval_jsonl_path: Path | None = None
    if eval_test_path is not None:
        eval_jsonl_path = eval_output_jsonl if eval_output_jsonl is not None else (output_dir / "eval_margin.jsonl")
        if eval_jsonl_path.exists():
            eval_jsonl_path.unlink()
        print(f">>> Test-margin eval enabled: every {eval_every_steps} steps → {eval_jsonl_path} <<<")

    # Pre-compute steps per stage so TestMarginCallback offsets are correct
    import math
    world_size = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps * world_size
    stage_steps: list[int] = []
    for b in buckets:
        steps_per_epoch = max(math.ceil(len(b) / effective_batch), 1)
        stage_steps.append(epochs_per_stage * steps_per_epoch)
    cumulative_steps_before_stage = 0

    for stage_idx in range(n_stages):
        is_last = (stage_idx == n_stages - 1)
        stage_dir = output_dir / f"stage_{stage_idx}"
        label = STAGE_LABELS[stage_idx] if stage_idx < len(STAGE_LABELS) else f"STAGE_{stage_idx}"

        print(f"\n{'=' * 60}")
        print(f"CURRI-DPO STAGE {stage_idx}/{n_stages - 1}: {label} "
              f"({len(buckets[stage_idx])} examples, {epochs_per_stage} epochs)")
        print(f"Base model: {current_base_path}")
        ref_label = "updated from previous stage" if stage_idx > 0 else "initial"
        print(f"Ref model: {current_base_path} ({ref_label}, shared base via TRL peft_config)")
        print(f"{'=' * 60}\n")

        # Load model with fresh LoRA
        model = AutoModelForCausalLM.from_pretrained(current_base_path, **from_pretrained_kwargs)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = DPOConfig(
            output_dir=str(stage_dir),
            num_train_epochs=epochs_per_stage,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=beta,
            remove_unused_columns=False,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            bf16=bf16,
            report_to="none",
            dataloader_num_workers=0,
        )

        # All stages: pass ref_model=None and let TRL handle the reference internally.
        # TRL with peft_config + ref_model=None freezes the base weights as the reference
        # and applies LoRA only to the policy. Since current_base_path is updated to the
        # merged model after each stage, the reference is always the "model just trained" —
        # exactly the Curri-DPO update rule. This also avoids loading a second full copy of
        # the model (which would OOM on a 48 GB GPU with 8B parameter models).
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=buckets[stage_idx],
            processing_class=processing_class,
            peft_config=peft_config,
        )

        if eval_jsonl_path is not None:
            from src.eval.test_margin_callback import TestMarginCallback
            cb = TestMarginCallback(
                test_path=eval_test_path,
                output_jsonl=eval_jsonl_path,
                eval_every_steps=eval_every_steps,
                batch_size=8,
                step_offset=cumulative_steps_before_stage,
                max_length=max_length,
            )
            trainer.add_callback(cb)
            print(f">>> Stage {stage_idx} TestMarginCallback: step_offset={cumulative_steps_before_stage} <<<")

        trainer.train()
        cumulative_steps_before_stage += stage_steps[stage_idx]

        # Get the PEFT-wrapped model (either applied by TRL or by us above)
        peft_model = trainer.model

        if is_last:
            # Save final LoRA adapter to output_dir (eval-compatible)
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            # Record which base model this adapter should be loaded on
            (output_dir / "curri_base_model.txt").write_text(current_base_path)
            print(f"\nFinal adapter saved to {output_dir}")
            print(f"Base model for this adapter: {current_base_path}")
            del peft_model
        else:
            # Merge LoRA into base → save merged model → becomes base for next stage
            merged_path = stage_dir / "merged"
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))
            print(f"Merged model saved to {merged_path}")

            # Free peft_model and merged_model before next stage.
            # peft_model must be explicitly deleted here — it persists across loop
            # iterations otherwise, keeping stage N's GPU tensors alive into stage N+1.
            del peft_model, merged_model
            torch.cuda.empty_cache()
            gc.collect()

            # Delete previous stage's merged weights (no longer needed)
            if stage_idx > 0:
                prev_merged = output_dir / f"stage_{stage_idx - 1}" / "merged"
                if prev_merged.exists():
                    shutil.rmtree(prev_merged)
                    print(f"Cleaned up {prev_merged}")

            current_base_path = str(merged_path)

        # Free GPU memory before next stage
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Curri-DPO training complete: {n_stages} stages × {epochs_per_stage} epochs")
    print(f"Final adapter: {output_dir}")
    print(f"{'=' * 60}")


def cli() -> None:
    p = argparse.ArgumentParser(
        description="Curri-DPO: Multi-stage DPO with reference model updates between stages."
    )
    p.add_argument("--dataset_path", type=Path, required=True,
                   help="Curriculum-sorted JSONL (easy→hard) with prompt/chosen/rejected + similarity fields")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory to save final LoRA adapter")
    p.add_argument("--model", default="informatiker/Llama-3-8B-Instruct-abliterated",
                   help="Base causal LM (HF model ID or local path)")
    p.add_argument("--n_stages", type=int, default=3,
                   help="Number of difficulty stages (default: 3)")
    p.add_argument("--epochs_per_stage", type=int, default=2,
                   help="Training epochs per stage (default: 2)")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--attn_implementation", default=None,
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--eval_test_path", type=Path, default=None,
                   help="If set, run TestMarginCallback to compute test-set reward margin every K steps")
    p.add_argument("--eval_every_steps", type=int, default=50,
                   help="How often (in optimizer steps) to evaluate test margin (default: 50)")
    p.add_argument("--eval_output_jsonl", type=Path, default=None,
                   help="Where to write eval records (default: <output_dir>/eval_margin.jsonl)")
    args = p.parse_args()

    run(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model,
        n_stages=args.n_stages,
        epochs_per_stage=args.epochs_per_stage,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        attn_implementation=args.attn_implementation,
        eval_test_path=args.eval_test_path,
        eval_every_steps=args.eval_every_steps,
        eval_output_jsonl=args.eval_output_jsonl,
    )


if __name__ == "__main__":
    cli()
