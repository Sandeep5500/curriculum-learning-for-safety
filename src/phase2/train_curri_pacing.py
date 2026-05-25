"""Curri-Pacing: Curri-DPO with sqrt-pacing sampling inside each stage.

Combines the two curriculum methods:
  * Curri-DPO's 3-stage setup: split data into easy/medium/hard buckets,
    update the reference model after each stage.
  * Sqrt pacing inside each stage: within a bucket, examples are further
    ranked by difficulty (easiest-of-easy → toughest-of-easy, etc.), and a
    Platanios et al. (2019) competence-based sampler grows the available
    pool over the stage's optimizer steps.

The outer stage progression (easy → medium → hard) is discrete; the inner
progression (easy-of-stage → hard-of-stage) is continuous via sqrt pacing.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from torch.utils.data import Sampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.phase2.train_dpo import CompetenceSampler

STAGE_LABELS = ["EASY", "MEDIUM", "HARD", "STAGE_3", "STAGE_4"]


def _fix_pad_token(tok) -> None:
    """Ensure pad_token_id is in-vocab (Llama tokenizers can set it OOB)."""
    if tok.pad_token is None or tok.pad_token_id is None or tok.pad_token_id >= tok.vocab_size:
        tok.pad_token_id = tok.eos_token_id


class CompetenceDPOTrainer(DPOTrainer):
    """DPOTrainer that overrides the train sampler to use CompetenceSampler.

    Used per-stage in curri-pacing: each stage runs for `total_steps` optimizer
    steps, and the sampler starts with the easiest c0 fraction of the stage's
    examples and grows to the full stage via sqrt pacing.
    """

    def __init__(
        self,
        *args,
        competence_difficulties: np.ndarray,
        competence_total_steps: int,
        competence_batch_size: int,
        competence_grad_accum: int,
        competence_c0: float = 0.01,
        competence_pacing: str = "sqrt",
        competence_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._comp_difficulties = competence_difficulties
        self._comp_total_steps = competence_total_steps
        self._comp_batch_size = competence_batch_size
        self._comp_grad_accum = competence_grad_accum
        self._comp_c0 = competence_c0
        self._comp_pacing = competence_pacing
        self._comp_seed = competence_seed

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Sampler:
        return CompetenceSampler(
            difficulties=self._comp_difficulties,
            total_steps=self._comp_total_steps,
            batch_size=self._comp_batch_size,
            grad_accum_steps=self._comp_grad_accum,
            c0=self._comp_c0,
            pacing=self._comp_pacing,
            seed=self._comp_seed,
        )


def _bucket_difficulties(margins_slice: np.ndarray) -> np.ndarray:
    """Convert a bucket's margins to [0, 1] difficulty ranks (0 = easiest).

    Highest margin → easiest → rank 0. Ranks are dense (i/N), so within the
    bucket the easiest c0 fraction is eligible at step 0 and the bucket fills
    to full availability by the end of the stage.
    """
    n = len(margins_slice)
    sorted_desc = np.argsort(-margins_slice)  # highest margin first
    ranks = np.empty(n, dtype=np.float32)
    ranks[sorted_desc] = np.arange(n, dtype=np.float32) / max(n, 1)
    return ranks


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
    rpo_alpha: float = 0.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    attn_implementation: str | None = None,
    bf16: bool | None = None,
    competence_c0: float = 0.01,
    competence_pacing: str = "sqrt",
    competence_seed: int | None = None,
    eval_test_path: Path | None = None,
    eval_every_steps: int = 50,
    eval_output_jsonl: Path | None = None,
    save_checkpoints: bool = True,
    start_stage: int = 0,
) -> None:
    if bf16 is None:
        bf16 = torch.cuda.is_available()
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ── Load dataset and split into difficulty buckets (easy → hard) ─────────
    dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")
    if "similarity_score" not in dataset.column_names or "similarity_rejected" not in dataset.column_names:
        raise ValueError(
            "Curri-Pacing requires 'similarity_score' and 'similarity_rejected' fields. "
            "Use a curriculum_dataset_train.jsonl from score_easiness."
        )

    margins_all = np.array(dataset["similarity_score"]) - np.array(dataset["similarity_rejected"])
    sorted_indices = np.argsort(-margins_all)  # highest margin (easiest) first

    N = len(dataset)
    bucket_size = N // n_stages
    buckets: list[Dataset] = []
    bucket_margins: list[np.ndarray] = []
    for i in range(n_stages):
        start = i * bucket_size
        end = (i + 1) * bucket_size if i < n_stages - 1 else N
        idxs = sorted_indices[start:end].tolist()
        buckets.append(dataset.select(idxs))
        bucket_margins.append(margins_all[idxs])

    print(f"=== Curri-Pacing: {n_stages} stages × {epochs_per_stage} epochs/stage, "
          f"inner sampler=sqrt (c0={competence_c0}) ===")
    for i, b in enumerate(buckets):
        label = STAGE_LABELS[i] if i < len(STAGE_LABELS) else f"STAGE_{i}"
        m = bucket_margins[i]
        print(f"  Stage {i} ({label}): {len(b)} examples, margin [{m.min():.4f}, {m.max():.4f}]")

    # ── Tokenizer / processor ────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch = (config.architectures or [""])[0]
    # Text-only models that should use AutoTokenizer even if they have ConditionalGeneration in arch
    text_only_models = ["gemma-3-4b", "gemma-3", "gemma3"]
    is_text_only = any(m in model_name.lower() for m in text_only_models)

    if "ConditionalGeneration" in arch and not is_text_only:
        processing_class = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _fix_pad_token(processing_class.tokenizer)
    else:
        processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _fix_pad_token(processing_class)
        # Expose tokenizer as itself for TRL compatibility
        processing_class.tokenizer = processing_class
    tokenizer = getattr(processing_class, "tokenizer", processing_class)

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

    effective_batch = per_device_train_batch_size * gradient_accumulation_steps

    current_base_path = model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # If resuming mid-run, fast-forward base model to the last completed merge
    if start_stage > 0:
        resume_base = output_dir / f"stage_{start_stage - 1}" / "merged"
        if not resume_base.exists():
            raise FileNotFoundError(f"Cannot resume from stage {start_stage}: {resume_base} not found")
        current_base_path = str(resume_base)
        print(f"[Resume] Starting from stage {start_stage}, base model: {current_base_path}")

    # Resolve eval JSONL path once so it's stitched across all stages.
    eval_jsonl_path: Path | None = None
    if eval_test_path is not None:
        eval_jsonl_path = eval_output_jsonl if eval_output_jsonl is not None else (output_dir / "eval_margin.jsonl")
        # Only wipe stale records when starting fresh (not resuming mid-run)
        if start_stage == 0 and eval_jsonl_path.exists():
            eval_jsonl_path.unlink()
        print(f">>> Test-margin eval enabled: every {eval_every_steps} steps → {eval_jsonl_path} <<<")

    # Pre-compute cumulative steps for skipped stages so TestMarginCallback offsets are correct
    import math
    cumulative_steps_before_stage = 0
    for i in range(start_stage):
        steps_per_epoch_i = max(math.ceil(len(buckets[i]) / effective_batch), 1)
        cumulative_steps_before_stage += epochs_per_stage * steps_per_epoch_i

    for stage_idx in range(start_stage, n_stages):
        is_last = (stage_idx == n_stages - 1)
        stage_dir = output_dir / f"stage_{stage_idx}"
        label = STAGE_LABELS[stage_idx] if stage_idx < len(STAGE_LABELS) else f"STAGE_{stage_idx}"
        bucket = buckets[stage_idx]

        # Inner difficulties: [0, 1] rank within the bucket (0 = easiest-of-stage)
        inner_difficulties = _bucket_difficulties(bucket_margins[stage_idx])

        # T = epochs_per_stage optimizer steps' worth of data for this bucket
        steps_per_epoch = max(math.ceil(len(bucket) / effective_batch), 1)
        stage_total_steps = epochs_per_stage * steps_per_epoch

        print(f"\n{'=' * 60}")
        print(f"CURRI-PACING STAGE {stage_idx}/{n_stages - 1}: {label}")
        print(f"  examples: {len(bucket)}  effective_batch: {effective_batch}")
        print(f"  T={stage_total_steps} steps  (epochs_per_stage={epochs_per_stage})")
        print(f"  inner sampler: {competence_pacing}  c0={competence_c0}")
        print(f"  base model: {current_base_path}")
        print(f"{'=' * 60}\n")

        model = AutoModelForCausalLM.from_pretrained(current_base_path, **from_pretrained_kwargs)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # CompetenceSampler yields T * effective_batch indices → let HF drive training
        # via max_steps=T with num_train_epochs=1 (single logical pass over the sampler).
        training_args = DPOConfig(
            output_dir=str(stage_dir),
            num_train_epochs=1,
            max_steps=stage_total_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            beta=beta,
            rpo_alpha=rpo_alpha if rpo_alpha > 0.0 else None,
            remove_unused_columns=False,
            logging_steps=10,
            save_strategy="steps" if save_checkpoints else "no",
            save_steps=500,
            save_total_limit=2,
            bf16=bf16,
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = CompetenceDPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=bucket,
            processing_class=processing_class,
            peft_config=peft_config,
            competence_difficulties=inner_difficulties,
            competence_total_steps=stage_total_steps,
            competence_batch_size=per_device_train_batch_size,
            competence_grad_accum=gradient_accumulation_steps,
            competence_c0=competence_c0,
            competence_pacing=competence_pacing,
            competence_seed=competence_seed,
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
        cumulative_steps_before_stage += stage_total_steps

        peft_model = trainer.model

        if is_last:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            (output_dir / "curri_base_model.txt").write_text(current_base_path)
            print(f"\nFinal adapter saved to {output_dir}")
            print(f"Base model for this adapter: {current_base_path}")
            del peft_model
        else:
            merged_path = stage_dir / "merged"
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))
            # Workaround: save_pretrained can drop chat_template in some transformers
            # versions. Restore it from the in-memory tokenizer if needed.
            _tok_cfg_path = merged_path / "tokenizer_config.json"
            if _tok_cfg_path.exists() and tokenizer.chat_template:
                import json as _json
                _cfg = _json.loads(_tok_cfg_path.read_text())
                if not _cfg.get("chat_template"):
                    _cfg["chat_template"] = tokenizer.chat_template
                    _tok_cfg_path.write_text(_json.dumps(_cfg, indent=2))
                    print(f"  Restored chat_template in {_tok_cfg_path}")
            print(f"Merged model saved to {merged_path}")

            del peft_model, merged_model
            torch.cuda.empty_cache()
            gc.collect()

            if stage_idx > 0 and not os.environ.get("KEEP_MERGED"):
                prev_merged = output_dir / f"stage_{stage_idx - 1}" / "merged"
                if prev_merged.exists():
                    shutil.rmtree(prev_merged)
                    print(f"Cleaned up {prev_merged}")

            current_base_path = str(merged_path)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Curri-Pacing training complete: {n_stages} stages × {epochs_per_stage} epochs")
    print(f"Final adapter: {output_dir}")
    print(f"{'=' * 60}")


def cli() -> None:
    p = argparse.ArgumentParser(
        description="Curri-Pacing: Curri-DPO stages with sqrt-pacing sampling inside each stage."
    )
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model", default="informatiker/Llama-3-8B-Instruct-abliterated")
    p.add_argument("--n_stages", type=int, default=3)
    p.add_argument("--epochs_per_stage", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--rpo_alpha", type=float, default=0.0,
                   help="RPO auxiliary NLL loss weight on chosen responses (0 = disabled).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--attn_implementation", default=None,
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--competence_c0", type=float, default=0.01,
                   help="Initial within-stage pool: fraction of easiest examples at stage start")
    p.add_argument("--competence_pacing", type=str, default="sqrt", choices=["sqrt", "linear"])
    p.add_argument("--competence_seed", type=int, default=None)
    p.add_argument("--eval_test_path", type=Path, default=None,
                   help="If set, run TestMarginCallback to compute test-set reward margin every K steps")
    p.add_argument("--eval_every_steps", type=int, default=50,
                   help="How often (in optimizer steps) to evaluate test margin (default: 50)")
    p.add_argument("--eval_output_jsonl", type=Path, default=None,
                   help="Where to write eval records (default: <output_dir>/eval_margin.jsonl)")
    p.add_argument("--no_save_checkpoints", action="store_true",
                   help="Disable intermediate checkpoint saving (only the final adapter is kept)")
    p.add_argument("--start_stage", type=int, default=0,
                   help="Resume from this stage index (0-based). Requires stage_{start_stage-1}/merged to exist.")
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
        rpo_alpha=args.rpo_alpha,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        attn_implementation=args.attn_implementation,
        competence_c0=args.competence_c0,
        competence_pacing=args.competence_pacing,
        competence_seed=args.competence_seed,
        eval_test_path=args.eval_test_path,
        eval_every_steps=args.eval_every_steps,
        eval_output_jsonl=args.eval_output_jsonl,
        save_checkpoints=not args.no_save_checkpoints,
        start_stage=args.start_stage,
    )


if __name__ == "__main__":
    cli()
