"""
Phase 2: DPO fine-tuning with PEFT (LoRA) via trl DPOTrainer.

Uses LoRA on Llama attention modules (q_proj, v_proj). Accepts curriculum or
baseline dataset path, output_dir, and standard hyperparameters.

Usage:
  python -m src.phase2.train_dpo --dataset_path data/processed/curriculum_dataset.jsonl --output_dir outputs/dpo_curriculum
  python -m src.phase2.train_dpo --dataset_path data/processed/baseline_dataset.jsonl --output_dir outputs/dpo_baseline
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset, Sampler, SequentialSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


class SequentialDPOTrainer(DPOTrainer):
    """DPOTrainer that preserves dataset ordering (no shuffle)."""

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> torch.utils.data.Sampler:
        if train_dataset is None:
            train_dataset = self.train_dataset
        return SequentialSampler(train_dataset)


class StagedSampler(Sampler):
    """Sampler that serves a different stage of data each epoch, shuffled independently.

    The sorted dataset is split into `n_stages` equal buckets (easy → hard).
    Each epoch, the sampler yields shuffled indices from one stage according
    to the provided schedule (e.g. [0,0,1,1,2,2] = easy×2, medium×2, hard×2).
    """

    def __init__(self, n_total: int, n_stages: int, schedule: list[int], seed: int = 42):
        self.n_total = n_total
        self.schedule = schedule
        self.seed = seed

        stage_size = n_total // n_stages
        self.stage_slices = []
        for i in range(n_stages):
            start = i * stage_size
            end = (i + 1) * stage_size if i < n_stages - 1 else n_total
            self.stage_slices.append(list(range(start, end)))

        self._epoch = 0

    def __iter__(self):
        stage_idx = self.schedule[self._epoch % len(self.schedule)]
        indices = self.stage_slices[stage_idx].copy()
        random.Random(self.seed + self._epoch).shuffle(indices)
        self._epoch += 1
        return iter(indices)

    def __len__(self):
        stage_idx = self.schedule[self._epoch % len(self.schedule)]
        return len(self.stage_slices[stage_idx])


class StagedDPOTrainer(DPOTrainer):
    """DPOTrainer with staged curriculum: each epoch trains on a different difficulty stage."""

    def __init__(self, *args, n_stages: int = 3, stage_schedule: list[int] | None = None,
                 staged_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_stages = n_stages
        self.stage_schedule = stage_schedule or []
        self.staged_seed = staged_seed

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Sampler:
        if train_dataset is None:
            train_dataset = self.train_dataset
        return StagedSampler(
            n_total=len(train_dataset),
            n_stages=self.n_stages,
            schedule=self.stage_schedule,
            seed=self.staged_seed,
        )


class CompetenceSampler(Sampler):
    """Competence-based curriculum sampler (Platanios et al., 2019).

    Maintains a growing pool of available examples based on a sqrt pacing function:
        c(t) = c0 + (1 - c0) * sqrt(t / T)
    where t is the current optimizer step and T is total optimizer steps.

    Examples are ranked by difficulty percentile d_i ∈ [0, 1] (0 = easiest).
    At step t, only examples with d_i <= c(t) are eligible for sampling.

    Yields total_steps * batch_size * grad_accum_steps indices total (one "epoch"
    of T optimizer steps). Sampling within the eligible pool is uniform with
    replacement (needed early when the pool is smaller than the effective batch).

    Seed defaults to time-based (stochastic on every run); pass an explicit seed
    for reproducibility.
    """

    def __init__(
        self,
        difficulties: np.ndarray,
        total_steps: int,
        batch_size: int,
        grad_accum_steps: int,
        c0: float = 0.01,
        pacing: str = "sqrt",
        seed: int | None = None,
    ):
        self.difficulties = np.asarray(difficulties, dtype=np.float32)
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.c0 = c0
        self.pacing = pacing
        self.seed = seed

    def __len__(self) -> int:
        return self.total_steps * self.batch_size * self.grad_accum_steps

    def __iter__(self):
        seed = self.seed if self.seed is not None else int(time.time() * 1000) % (2 ** 31)
        rng = np.random.default_rng(seed)
        effective_batch = self.batch_size * self.grad_accum_steps
        T = self.total_steps

        print(f"[CompetenceSampler] T={T}, c0={self.c0}, effective_batch={effective_batch}, seed={seed}")

        for step in range(T):
            if self.pacing == "linear":
                # Platanios et al. 2019, eq. 5: c(t) = min(1, t*(1-c0)/T + c0)
                c_t = min(1.0, step * (1.0 - self.c0) / T + self.c0) if T > 1 else 1.0
            else:
                # Platanios et al. 2019, eq. 7 (default sqrt): c(t) = sqrt((1-c0^2)*t/T + c0^2)
                c_t = math.sqrt((1.0 - self.c0 ** 2) * step / T + self.c0 ** 2) if T > 1 else 1.0
            available = np.where(self.difficulties <= c_t)[0]
            # Sample with replacement so we always fill the batch even when pool is small
            indices = rng.choice(available, size=effective_batch, replace=True)
            yield from indices.tolist()


class CompetenceDPOTrainer(DPOTrainer):
    """DPOTrainer with competence-based curriculum (Platanios et al., 2019).

    Trains for a single logical epoch of `competence_total_steps` optimizer steps.
    The available pool of examples grows continuously via a sqrt pacing function.
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def run(
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    max_prompt_length: int,
    beta: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | None = None,
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    sequential: bool = False,
    staged: bool = False,
    n_stages: int = 3,
    staged_seed: int = 42,
    competence: bool = False,
    competence_c0: float = 0.01,
    competence_pacing: str = "sqrt",
    competence_total_steps: int | None = None,
    competence_seed: int | None = None,
    eval_test_path: Path | None = None,
    eval_every_steps: int = 50,
    eval_output_jsonl: Path | None = None,
    save_checkpoints: bool = True,
) -> None:
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset (JSONL with prompt, chosen, rejected)
    dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")

    # Competence-based curriculum: compute difficulty percentile ranks from margin scores.
    # difficulty=0 → easiest (highest margin sc-sr), difficulty≈1 → hardest.
    # Requires similarity_score and similarity_rejected fields in the JSONL.
    comp_difficulties: np.ndarray | None = None
    comp_total_steps: int = 0
    if competence:
        if "similarity_score" not in dataset.column_names or "similarity_rejected" not in dataset.column_names:
            raise ValueError(
                "Competence curriculum requires 'similarity_score' and 'similarity_rejected' "
                "fields in the dataset JSONL. Use a curriculum_dataset_train.jsonl from score_easiness."
            )
        margins = np.array(dataset["similarity_score"]) - np.array(dataset["similarity_rejected"])
        N = len(margins)
        # Rank descending (highest margin = easiest = rank 0 = difficulty 0)
        sorted_desc = np.argsort(-margins)
        difficulties = np.empty(N, dtype=np.float32)
        difficulties[sorted_desc] = np.arange(N, dtype=np.float32) / N
        comp_difficulties = difficulties

        # Total optimizer steps T: default to num_train_epochs × steps_per_epoch
        effective_batch = per_device_train_batch_size * gradient_accumulation_steps
        steps_per_epoch = math.ceil(N / effective_batch)
        comp_total_steps = competence_total_steps if competence_total_steps is not None \
            else num_train_epochs * steps_per_epoch

        print(f">>> Competence curriculum: N={N}, effective_batch={effective_batch}, "
              f"T={comp_total_steps} steps, c0={competence_c0}, pacing={competence_pacing} <<<")
        print(f">>> Difficulty range: [{difficulties.min():.4f}, {difficulties.max():.4f}], "
              f"initial pool: ~{int(competence_c0 * N)} examples <<<")
        print(f">>> Equivalent full-dataset epochs: {comp_total_steps / steps_per_epoch:.1f} <<<")


    # LoRA: auto-detect target modules or use provided list.
    # Llama / Gemma / Qwen all use q_proj + v_proj; gate_proj added for MoE variants.
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # For multimodal models (e.g. Gemma3ForConditionalGeneration), TRL expects an
    # AutoProcessor (which exposes .tokenizer). For text-only models, use AutoTokenizer.
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch = (config.architectures or [""])[0]
    def _fix_pad_token(tok) -> None:
        """Ensure pad_token_id is in-vocab. Newer transformers auto-adds '[PAD]' at
        vocab_size (out of range) for Llama tokenizers, causing embedding OOB asserts."""
        if tok.pad_token is None or tok.pad_token_id is None or tok.pad_token_id >= tok.vocab_size:
            tok.pad_token_id = tok.eos_token_id

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

    # Keep a reference to the tokenizer for saving
    tokenizer = getattr(processing_class, "tokenizer", processing_class)

    from_pretrained_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    # For text-only models, override model_type to prevent TRL from treating as vision model
    if is_text_only:
        from_pretrained_kwargs["model_type"] = "llama"  # Neutral text-only type

    model = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        # Competence mode: 1 epoch; max_steps=T stops training after exactly T optimizer steps.
        num_train_epochs=1 if competence else num_train_epochs,
        max_steps=comp_total_steps if competence else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        beta=beta,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="steps" if save_checkpoints else "no",
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=processing_class,  # trl>=0.9: processor for VLMs, tokenizer for text-only
        peft_config=peft_config,
    )

    if competence:
        assert comp_difficulties is not None
        trainer = CompetenceDPOTrainer(
            **trainer_kwargs,
            competence_difficulties=comp_difficulties,
            competence_total_steps=comp_total_steps,
            competence_batch_size=per_device_train_batch_size,
            competence_grad_accum=gradient_accumulation_steps,
            competence_c0=competence_c0,
            competence_pacing=competence_pacing,
            competence_seed=competence_seed,
        )
    elif staged:
        # Schedule: [0, 1, 2, ..., n_stages-1] × num_train_epochs
        # Each cycle = 1 full dataset pass (N/3 per stage × 3 stages = N)
        # Total stage-epochs = n_stages × num_train_epochs
        one_cycle = list(range(n_stages))
        schedule = one_cycle * num_train_epochs
        total_stage_epochs = len(schedule)

        # Override num_train_epochs to the actual number of stage-epochs
        training_args.num_train_epochs = total_stage_epochs
        print(f">>> Staged curriculum: {n_stages} stages, {num_train_epochs} full passes "
              f"→ {total_stage_epochs} stage-epochs <<<")
        print(f">>> Schedule: {schedule} <<<")

        trainer = StagedDPOTrainer(
            **trainer_kwargs,
            n_stages=n_stages,
            stage_schedule=schedule,
            staged_seed=staged_seed,
        )
    elif sequential:
        print(">>> Using SequentialDPOTrainer (no shuffle, preserving dataset order) <<<")
        trainer = SequentialDPOTrainer(**trainer_kwargs)
    else:
        trainer = DPOTrainer(**trainer_kwargs)

    if eval_test_path is not None:
        from src.eval.test_margin_callback import TestMarginCallback
        out_jsonl = eval_output_jsonl if eval_output_jsonl is not None else (output_dir / "eval_margin.jsonl")
        cb = TestMarginCallback(
            test_path=eval_test_path,
            output_jsonl=out_jsonl,
            eval_every_steps=eval_every_steps,
            batch_size=8,
            step_offset=0,
            max_length=max_length,
        )
        trainer.add_callback(cb)
        print(f">>> TestMarginCallback enabled: every {eval_every_steps} steps, "
              f"test={eval_test_path} → {out_jsonl} <<<")

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def cli() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Phase 2: DPO fine-tuning with PEFT (LoRA).")
    p.add_argument("--dataset_path", type=Path, required=True, help="Path to JSONL (curriculum or baseline) with prompt/chosen/rejected")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to save checkpoints and final adapter")
    p.add_argument("--model", default="informatiker/Llama-3-8B-Instruct-abliterated", help="Base causal LM to fine-tune")
    p.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate")
    p.add_argument("--max_length", type=int, default=1024, help="Max total sequence length")
    p.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta (preference strength)")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"],
                   help="LoRA target module names (default: q_proj v_proj, works for Llama/Gemma/Qwen)")
    p.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"],
                   help="Attention implementation override (use 'eager' for old Llama-2 models)")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit (NF4) quantization for large models that don't fit in bf16")
    p.add_argument("--sequential", action="store_true",
                   help="Disable dataloader shuffle (preserve dataset order for curriculum training)")
    p.add_argument("--staged", action="store_true",
                   help="Staged curriculum: each epoch trains on one difficulty stage (easy→medium→hard), "
                        "repeated for num_train_epochs full passes. Dataset must be sorted easy→hard.")
    p.add_argument("--n_stages", type=int, default=3,
                   help="Number of difficulty stages for staged curriculum (default: 3)")
    p.add_argument("--staged_seed", type=int, default=42,
                   help="Random seed for staged within-stage shuffling (default: 42)")
    p.add_argument("--competence", action="store_true",
                   help="Competence-based curriculum (Platanios et al. 2019): examples unlock "
                        "via sqrt pacing c(t)=c0+(1-c0)*sqrt(t/T). Trains for 1 epoch of T steps. "
                        "Dataset must have similarity_score and similarity_rejected fields.")
    p.add_argument("--competence_c0", type=float, default=0.01,
                   help="Initial competence: fraction of easiest examples available at step 0 (default: 0.01)")
    p.add_argument("--competence_pacing", type=str, default="sqrt", choices=["sqrt", "linear"],
                   help="Pacing function variant (Platanios et al. 2019): "
                        "sqrt = eq.7 c(t)=sqrt((1-c0^2)*t/T+c0^2), "
                        "linear = eq.5 c(t)=min(1, t*(1-c0)/T+c0) (default: sqrt)")
    p.add_argument("--competence_total_steps", type=int, default=None,
                   help="Total optimizer steps T (default: num_train_epochs × ceil(N/effective_batch))")
    p.add_argument("--competence_seed", type=int, default=None,
                   help="RNG seed for competence sampling (default: time-based, stochastic on each run)")
    p.add_argument("--eval_test_path", type=Path, default=None,
                   help="If set, run TestMarginCallback to compute test-set reward margin every K steps")
    p.add_argument("--eval_every_steps", type=int, default=50,
                   help="How often (in optimizer steps) to evaluate test margin (default: 50)")
    p.add_argument("--eval_output_jsonl", type=Path, default=None,
                   help="Where to write eval records (default: <output_dir>/eval_margin.jsonl)")
    p.add_argument("--no_save_checkpoints", action="store_true",
                   help="Disable intermediate checkpoint saving (only the final adapter is kept)")
    args = p.parse_args()

    run(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model,
        num_train_epochs=args.num_train_epochs,
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
        load_in_4bit=args.load_in_4bit,
        sequential=args.sequential,
        staged=args.staged,
        n_stages=args.n_stages,
        staged_seed=args.staged_seed,
        competence=args.competence,
        competence_c0=args.competence_c0,
        competence_pacing=args.competence_pacing,
        competence_total_steps=args.competence_total_steps,
        competence_seed=args.competence_seed,
        eval_test_path=args.eval_test_path,
        eval_every_steps=args.eval_every_steps,
        eval_output_jsonl=args.eval_output_jsonl,
        save_checkpoints=not args.no_save_checkpoints,
    )


if __name__ == "__main__":
    cli()
