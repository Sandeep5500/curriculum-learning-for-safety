"""TrainerCallback that periodically evaluates DPO reward margin on a fixed
held-out test set. Used to plot reward-margin trajectory across training steps
for fair cross-method comparison.

Metric (matches src/eval/evaluate_dpo_test.py):
    margin_i  = log p_theta(chosen_i | prompt_i) - log p_theta(rejected_i | prompt_i)
    accuracy  = (margin_i > 0).mean()

This is the pure policy log-prob margin (no reference model term, no beta scaling).
The reference-model term cancels when comparing two methods that share the same
underlying base, and beta is identical (0.1) across all our runs.

Records are appended to a JSONL file with global_step, accuracy, mean_margin,
and (for staged trainers) a step_offset that lets you stitch multiple stages
into a single monotonic step axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import TrainerCallback


class TestMarginCallback(TrainerCallback):
    """Evaluate test-set reward margin every `eval_every_steps` global steps.

    Args:
        test_path: JSONL file with prompt/chosen/rejected fields.
        output_jsonl: where to append eval records.
        eval_every_steps: how often to fire (in optimizer steps).
        batch_size: number of sequences per forward pass during eval.
        step_offset: added to state.global_step before logging — used by
            curri-pacing to stitch per-stage step counts into one axis.
    """

    def __init__(
        self,
        test_path: str | Path,
        output_jsonl: str | Path,
        eval_every_steps: int = 50,
        batch_size: int = 8,
        step_offset: int = 0,
        max_length: int = 1024,
    ):
        self.test_path = Path(test_path)
        self.output_jsonl = Path(output_jsonl)
        self.eval_every_steps = int(eval_every_steps)
        self.batch_size = int(batch_size)
        self.step_offset = int(step_offset)
        self.max_length = int(max_length)

        self._pairs: list[dict] | None = None
        self._tokenizer = None  # set in on_train_begin
        # Cached pre-tokenized sequences (input_ids tensor + prompt_len)
        self._chosen_cache: list[tuple[torch.Tensor, int]] = []
        self._rejected_cache: list[tuple[torch.Tensor, int]] = []
        self._last_eval_step: int = -1

    # ─────────────────────────────────────────────────────────── lifecycle

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self._pairs is not None:
            return
        # Resolve tokenizer
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        if tokenizer is None:
            # Fall back: try to get it from the model
            tokenizer = kwargs.get("tokenizer")
        self._tokenizer = tokenizer

        # Load test pairs
        rows: list[dict] = []
        with open(self.test_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        self._pairs = rows
        print(f"[TestMarginCallback] loaded {len(rows)} test pairs from {self.test_path}")

        # Pre-tokenize once
        for row in rows:
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            prompt_ids = self._tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            chosen_full = self._tokenizer(prompt + chosen, add_special_tokens=False,
                                          return_tensors="pt", truncation=True,
                                          max_length=self.max_length).input_ids[0]
            rejected_full = self._tokenizer(prompt + rejected, add_special_tokens=False,
                                            return_tensors="pt", truncation=True,
                                            max_length=self.max_length).input_ids[0]
            self._chosen_cache.append((chosen_full, int(prompt_ids.shape[0])))
            self._rejected_cache.append((rejected_full, int(prompt_ids.shape[0])))

        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.step_offset == 0 and self.output_jsonl.exists():
            # Fresh run with stage 0: clear stale file from a prior aborted run
            self.output_jsonl.unlink()

        # Run an eval at step 0 (untrained adapter) only when starting fresh.
        if self.step_offset == 0 and model is not None:
            self._run_eval(model, current_step=0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step == self._last_eval_step:
            return
        if step % self.eval_every_steps != 0:
            return
        self._run_eval(model, current_step=step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step != self._last_eval_step:
            self._run_eval(model, current_step=step)

    # ─────────────────────────────────────────────────────────── eval impl

    @torch.no_grad()
    def _run_eval(self, model, current_step: int) -> None:
        if model is None or self._tokenizer is None or not self._chosen_cache:
            return

        was_training = model.training
        model.eval()

        device = next(model.parameters()).device
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        # Bypass accelerate's convert_to_fp32 wrapper to avoid OOM on large-vocab models
        # (e.g. qwen3 with 151k tokens). Patch at module level so the wrapper calls
        # the no-op instead of allocating a full fp32 logit tensor.
        import accelerate.utils.operations as _acc_ops
        _orig_ctf = _acc_ops.convert_to_fp32
        _acc_ops.convert_to_fp32 = lambda x: x
        try:
            chosen_lps = self._batched_log_probs(model, self._chosen_cache, pad_id, device)
            rejected_lps = self._batched_log_probs(model, self._rejected_cache, pad_id, device)
        finally:
            _acc_ops.convert_to_fp32 = _orig_ctf
        margins = np.array(chosen_lps, dtype=np.float64) - np.array(rejected_lps, dtype=np.float64)

        record = {
            "global_step": int(current_step) + self.step_offset,
            "trainer_step": int(current_step),
            "step_offset": self.step_offset,
            "n_pairs": int(len(self._pairs)),
            "accuracy": float((margins > 0).mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
        }
        with open(self.output_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[TestMarginCallback] step={record['global_step']} "
            f"acc={record['accuracy']:.4f} mean_margin={record['mean_margin']:.3f}"
        )
        self._last_eval_step = current_step

        if was_training:
            model.train()

    def _batched_log_probs(
        self,
        model,
        cache: list[tuple[torch.Tensor, int]],
        pad_id: int,
        device,
    ) -> list[float]:
        """For each (full_ids, prompt_len), compute sum log p(response | prompt)."""
        out: list[float] = []
        for batch_start in range(0, len(cache), self.batch_size):
            batch = cache[batch_start : batch_start + self.batch_size]
            max_len = max(item[0].shape[0] for item in batch)
            bsz = len(batch)
            input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
            for i, (ids, _plen) in enumerate(batch):
                L = ids.shape[0]
                # Right-pad (so unpadded prefix sits at positions 0..L-1)
                input_ids[i, :L] = ids
                attention_mask[i, :L] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # keep in model dtype to avoid OOM on large vocab

            for i, (ids, plen) in enumerate(batch):
                L = ids.shape[0]
                if L - plen <= 0:
                    out.append(0.0)
                    continue
                # Response tokens are at positions plen..L-1; their context positions
                # (which produce the next-token distributions) are plen-1..L-2.
                token_ids = input_ids[i, plen:L]                    # (R,)
                ctx_log_probs = log_probs[i, plen - 1 : L - 1]      # (R, V)
                token_lps = ctx_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
                out.append(float(token_lps.sum().item()))
        return out
