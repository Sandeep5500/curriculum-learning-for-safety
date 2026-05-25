# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout & Execution Model

- All training and evaluation runs on a SLURM cluster. The cluster checkout path is hard-coded as `REPO_ROOT=/data/user_data/sandeep3/curriculum-safety` in `scripts/submit_runs.sh` and every `scripts/*.slurm` script. Local edits must be synced to that path before sbatch picks them up.
- SLURM scripts wrap Python entrypoints under `src/` invoked as modules (e.g. `python3 -m src.phase2.train_dpo ...`). Avoid running training scripts directly; always go through `sbatch`.
- `scripts/model_configs.sh` is sourced by every SLURM script and resolves `MODEL_KEY` → `MODEL_ID`, `MODEL_SHORT`, `LORA_MODULES`, `VENV_PATH`, `ATTN_IMPL`, plus `DATASET_KEY` → `DATASET_INPUT`, `PROCESSED_DIR`, `DEFAULT_EPOCHS`. Adding a new model or dataset means adding a `case` arm there.

## Pipeline Phases

```
src/phase1/   Score (prompt, chosen, rejected) by easy/hard, split 80/20, emit
              curriculum_dataset_train.jsonl + scored_dataset_test.jsonl.
src/phase2/   DPO training. Three Python entrypoints, one per training-method family:
                train_dpo.py         — baseline / sqrt-pacing / sequential (single-stage)
                train_curri_dpo.py   — multi-stage with reference model updates (shuffled)
                train_curri_pacing.py — multi-stage + sqrt pacing within stages
src/eval/     Phase 3 evaluation: safety judges (LlamaGuard / GPT-4o-mini), DPO test
              reward accuracy, jailbreak / GCG / prefill, training-loss + margin plots.
src/data/     Dataset preparation (PKU SafeRLHF, HH-RLHF, DOOR) → JSONL with
              prompt/chosen/rejected fields.
src/interpretability/  Probes (alignment_depth, attention_safety, probe_safety).
src/utils/    write_experiment_meta.py — emits experiment.json after every train run.
```

## Common Commands

```bash
# Full pipeline (phase1 → 2 baseline+curriculum → 3 evals) with SLURM dependencies
bash scripts/submit_runs.sh <MODEL_KEY> <DATASET_KEY> <EXP_ID> [NUM_EPOCHS]

# Reuse phase1 scoring from a prior experiment (skips the phase1 job)
DATA_EXP_ID=<prior_exp> bash scripts/submit_runs.sh <MODEL_KEY> <DATASET_KEY> <EXP_ID>

# Individual training methods (read MODEL_KEY/DATASET_KEY/DATA_EXP_ID from env)
sbatch scripts/train_baseline_dpo.slurm    <EXP_ID>           # baseline
SEQUENTIAL=0 COMPETENCE=1 sbatch scripts/train_curriculum_dpo.slurm <EXP_ID>   # sqrt-pacing
SEQUENTIAL=1              sbatch scripts/train_curriculum_dpo.slurm <EXP_ID>   # sequential
sbatch scripts/train_curri_dpo.slurm       <EXP_ID>           # curri-dpo (staged)
sbatch scripts/train_curri_pacing.slurm    <EXP_ID>           # curri-pacing (staged)

# Phase 3 evals (re-runnable independently of training)
sbatch scripts/run_safety_eval.slurm       <EXP_ID>   # AdvBench/SorryBench/HEx-PHI/XSTest
sbatch scripts/run_safety_gpt_judge.slurm  <EXP_ID>   # GPT-4o-mini judge → summary_gpt.json
sbatch scripts/run_dpo_test_eval.slurm     <EXP_ID>   # held-out reward accuracy
sbatch scripts/run_harmless_eval.slurm     <EXP_ID>   # Alpaca over-refusal
sbatch scripts/run_quality_eval.slurm      <EXP_ID>   # MMLU / HellaSwag

# Common knobs passed through as env vars
DPO_BETA=0.05 LORA_MODULES="q_proj v_proj k_proj o_proj" \
  LEARNING_RATE=5e-5 GRAD_ACCUM=16 NUM_EPOCHS=5 sbatch ...

# Monitor jobs
squeue -u $USER
```

## Training Methods

There are 5 distinct training methods. Each maps to a specific script and `shuffling` label.

| Method | Script | `shuffling` value | Description |
|--------|--------|-------------------|-------------|
| **Baseline DPO** | `train_baseline_dpo.slurm` | `baseline` | Randomly shuffled, 5 epochs |
| **Sqrt Pacing** | `train_curriculum_dpo.slurm` (sequential=0) | `sqrt_pacing` | Single-stage, competence sampling with sqrt pacing, same total steps as baseline |
| **Sequential** | `train_curriculum_dpo.slurm` (sequential=1) | `sequential` | Single-stage, fixed easy-to-hard ordering each epoch, 5 epochs |
| **Curri-DPO** | `train_curri_dpo.slurm` | `curri_dpo` | 3 stages, updated reference model between stages, **shuffled** within each stage |
| **Curri-Pacing** | `train_curri_pacing.slurm` | `curri_pacing` | 3 stages, updated reference model between stages, **sqrt pacing** within each stage |

**Key distinctions:**
- Sqrt Pacing vs Sequential: both use `train_curriculum_dpo.slurm`, distinguished by `sequential=0` vs `sequential=1`
- Curri-DPO vs Curri-Pacing: both are 3-stage with updated reference models; differ only in within-stage sampling (shuffled vs sqrt pacing)
- Curri-DPO and Curri-Pacing are the only methods that update the reference model between stages

## Model Registry

Defined in `scripts/model_configs.sh`. Current models:

| `MODEL_KEY` | Model ID | Notes |
|-------------|----------|-------|
| `llama3` | informatiker/Llama-3-8B-Instruct-abliterated | Main LLaMA model for paper |
| `gemma3_4b` | mlabonne/gemma-3-4b-it-abliterated | Main Gemma model for paper |
| `qwen3-8b-v3` | Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1 | **Main Qwen model for paper** (Josiefied abliteration) |
| `qwen3` | mlabonne/Qwen3-8B-abliterated | v1, sketchy generation quality |
| `qwen3_8b_v2` | huihui-ai/Huihui-Qwen3-8B-abliterated-v2 | v2, intermediate methods didn't outperform baseline |
| `gemma2_9b` | IlyaGusev/gemma-2-9b-it-abliterated | |
| `qwen3_4b` | mlabonne/Qwen3-4B-abliterated | |
| `qwen3_14b` | mlabonne/Qwen3-14B-abliterated | Needs A100 80GB (DPO model+ref >48GB) |
| `qwen3_14b_v2` | huihui-ai/Huihui-Qwen3-14B-abliterated-v2 | |
| `qwen3_1b7` | mlabonne/Qwen3-1.7B-abliterated | |
| `llama2_7b` | georgesung/llama2_7b_chat_uncensored | attn_implementation=eager |
| `gemma2_2b` | IlyaGusev/gemma-2-2b-it-abliterated | |
| `mistral3` | evolveon/Mistral-7B-Instruct-v0.3-abliterated | |
| `zephyr_7b` | richardyoung/zephyr-7b-beta-abliterated | |
| `llama31` | mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated | |
| `phi4` | huihui-ai/phi-4-abliterated | LoRA modules: `qkv_proj o_proj` (not q/v) |
| `ministral3_8b` | local snapshot | |
| `gemma4_e4b` | TrevorJS/gemma-4-E4B-it-uncensored | |

## Dataset Registry

| `DATASET_KEY` | File | Notes |
|---------------|------|-------|
| `pku` | data/processed/clean_parsed.jsonl | |
| `pku_gpt` | data/processed/clean_parsed_gpt.jsonl | |
| `door` | data/raw/door_full_parsed.jsonl | |
| `hh` | data/processed/hh_rlhf_clean_gpt.jsonl | |
| `combined_pku_hh` | data/processed/combined_pku_hh.jsonl | Main dataset used in recent experiments |

## Evaluation Benchmarks

| Benchmark | Metric | Notes |
|-----------|--------|-------|
| AdvBench | `violation_rate` (lower=safer) | 520 prompts, GPT-4o-mini judge |
| SorryBench | `violation_rate` (lower=safer) | 450 prompts, GPT-4o-mini judge |
| HEx-PHI | `violation_rate` (lower=safer) | 300 prompts, GPT-4o-mini judge |
| XSTest | `over_refusal_rate` (lower=better) | 250 prompts, refusal detector |
| DPO test | `reward_accuracy` (higher=better) | Held-out preference pairs |

## Result File Structure

```
results/{exp_id}/combined_pku_hh/{model}/
  phase2/
    dpo_test_baseline.json     # reward accuracy for baseline DPO adapter
    dpo_test_curriculum.json   # reward accuracy for curriculum adapter
    test_margin_curves.png
  phase3/
    advbench_{baseline,curriculum,raw}.json
    sorrybench_{...}.json
    hexphi_{...}.json
    xstest_{...}.json
    summary_gpt.json
```

## Output / Adapter Structure

```
outputs/{exp_id}/combined_pku_hh/{model}/
  dpo_baseline/          # Baseline DPO adapter (load on top of base model)
  dpo_curriculum/        # Curriculum adapter
    eval_margin.jsonl    # Per-step test reward margin (from TestMarginCallback)
    curri_base_model.txt # For Curri-DPO/Pacing: base model for the final adapter
    stage_0/merged/      # Merged checkpoint after stage 0 (base for stage 1)
    stage_1/merged/      # Merged checkpoint after stage 1 (base for stage 2)
  experiment.json        # Metadata (note: shuffling field has been wrong in some older runs)
```

## Known Issues / Gotchas

- **phi4 LoRA modules**: phi4 uses fused `qkv_proj o_proj`, NOT `q_proj v_proj`. Using the wrong modules causes `ValueError: Target modules not found`.
- **qwen3 OOM in TestMarginCallback**: qwen3 models (151k vocab) OOM if accelerate's fp32 cast is applied to logits. Fixed in `src/eval/test_margin_callback.py` by unwrapping accelerate's `ConvertOutputsToFp32` wrapper before the forward pass.
- **qwen3_14b needs A100**: DPO requires model + reference model in GPU memory (~56GB for 14B), exceeds A6000 48GB. Submit with `--gres=gpu:A100_80GB:1 --mem=80G`.
- **experiment.json shuffling field**: Some older runs have incorrect `shuffling` metadata. Always verify against the training log (grep for `sequential=`, `Curri-Pacing`, `curri_dpo`, etc.).
- **Curri-DPO staged base model**: When loading the final adapter for a Curri-DPO/Pacing run, the base model is NOT the original model — it's the merged checkpoint from the previous stage, stored in `curri_base_model.txt`.
- **LLaMA3 over-refusal**: Staged methods (Curri-DPO, Curri-Pacing) cause LLaMA3 to over-refuse on quality benchmarks (MMLU/HellaSwag), producing "I cannot answer that" for benign questions. This is LLaMA3-specific; Qwen3 and Gemma3 don't show this.
- **Repo path divergence**: `submit_runs.sh` and every `train_*.slurm` hard-code `REPO_ROOT=/data/user_data/sandeep3/curriculum-safety`. Local edits don't take effect until synced to that path on the cluster.
- **PYTHONPATH layering**: SLURM scripts prepend `$VENV_PATH/lib/python3.9/site-packages` and `~/.local/lib/python3.9/site-packages`. New dependencies must be installed into the chosen venv (default `venv-new`), not the user site, or jobs will resolve the wrong version.

## Final Experiment IDs (for paper)

### LLaMA3-8B (`llama3`)
| Method | Exp ID | DATA_EXP_ID |
|--------|--------|-------------|
| Baseline | exp_127 | exp_127 |
| Sqrt-Pacing | exp_100 | exp_073 |
| Sequential | exp_119 | exp_073 |
| Curri-DPO | exp_109 | exp_073 |
| Curri-Pacing | exp_128 | exp_127 |

### Qwen3-8B-v3 (`qwen3-8b-v3`)
| Method | Exp ID | DATA_EXP_ID |
|--------|--------|-------------|
| Baseline | exp_163 | exp_163 |
| Sqrt-Pacing | exp_167 | exp_163 |
| Sequential | exp_165 | exp_163 |
| Curri-DPO | exp_166 | exp_163 |
| Curri-Pacing | exp_164 | exp_163 |
| Curri-Pacing 50% | exp_168 | exp_168 |
| Curri-Pacing 75% | exp_169 | exp_169 |

### Gemma3-4B (`gemma3_4b`)
| Method | Exp ID | DATA_EXP_ID |
|--------|--------|-------------|
| Baseline | exp_129 | exp_129 |
| Sqrt-Pacing | exp_134 | exp_130 |
| Sequential | exp_162 | exp_130 |
| Curri-DPO | exp_149 | exp_130 |
| Curri-Pacing | exp_148 | exp_130 |
