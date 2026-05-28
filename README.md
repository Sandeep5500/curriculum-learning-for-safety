# Curriculum Learning for Safety Alignment

Reference implementation for **Staged-Competence**, a curriculum-learning method for safety alignment of large language models via DPO. We also release **Cleaned-PKU-HH-SafeRLHF**, a curated safety preference dataset derived from PKU-SafeRLHF and HH-RLHF.

## Cleaned-PKU-HH-SafeRLHF

The released dataset combines the GPT-4o-mini-filtered PKU-SafeRLHF and HH-RLHF preference pairs, keeping only those where the **chosen** response is safe and the **rejected** response is unsafe. Format:

```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

| Source | Raw pairs | After filtering |
|--------|----------:|----------------:|
| PKU-SafeRLHF | 43,452 | 6,962 |
| HH-RLHF (single-turn) | 49,388 | 3,969 |
| **Combined** | 92,840 | **10,931** |

Train/test split (stratified by difficulty): 8,744 / 2,187.

Loading:
```python
import json
with open("data/processed/combined_pku_hh.jsonl") as f:
    pairs = [json.loads(line) for line in f]
```

To re-run cleaning from the raw sources (`data/raw/`), see `src/data/clean_with_gpt.py` and `src/data/create_staged_curriculum.py`.

## Setup

```bash
# Clone, create venv, install deps
python3 -m venv venv-new
source venv-new/bin/activate
pip install -r requirements.txt

# Required env vars
export HF_TOKEN=...           # for gated HF models
export OPENAI_API_KEY=...     # for GPT-4o-mini judge (data cleaning + safety eval)
```

The training and evaluation scripts are SLURM jobs that hard-code `REPO_ROOT=/data/user_data/sandeep3/curriculum-safety`. Replace this with your own checkout path in `scripts/submit_runs.sh` and every `scripts/*.slurm`, or set the variable before invoking.

## Pipeline

The pipeline has three phases. The main entry point is `scripts/submit_runs.sh`, which chains them with SLURM dependencies.

```bash
bash scripts/submit_runs.sh <MODEL_KEY> <DATASET_KEY> <EXP_ID> [NUM_EPOCHS]
# Example:
bash scripts/submit_runs.sh llama3 combined_pku_hh exp_001 5
```

Supported `MODEL_KEY` values (HuggingFace IDs in `scripts/model_configs.sh`):
- `llama3` — Llama-3-8B-Instruct (abliterated)
- `qwen3-8b-v3` — Josiefied-Qwen3-8B (abliterated)
- `yi15_9b` — Yi-1.5-9B-Chat (abliterated)
- `gemma3_4b` — Gemma-3-4B (abliterated)
- `qwen3_1b7`, `qwen3_4b` — scaling experiments

Supported `DATASET_KEY` values: `combined_pku_hh` (recommended), `pku_gpt`, `hh`.

### Phase 1: Difficulty scoring

Computes the **preference alignment margin** for each pair:
`m_i = cos(e_ŷ, e_y+) − cos(e_ŷ, e_y-)` using `all-MiniLM-L6-v2` embeddings, where `ŷ` is the unaligned model's zero-shot response. Sorts the training set easy-to-hard and emits `curriculum_dataset_train.jsonl` and `scored_dataset_test.jsonl`.

```bash
sbatch scripts/run_phase1_scoring.slurm <EXP_ID>
```

Generation hyperparameters: temperature=0.7, max_new_tokens=512 (see `src/phase1/score_easiness.py`).

### Phase 2: DPO training

Five methods, one SLURM script each. All accept `MODEL_KEY`, `DATASET_KEY`, `DATA_EXP_ID` via env vars:

| Method | Script | Description |
|--------|--------|-------------|
| Baseline DPO | `train_baseline_dpo.slurm` | Random shuffling, single stage, fixed reference |
| Sqrt-Pacing | `train_curriculum_dpo.slurm` (with `SEQUENTIAL=0 COMPETENCE=1`) | Competence-based sampling, single stage |
| Sequential | `train_curriculum_dpo.slurm` (with `SEQUENTIAL=1`) | Fixed easy-to-hard ordering each epoch |
| Curri-DPO | `train_curri_dpo.slurm` | K=3 stages with reference-model updates, shuffled within stages |
| **Staged-Competence** (paper) | `train_curri_pacing.slurm` | K=3 stages with reference-model updates + sqrt pacing within stages |

```bash
# Example: Staged-Competence on LLaMA-3, reusing phase-1 scores from exp_001
DATA_EXP_ID=exp_001 MODEL_KEY=llama3 DATASET_KEY=combined_pku_hh \
  sbatch scripts/train_curri_pacing.slurm exp_002
```

Common knobs (env vars): `DPO_BETA`, `LORA_MODULES`, `LEARNING_RATE`, `GRAD_ACCUM`, `NUM_EPOCHS`.

### Phase 3: Evaluation

| Benchmark | Script | Metric |
|-----------|--------|--------|
| AdvBench / SorryBench / HEx-PHI / XSTest | `run_safety_eval.slurm` + `run_safety_gpt_judge.slurm` | Harmful-response rate (↓), over-refusal rate (↓) |
| DPO test reward accuracy | `run_dpo_test_eval.slurm` | log π(y+) > log π(y−) (↑) |
| MMLU, HellaSwag | `run_quality_eval.slurm` | Accuracy (↑) |
| Alpaca over-refusal | `run_harmless_eval.slurm` | Comply rate on benign prompts (↑) |
| GCG transfer attack | `run_gcg_generation.slurm` → `run_jailbreak_inference.slurm` → `run_jailbreak_judge.slurm` | ASR (↓) |
| Prefill attack | `run_prefill_inference.slurm` → `run_prefill_judge.slurm` | ASR (↓) |
| Alignment depth | `run_alignment_depth.slurm` | Per-token suppression curves |

All evaluation scripts are re-runnable independently of training.

## Outputs

```
outputs/{exp_id}/{dataset}/{model}/
  dpo_baseline/, dpo_curriculum/    # LoRA adapters + checkpoints
  experiment.json                   # Hyperparams + method metadata
  eval_margin.jsonl                 # Per-step test reward margin

results/{exp_id}/{dataset}/{model}/
  phase2/dpo_test_*.json            # Reward accuracy
  phase3/{advbench,sorrybench,hexphi,xstest,alpaca,jailbreak,prefill}_*.json
  phase3/summary_gpt.json           # GPT-4o-mini judge summary
```

See `CLAUDE.md` for full developer details (known issues, gotchas, all hyperparameters).

## License

Code released under MIT. The Cleaned-PKU-HH-SafeRLHF dataset inherits the licenses of its sources:
PKU-SafeRLHF (CC BY-NC 4.0) and HH-RLHF (MIT).

## Citation
```bibtex
@misc{kumar2026curriculumlearningsafetyalignment,
      title={Curriculum Learning for Safety Alignment}, 
      author={Sandeep Kumar and Virginia Smith and Chhavi Yadav},
      year={2026},
      eprint={2605.26315},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.26315}, 
}
```
