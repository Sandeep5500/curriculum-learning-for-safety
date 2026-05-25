#!/bin/bash
# Submit a full pipeline run (Phase 1 → training → eval) for a given model + dataset.
#
# Usage:
#   bash scripts/submit_runs.sh <MODEL_KEY> <DATASET_KEY> <EXP_ID> [NUM_EPOCHS]
#
# Optional env vars:
#   DATA_EXP_ID   - reuse phase1 data from a different experiment (skips phase1 job)
#                   e.g. DATA_EXP_ID=exp_004 bash scripts/submit_runs.sh llama3 pku exp_007
#   DPO_BETA      - DPO beta (default: 0.1); e.g. DPO_BETA=0.05
#   LORA_MODULES  - space-separated LoRA targets (default: model-specific); e.g. "q_proj v_proj k_proj o_proj"
#
# Examples:
#   bash scripts/submit_runs.sh llama3 pku  exp_004
#   bash scripts/submit_runs.sh llama3 door exp_005
#   bash scripts/submit_runs.sh llama3 door exp_006 30
#   DATA_EXP_ID=exp_004 DPO_BETA=0.05 LORA_MODULES="q_proj v_proj k_proj o_proj" \
#     bash scripts/submit_runs.sh llama3 pku exp_007 5
#
# Jobs are submitted with SLURM dependencies so they run in order:
#   [phase1] → [baseline, curriculum] (parallel) → [safety_eval, dpo_test, harmless] (parallel)

set -e

MODEL_KEY="${1:-llama3}"
DATASET_KEY="${2:-pku}"
EXP_ID="${3:-exp_004}"

REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
SCRIPTS="$REPO_ROOT/scripts"

# Resolve dataset-specific epoch default (pku=3, door=5) unless caller overrides
source "$REPO_ROOT/scripts/model_configs.sh" > /dev/null
NUM_EPOCHS="${4:-$DEFAULT_EPOCHS}"

# DATA_EXP_ID: which phase1 data to use (defaults to EXP_ID = run phase1 fresh)
DATA_EXP_ID="${DATA_EXP_ID:-$EXP_ID}"

# DPO_BETA, LORA_MODULES, LEARNING_RATE, GRAD_ACCUM: passed through to training scripts
DPO_BETA="${DPO_BETA:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# LORA_MODULES already set by model_configs.sh above if not in env

echo "=== Submitting pipeline: model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID epochs=$NUM_EPOCHS ==="
echo "    data from: $DATA_EXP_ID | beta=$DPO_BETA | lr=$LEARNING_RATE | grad_accum=$GRAD_ACCUM | LoRA: $LORA_MODULES"

# GEN_BATCH_SIZE: smaller batch for large models to avoid OOM (default 4, use 2 for >=10B)
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"

# Phase 1: scoring + curriculum creation (skip if reusing existing data)
if [ "$DATA_EXP_ID" = "$EXP_ID" ]; then
  PHASE1_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
    sbatch --parsable "$SCRIPTS/run_phase1_scoring.slurm" "$EXP_ID")
  echo "Phase 1 (scoring):          job $PHASE1_JOB"
  PHASE1_DEP="afterok:$PHASE1_JOB"
else
  echo "Phase 1 (scoring):          SKIPPED (reusing data from $DATA_EXP_ID)"
  PHASE1_DEP=""
fi

# Phase 2: three training jobs in parallel
_dep_flag() { [ -n "$1" ] && echo "--dependency=$1"; }

BASELINE_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" NUM_EPOCHS="$NUM_EPOCHS" \
  DATA_EXP_ID="$DATA_EXP_ID" DPO_BETA="$DPO_BETA" LORA_MODULES="$LORA_MODULES" \
  LEARNING_RATE="$LEARNING_RATE" GRAD_ACCUM="$GRAD_ACCUM" \
  sbatch --parsable $(_dep_flag "$PHASE1_DEP") \
  "$SCRIPTS/train_baseline_dpo.slurm" "$EXP_ID")
echo "Phase 2 (baseline DPO):     job $BASELINE_JOB"

CURRICULUM_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" NUM_EPOCHS="$NUM_EPOCHS" \
  DATA_EXP_ID="$DATA_EXP_ID" DPO_BETA="$DPO_BETA" LORA_MODULES="$LORA_MODULES" \
  LEARNING_RATE="$LEARNING_RATE" GRAD_ACCUM="$GRAD_ACCUM" \
  sbatch --parsable $(_dep_flag "$PHASE1_DEP") \
  "$SCRIPTS/train_curriculum_dpo.slurm" "$EXP_ID")
echo "Phase 2 (curriculum DPO):   job $CURRICULUM_JOB"

# Phase 3: two eval jobs in parallel after all training finishes
TRAIN_DEPS="afterok:${BASELINE_JOB}:${CURRICULUM_JOB}"

SAFETY_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" DATA_EXP_ID="$DATA_EXP_ID" \
  sbatch --parsable --dependency="$TRAIN_DEPS" \
  "$SCRIPTS/run_safety_eval.slurm" "$EXP_ID")
echo "Phase 3a (safety eval):     job $SAFETY_JOB"

DPO_TEST_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" DATA_EXP_ID="$DATA_EXP_ID" \
  sbatch --parsable --dependency="$TRAIN_DEPS" \
  "$SCRIPTS/run_dpo_test_eval.slurm" "$EXP_ID")
echo "Phase 3b (dpo test + plot): job $DPO_TEST_JOB"

HARMLESS_JOB=$(MODEL_KEY="$MODEL_KEY" DATASET_KEY="$DATASET_KEY" \
  sbatch --parsable --dependency="$TRAIN_DEPS" \
  "$SCRIPTS/run_harmless_eval.slurm" "$EXP_ID")
echo "Phase 3c (harmless/alpaca): job $HARMLESS_JOB"

echo ""
echo "=== All jobs submitted for $MODEL_KEY / $DATASET_KEY / $EXP_ID (epochs=$NUM_EPOCHS) ==="
echo "Monitor with: squeue -u \$USER"
