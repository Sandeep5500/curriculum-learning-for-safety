#!/bin/bash
#SBATCH --job-name=dpo_baseline
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/train_baseline_dpo_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/train_baseline_dpo_%j.err

# Phase 2: DPO training on baseline (shuffled) dataset with train/test split.
# Run: MODEL_KEY=gemma2 sbatch scripts/train_baseline_dpo.slurm [exp_id]
#   MODEL_KEY: llama3 (default), gemma2, qwen3
#   exp_id: experiment ID (default: exp_004)
# Example: MODEL_KEY=qwen3 sbatch scripts/train_baseline_dpo.slurm exp_005

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_004}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
source "$REPO_ROOT/scripts/model_configs.sh"
NUM_EPOCHS="${NUM_EPOCHS:-$DEFAULT_EPOCHS}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

echo "Training baseline DPO: model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID epochs=$NUM_EPOCHS lr=$LEARNING_RATE grad_accum=$GRAD_ACCUM"

module load cuda-12.1

if [ ! -d "$REPO_ROOT/src" ]; then
  echo "ERROR: $REPO_ROOT/src not found"
  exit 1
fi

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

mkdir -p "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"

EVAL_TEST_SET="${EVAL_TEST_SET:-$PROCESSED_DIR/curriculum_dataset_test.jsonl}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-50}"
EVAL_FLAGS=""
if [ -f "$EVAL_TEST_SET" ]; then
  EVAL_FLAGS="--eval_test_path $EVAL_TEST_SET --eval_every_steps $EVAL_EVERY_STEPS"
  echo "Test-margin eval enabled: every $EVAL_EVERY_STEPS steps on $EVAL_TEST_SET"
fi
NO_SAVE_FLAG=""
if [ "${NO_SAVE_CHECKPOINTS:-0}" = "1" ]; then
  NO_SAVE_FLAG="--no_save_checkpoints"
  echo "Intermediate checkpoint saving DISABLED"
fi

env HF_HOME=/data/user_data/sandeep3/.cache/huggingface PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.phase2.train_dpo \
  --dataset_path "$PROCESSED_DIR/baseline_dataset_train.jsonl" \
  --output_dir "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/dpo_baseline" \
  --model "$MODEL_ID" \
  --lora_target_modules $LORA_MODULES \
  --beta "$DPO_BETA" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  ${ATTN_IMPL:+--attn_implementation $ATTN_IMPL} \
  ${LOAD_4BIT:+--load_in_4bit} \
  $EVAL_FLAGS \
  $NO_SAVE_FLAG

echo "Baseline DPO training complete: $MODEL_KEY / $EXP_ID"
