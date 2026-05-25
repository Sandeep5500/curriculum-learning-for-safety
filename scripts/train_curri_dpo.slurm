#!/bin/bash
#SBATCH --job-name=curri_dpo
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/train_curri_dpo_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/train_curri_dpo_%j.err

# Curri-DPO: multi-stage DPO with reference model updates between stages.
# Splits sorted data into N buckets (easy→hard), trains DPO on each,
# and updates the reference model after each stage.
#
# Usage:
#   MODEL_KEY=llama3 DATASET_KEY=combined_pku_hh GRAD_ACCUM=16 \
#     sbatch scripts/train_curri_dpo.slurm exp_107
#
# Env vars:
#   CURRI_STAGES       - number of difficulty stages (default: 3)
#   CURRI_EPOCHS       - epochs per stage (default: 5)
#   GRAD_ACCUM         - gradient accumulation steps (default: 16)
#   LEARNING_RATE      - peak learning rate (default: 5e-5)

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_107}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-combined_pku_hh}"
source "$REPO_ROOT/scripts/model_configs.sh"

CURRI_STAGES="${CURRI_STAGES:-3}"
CURRI_EPOCHS="${CURRI_EPOCHS:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

echo "=== Curri-DPO Training ==="
echo "  model=$MODEL_KEY  dataset=$DATASET_KEY  exp=$EXP_ID"
echo "  stages=$CURRI_STAGES  epochs/stage=$CURRI_EPOCHS  grad_accum=$GRAD_ACCUM  lr=$LEARNING_RATE"

module load cuda-12.1

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

env HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
    PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
  python3 -m src.phase2.train_curri_dpo \
    --dataset_path "$PROCESSED_DIR/curriculum_dataset_train.jsonl" \
    --output_dir "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/dpo_curriculum" \
    --model "$MODEL_ID" \
    --n_stages "$CURRI_STAGES" \
    --epochs_per_stage "$CURRI_EPOCHS" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$DPO_BETA" \
    --lora_target_modules $LORA_MODULES \
    ${ATTN_IMPL:+--attn_implementation $ATTN_IMPL} \
    $EVAL_FLAGS

echo "Curri-DPO training complete: $MODEL_KEY / $EXP_ID"

env PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.utils.write_experiment_meta \
  --exp-id "$EXP_ID" \
  --model-key "$MODEL_KEY" \
  --model-id "$MODEL_ID" \
  --model-short "$MODEL_SHORT" \
  --dataset-key "$DATASET_KEY" \
  --data-source-exp "${DATA_EXP_ID:-$EXP_ID}" \
  --ordering "curr" \
  --shuffling "curri_dpo" \
  --beta "$DPO_BETA" \
  --num-epochs "$CURRI_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --grad-accum "$GRAD_ACCUM" \
  --output-dir "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"

echo "Wrote experiment metadata"
