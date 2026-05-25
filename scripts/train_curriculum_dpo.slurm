#!/bin/bash
#SBATCH --job-name=dpo_curriculum
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/train_curriculum_dpo_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/train_curriculum_dpo_%j.err

# Phase 2: DPO training on curriculum (easy→hard) dataset with train/test split.
# Run: MODEL_KEY=gemma2 sbatch scripts/train_curriculum_dpo.slurm [exp_id]
#   MODEL_KEY: llama3 (default), gemma2, qwen3
# Example: MODEL_KEY=gemma2 sbatch scripts/train_curriculum_dpo.slurm exp_004

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
SEQUENTIAL="${SEQUENTIAL:-0}"
STAGED="${STAGED:-0}"
N_STAGES="${N_STAGES:-3}"
COMPETENCE="${COMPETENCE:-0}"
COMPETENCE_C0="${COMPETENCE_C0:-0.01}"
COMPETENCE_PACING="${COMPETENCE_PACING:-sqrt}"
COMPETENCE_TOTAL_STEPS="${COMPETENCE_TOTAL_STEPS:-}"  # empty = auto (num_epochs × steps_per_epoch)
ORDERING="${ORDERING:-curr}"
SHUFFLING="${SHUFFLING:-sequential}"

SEQUENTIAL_FLAG=""
if [ "$SEQUENTIAL" = "1" ]; then
  SEQUENTIAL_FLAG="--sequential"
  echo ">>> Sequential mode ENABLED (preserving dataset order) <<<"
fi

STAGED_FLAG=""
if [ "$STAGED" = "1" ]; then
  STAGED_FLAG="--staged --n_stages $N_STAGES"
  echo ">>> Staged curriculum ENABLED ($N_STAGES stages, runtime shuffling) <<<"
fi

COMPETENCE_FLAG=""
if [ "$COMPETENCE" = "1" ]; then
  COMPETENCE_FLAG="--competence --competence_c0 $COMPETENCE_C0 --competence_pacing $COMPETENCE_PACING"
  if [ -n "$COMPETENCE_TOTAL_STEPS" ]; then
    COMPETENCE_FLAG="$COMPETENCE_FLAG --competence_total_steps $COMPETENCE_TOTAL_STEPS"
  fi
  echo ">>> Competence curriculum ENABLED (c0=$COMPETENCE_C0, pacing=$COMPETENCE_PACING, T=${COMPETENCE_TOTAL_STEPS:-auto}) <<<"
fi

echo "Training curriculum DPO (similarity): model=$MODEL_KEY dataset=$DATASET_KEY exp=$EXP_ID epochs=$NUM_EPOCHS lr=$LEARNING_RATE grad_accum=$GRAD_ACCUM sequential=$SEQUENTIAL"

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

env HF_HOME=/data/user_data/sandeep3/.cache/huggingface PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.phase2.train_dpo \
  --dataset_path "$PROCESSED_DIR/curriculum_dataset_train.jsonl" \
  --output_dir "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT/dpo_curriculum" \
  --model "$MODEL_ID" \
  --lora_target_modules $LORA_MODULES \
  --beta "$DPO_BETA" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  ${ATTN_IMPL:+--attn_implementation $ATTN_IMPL} \
  ${LOAD_4BIT:+--load_in_4bit} \
  $SEQUENTIAL_FLAG \
  $STAGED_FLAG \
  $COMPETENCE_FLAG \
  $EVAL_FLAGS

echo "Curriculum DPO training complete: $MODEL_KEY / $EXP_ID"

env PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.utils.write_experiment_meta \
  --exp-id "$EXP_ID" \
  --model-key "$MODEL_KEY" \
  --model-id "$MODEL_ID" \
  --model-short "$MODEL_SHORT" \
  --dataset-key "$DATASET_KEY" \
  --data-source-exp "${DATA_EXP_ID:-$EXP_ID}" \
  --ordering "$ORDERING" \
  --shuffling "$SHUFFLING" \
  --beta "$DPO_BETA" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --grad-accum "$GRAD_ACCUM" \
  --output-dir "$REPO_ROOT/outputs/$EXP_ID/$DATASET_KEY/$MODEL_SHORT"
