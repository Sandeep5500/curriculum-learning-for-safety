#!/bin/bash
#SBATCH --job-name=phase1_score
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/phase1_scoring_%j.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/phase1_scoring_%j.err

# Phase 1: Generation and easiness scoring with train/test split (vLLM + sentence-transformer).
# Outputs train/test split (80/20) for fair comparison across curriculum strategies.
# Each model generates different responses → stored in model-specific subdirectory.
#
# Run: MODEL_KEY=gemma2 DATASET_KEY=pku sbatch scripts/run_phase1_scoring.slurm [exp_id]
#   MODEL_KEY: llama3 (default), gemma2, qwen3
#   DATASET_KEY: pku (default), door
#   exp_id: experiment ID (default: exp_004)
# Example: MODEL_KEY=llama3 DATASET_KEY=door sbatch scripts/run_phase1_scoring.slurm exp_007

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

EXP_ID="${1:-exp_004}"
MODEL_KEY="${MODEL_KEY:-llama3}"
DATASET_KEY="${DATASET_KEY:-pku}"
source "$REPO_ROOT/scripts/model_configs.sh"

DATA_SITE="$REPO_ROOT/$VENV_PATH/lib/python3.9/site-packages"
USER_SITE="/home/sandeep3/.local/lib/python3.9/site-packages"

# Auto-prepare DOOR data if not yet downloaded
if [ "$DATASET_KEY" = "door" ] && [ ! -f "$DATASET_INPUT" ]; then
  echo "DOOR dataset not found at $DATASET_INPUT, downloading..."
  env PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.data.prepare_door_alignment
fi

INPUT_JSONL="$DATASET_INPUT"
mkdir -p "$PROCESSED_DIR"

module load cuda-12.1

if [ ! -d "$REPO_ROOT/src" ]; then
  echo "ERROR: $REPO_ROOT/src not found"
  exit 1
fi

echo "=== Phase 1: Scoring & Curriculum Creation ==="
echo "Model: $MODEL_KEY ($MODEL_ID)"
echo "Input: $INPUT_JSONL"
echo "Output dir: $PROCESSED_DIR"
echo ""

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
echo "Step 1: Generating responses and computing similarity scores (batch_size=$GEN_BATCH_SIZE)..."
env PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" \
  HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m src.phase1.score_easiness \
  --input "$INPUT_JSONL" \
  --output "$PROCESSED_DIR/scored_dataset.jsonl" \
  --scorer sentence-transformer \
  --model "$MODEL_ID" \
  --batch-size "$GEN_BATCH_SIZE" \
  --split-train-ratio 0.8 \
  --split-seed 42

echo ""
echo "Step 2: Creating curriculum orderings from train split..."

# 1) Similarity-based curriculum (80% train split)
env PYTHONPATH="$REPO_ROOT:$DATA_SITE:$USER_SITE:$PYTHONPATH" python3 -m src.phase1.create_curriculum \
  --input-train "$PROCESSED_DIR/scored_dataset_train.jsonl" \
  --input-test "$PROCESSED_DIR/scored_dataset_test.jsonl" \
  --curriculum-train-out "$PROCESSED_DIR/curriculum_dataset_train.jsonl" \
  --curriculum-test-out "$PROCESSED_DIR/curriculum_dataset_test.jsonl" \
  --baseline-train-out "$PROCESSED_DIR/baseline_dataset_train.jsonl" \
  --baseline-test-out "$PROCESSED_DIR/baseline_dataset_test.jsonl" \
  --sort-key similarity

echo ""
echo "=== Phase 1 Complete for $MODEL_KEY ==="
echo "Outputs in $PROCESSED_DIR:"
echo "  - Similarity curriculum: curriculum_dataset_train/test.jsonl"
echo "  - Baseline: baseline_dataset_train/test.jsonl"
