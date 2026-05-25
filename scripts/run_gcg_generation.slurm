#!/bin/bash
#SBATCH --job-name=gcg_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --array=0-397
#SBATCH --output=/data/user_data/sandeep3/curriculum-safety/logs/gcg_gen_%A_%a.out
#SBATCH --error=/data/user_data/sandeep3/curriculum-safety/logs/gcg_gen_%A_%a.err

# Generate GCG adversarial suffixes for all 398 HarmBench behaviors against
# Qwen3-8B-Instruct. One SLURM array task per behavior.
#
# Usage:
#   sbatch scripts/run_gcg_generation.slurm
#
# Optional overrides (env vars):
#   MODEL=Qwen/Qwen3-8B  (default)
#   NUM_STEPS=500                  (default)
#   SEARCH_WIDTH=512               (default)

set -e
REPO_ROOT="/data/user_data/sandeep3/curriculum-safety"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_STEPS="${NUM_STEPS:-500}"
SEARCH_WIDTH="${SEARCH_WIDTH:-512}"
BEHAVIORS_FILE="$REPO_ROOT/data/harmbench/behaviors.json"
OUTPUT_DIR="$REPO_ROOT/data/harmbench/gcg_qwen3/individual"

# Look up behavior_id at this array index
BEHAVIOR_ID=$(python3 -c "
import json
ids = list(json.load(open('$BEHAVIORS_FILE')).keys())
print(ids[$SLURM_ARRAY_TASK_ID])
")

echo "=== GCG Generation ==="
echo "  Array task : $SLURM_ARRAY_TASK_ID / 397"
echo "  Behavior   : $BEHAVIOR_ID"
echo "  Model      : $MODEL"
echo "  Steps      : $NUM_STEPS"

module load cuda-12.1

VENV_PYTHON="$REPO_ROOT/venv-new/bin/python3"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME=/data/user_data/sandeep3/.cache/huggingface \
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt \
  SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt \
  PYTHONPATH=$REPO_ROOT \
  "$VENV_PYTHON" -m src.eval.generate_gcg \
    --behavior-id "$BEHAVIOR_ID" \
    --behaviors-file "$BEHAVIORS_FILE" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-steps "$NUM_STEPS" \
    --search-width "$SEARCH_WIDTH" \
    --seed 42

echo ""
echo "=== GCG complete: $BEHAVIOR_ID ==="
